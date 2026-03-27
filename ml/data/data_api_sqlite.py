import time
import sqlite3
import signal
from riotwatcher import LolWatcher, RiotWatcher, ApiError
from ml.data.match_format import try_build_ordered_match_record
from ml.data.match_storage import (
    connect_sqlite,
    ensure_match_schema,
    extract_storage_payload,
    upsert_match_record,
)
from ml.runtime_config import (
    get_api_key,
    get_batch_size,
    get_collector_id,
    get_db_path,
    get_queue_id,
    get_scraper_targets,
    get_start_time,
    get_status_interval_sec,
    load_runtime_env,
)

load_runtime_env()
KEY = get_api_key()
DB_PATH = get_db_path()
COLLECTOR_ID = get_collector_id()
BATCH_SIZE = get_batch_size()
QUEUE_ID = get_queue_id()
START_TIME = get_start_time()
STATUS_INTERVAL_SEC = get_status_interval_sec()
TARGETS = get_scraper_targets()

lol_watcher = None
riot_watcher = None
keep_running = True
stop_requests = 0
runtime_stats = {
    "matches_saved": 0,
    "players_scanned": 0,
    "api_calls": 0,
}


def increment_api_calls():
    runtime_stats["api_calls"] += 1


def ensure_watchers():
    global lol_watcher, riot_watcher

    if lol_watcher is not None and riot_watcher is not None:
        return lol_watcher, riot_watcher

    if not KEY:
        raise ValueError("RIOT_API_KEY not found in environment.")

    lol_watcher = LolWatcher(KEY)
    riot_watcher = RiotWatcher(KEY)
    return lol_watcher, riot_watcher

def signal_handler(sig, frame):
    global keep_running, stop_requests
    stop_requests += 1
    if stop_requests == 1:
        print("\n[STOP] Stopping safely... Press Ctrl+C again to exit immediately.")
        keep_running = False
        return

    print("\n[STOP] Force exiting now.")
    signal.default_int_handler(sig, frame)

signal.signal(signal.SIGINT, signal_handler)


def print_status(conn, started_at, final=False):
    total_matches = conn.execute("SELECT count(*) FROM matches").fetchone()[0]
    elapsed = time.time() - started_at
    label = "FINAL" if final else "STATUS"
    print(
        f"[{label}] elapsed={elapsed/60:.1f}m | "
        f"saved_this_run={runtime_stats['matches_saved']:,} | "
        f"players_scanned={runtime_stats['players_scanned']:,} | "
        f"api_calls={runtime_stats['api_calls']:,} | "
        f"total_matches={total_matches:,}"
    )

def init_db():
    conn = connect_sqlite(DB_PATH)
    c = conn.cursor()
    ensure_match_schema(conn)

    # Create queues for each platform
    for t in TARGETS:
        plat = t['platform']
        c.execute(f"CREATE TABLE IF NOT EXISTS match_queue_{plat} (match_id TEXT PRIMARY KEY)")
        c.execute(f"CREATE TABLE IF NOT EXISTS puuid_queue_{plat} (puuid TEXT PRIMARY KEY)")
        c.execute(f"CREATE TABLE IF NOT EXISTS processed_puuids_{plat} (puuid TEXT PRIMARY KEY)")

    conn.commit()
    conn.close()

def seed_if_needed(conn, target):
    _, local_riot_watcher = ensure_watchers()
    plat = target['platform']
    region = target['region']

    # Check if this region needs a seed
    c = conn.cursor()
    c.execute(f"SELECT count(*) FROM puuid_queue_{plat}")
    p_count = c.fetchone()[0]
    c.execute(f"SELECT count(*) FROM match_queue_{plat}")
    m_count = c.fetchone()[0]

    if p_count == 0 and m_count == 0:
        print(f"[{plat}] Queue empty. Seeding {target['seed_name']}...", end=" ")
        try:
            acct = local_riot_watcher.account.by_riot_id(region, target['seed_name'], target['seed_tag'])
            increment_api_calls()
            pid = acct['puuid']
            c.execute(f"INSERT OR IGNORE INTO puuid_queue_{plat} VALUES (?)", (pid,))
            conn.commit()
            print("Done.")
        except Exception as e:
            print(f"Failed: {e}")

def process_region(conn, target):
    """Does ONE unit of work for ONE region."""
    local_lol_watcher, _ = ensure_watchers()
    c = conn.cursor()
    plat = target['platform']
    region = target['region']

    # 1. Try to process a MATCH
    c.execute(f"SELECT match_id FROM match_queue_{plat} LIMIT 1")
    row = c.fetchone()

    if row:
        mid = row[0]
        c.execute(f"DELETE FROM match_queue_{plat} WHERE match_id=?", (mid,))

        # Skip fully stored matches, but allow refreshes for older/incomplete rows.
        c.execute("SELECT raw_match_json, ordered_match_json FROM matches WHERE match_id=?", (mid,))
        existing = c.fetchone()
        has_full_match = bool(existing and existing[0] and existing[1])
        if has_full_match:
            return False # No API call made

        try:
            match = local_lol_watcher.match.by_id(region, mid)
            increment_api_calls()
            info = match['info']

            match_data, reason = try_build_ordered_match_record(info)
            if match_data is None:
                print(f"[{plat}] Skipped {mid}: {reason}")
            else:
                payload = extract_storage_payload(
                    match,
                    plat,
                    match_data,
                    COLLECTOR_ID,
                )
                upsert_match_record(conn, mid, payload)
                runtime_stats["matches_saved"] += 1

            # Harvest PUUIDs
            for p in info['participants']:
                pid = p['puuid']
                # Check if processed
                c.execute(f"SELECT 1 FROM processed_puuids_{plat} WHERE puuid=?", (pid,))
                if not c.fetchone():
                    c.execute(f"INSERT OR IGNORE INTO puuid_queue_{plat} VALUES (?)", (pid,))

            if match_data is not None:
                print(f"[{plat}] Saved {mid}")
            return True # 1 API call made

        except ApiError as e:
            if e.response.status_code == 429:
                print(f"[{plat}] Rate Limit 429 (Match). Skipping...")
                c.execute(f"INSERT OR IGNORE INTO match_queue_{plat} VALUES (?)", (mid,))
                return False
            elif e.response.status_code == 404:
                print(f"[{plat}] Match {mid} missing.")
                return False
            else:
                print(f"[{plat}] Error {mid}: {e}")
                return False

    # 2. If no matches, try to process a PLAYER
    c.execute(f"SELECT puuid FROM puuid_queue_{plat} LIMIT 1")
    row = c.fetchone()

    if row:
        pid = row[0]
        c.execute(f"DELETE FROM puuid_queue_{plat} WHERE puuid=?", (pid,))
        c.execute(f"INSERT OR IGNORE INTO processed_puuids_{plat} VALUES (?)", (pid,))

        try:
            history = local_lol_watcher.match.matchlist_by_puuid(region, pid, queue=QUEUE_ID, start_time=START_TIME, count=100)
            increment_api_calls()

            new_count = 0
            for m in history:
                c.execute("SELECT 1 FROM matches WHERE match_id=?", (m,))
                if not c.fetchone():
                    c.execute(f"INSERT OR IGNORE INTO match_queue_{plat} VALUES (?)", (m,))
                    new_count += 1

            runtime_stats["players_scanned"] += 1
            print(f"[{plat}] Scanned Player -> {new_count} matches")
            return True

        except ApiError as e:
            if e.response.status_code == 429:
                print(f"[{plat}] Rate Limit 429 (Player). Skipping...")
                c.execute(f"INSERT OR IGNORE INTO puuid_queue_{plat} VALUES (?)", (pid,))
                return False
            else:
                print(f"[{plat}] Error Player: {e}")
                return False

    return False # Nothing to do

def main():
    ensure_watchers()
    init_db()
    conn = connect_sqlite(DB_PATH)
    started_at = time.time()
    last_status_at = started_at

    try:
        # Initial Seeding
        for t in TARGETS:
            seed_if_needed(conn, t)

        print("Starting Round-Robin Scraper...")
        ops = 0

        while keep_running:
            # Loop through regions one by one
            for target in TARGETS:
                if not keep_running:
                    break

                # Do work for this region
                did_work = process_region(conn, target)

                if did_work:
                    ops += 1

            # Save periodically
            if ops >= BATCH_SIZE:
                conn.commit()
                print("--- Committed Batch ---")
                print_status(conn, started_at)
                last_status_at = time.time()
                ops = 0

            now = time.time()
            if now - last_status_at >= STATUS_INTERVAL_SEC:
                print_status(conn, started_at)
                last_status_at = now

            # Small sleep to prevent CPU spinning if queues are empty/rate limited
            # Also helps smooth out the requests
            time.sleep(0.1)
    finally:
        conn.commit()
        print_status(conn, started_at, final=True)
        conn.close()
        print("Exited.")

if __name__ == "__main__":
    main()
