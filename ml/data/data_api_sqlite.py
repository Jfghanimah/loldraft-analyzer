import time
import sqlite3
import signal
from riotwatcher import LolWatcher, RiotWatcher, ApiError
from requests.exceptions import RequestException
from ml.data.compact_parquet import CompactParquetWriter, get_compact_dataset_dir
from ml.data.compact_records import extract_compact_records
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
    get_storage_mode,
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
STORAGE_MODE = get_storage_mode()
COMPACT_DATASET_DIR = get_compact_dataset_dir()

lol_watcher = None
riot_watcher = None
keep_running = True
stop_requests = 0
runtime_stats = {
    "matches_saved": 0,
    "players_scanned": 0,
    "api_calls": 0,
}
AUTH_ERROR_STATUS_CODES = {401, 403}
TRANSIENT_REQUEST_SLEEP_SEC = 5


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
    total_matches = count_seen_matches(conn)
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
    ensure_collector_state_schema(conn)

    # Create queues for each platform
    for t in TARGETS:
        plat = t['platform']
        c.execute(f"CREATE TABLE IF NOT EXISTS match_queue_{plat} (match_id TEXT PRIMARY KEY)")
        c.execute(f"CREATE TABLE IF NOT EXISTS puuid_queue_{plat} (puuid TEXT PRIMARY KEY)")
        c.execute(f"CREATE TABLE IF NOT EXISTS processed_puuids_{plat} (puuid TEXT PRIMARY KEY)")

    conn.commit()
    conn.close()


def ensure_collector_state_schema(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seen_matches (
            match_id TEXT PRIMARY KEY,
            platform TEXT,
            storage_mode TEXT,
            saved_at_ts INTEGER
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_seen_matches_platform ON seen_matches(platform)")


def count_seen_matches(conn):
    try:
        row = conn.execute(
            """
            SELECT count(*) FROM (
                SELECT match_id FROM seen_matches
                UNION
                SELECT match_id FROM matches
            )
            """
        ).fetchone()
        return int(row[0] or 0) if row else 0
    except sqlite3.OperationalError:
        pass
    try:
        return int(conn.execute("SELECT count(*) FROM matches").fetchone()[0])
    except sqlite3.OperationalError:
        return 0


def has_seen_match(conn, match_id):
    row = conn.execute("SELECT 1 FROM seen_matches WHERE match_id = ?", (match_id,)).fetchone()
    if row:
        return True
    row = conn.execute("SELECT 1 FROM matches WHERE match_id = ?", (match_id,)).fetchone()
    return bool(row)


def mark_seen_match(conn, match_id, platform, storage_mode):
    conn.execute(
        """
        INSERT OR REPLACE INTO seen_matches (match_id, platform, storage_mode, saved_at_ts)
        VALUES (?, ?, ?, ?)
        """,
        (match_id, platform, storage_mode, int(time.time())),
    )


def requeue_match(conn, platform, match_id):
    conn.execute(f"INSERT OR IGNORE INTO match_queue_{platform} VALUES (?)", (match_id,))


def requeue_puuid(conn, platform, puuid):
    conn.execute(f"DELETE FROM processed_puuids_{platform} WHERE puuid = ?", (puuid,))
    conn.execute(f"INSERT OR IGNORE INTO puuid_queue_{platform} VALUES (?)", (puuid,))


def handle_auth_api_error(conn, platform, queue_kind, value, error):
    global keep_running

    if queue_kind == "match":
        requeue_match(conn, platform, value)
    elif queue_kind == "puuid":
        requeue_puuid(conn, platform, value)
    else:
        raise ValueError(f"Unsupported queue_kind={queue_kind!r}")

    keep_running = False
    print(
        f"[{platform}] API auth error while fetching {queue_kind} {value}: {error}. "
        "Requeued and stopping; refresh RIOT_API_KEY before restarting."
    )


def handle_transient_request_error(conn, platform, queue_kind, value, error):
    if queue_kind == "match":
        requeue_match(conn, platform, value)
    elif queue_kind == "puuid":
        requeue_puuid(conn, platform, value)
    else:
        raise ValueError(f"Unsupported queue_kind={queue_kind!r}")

    print(f"[{platform}] Transient network error while fetching {queue_kind} {value}: {error}. Requeued.")
    time.sleep(TRANSIENT_REQUEST_SLEEP_SEC)


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

def store_match(conn, match_id, match, platform, ordered_match_data, compact_writer=None):
    if STORAGE_MODE == "sqlite":
        payload = extract_storage_payload(
            match,
            platform,
            ordered_match_data,
            COLLECTOR_ID,
        )
        upsert_match_record(conn, match_id, payload)
        mark_seen_match(conn, match_id, platform, "sqlite")
        return
    if STORAGE_MODE != "compact":
        raise ValueError(f"Unsupported LOL_DRAFT_STORAGE_MODE={STORAGE_MODE!r}. Use 'compact' or 'sqlite'.")
    batch, reason = extract_compact_records(match, platform, collector_id=COLLECTOR_ID)
    if batch is None:
        raise ValueError(reason or f"Could not extract compact records for {match_id}")
    created_writer = compact_writer is None
    if compact_writer is None:
        compact_writer = CompactParquetWriter(COMPACT_DATASET_DIR)
    compact_writer.add_batch(batch)
    if created_writer:
        compact_writer.flush()
    mark_seen_match(conn, match_id, platform, "compact")


def flush_compact_writer(compact_writer):
    if compact_writer is None or STORAGE_MODE != "compact":
        return []
    return compact_writer.flush()


def process_region(conn, target, compact_writer=None):
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

        if has_seen_match(conn, mid):
            return False # No API call made

        try:
            match = local_lol_watcher.match.by_id(region, mid)
            increment_api_calls()
            info = match['info']

            match_data, reason = try_build_ordered_match_record(info)
            if match_data is None:
                print(f"[{plat}] Skipped {mid}: {reason}")
            else:
                store_match(conn, mid, match, plat, match_data, compact_writer=compact_writer)
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
                requeue_match(conn, plat, mid)
                return False
            elif e.response.status_code in AUTH_ERROR_STATUS_CODES:
                handle_auth_api_error(conn, plat, "match", mid, e)
                return False
            elif e.response.status_code == 404:
                print(f"[{plat}] Match {mid} missing.")
                return False
            else:
                print(f"[{plat}] Error {mid}: {e}")
                return False
        except RequestException as e:
            handle_transient_request_error(conn, plat, "match", mid, e)
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
                if not has_seen_match(conn, m):
                    c.execute(f"INSERT OR IGNORE INTO match_queue_{plat} VALUES (?)", (m,))
                    new_count += 1

            runtime_stats["players_scanned"] += 1
            print(f"[{plat}] Scanned Player -> {new_count} matches")
            return True

        except ApiError as e:
            if e.response.status_code == 429:
                print(f"[{plat}] Rate Limit 429 (Player). Skipping...")
                requeue_puuid(conn, plat, pid)
                return False
            elif e.response.status_code in AUTH_ERROR_STATUS_CODES:
                handle_auth_api_error(conn, plat, "puuid", pid, e)
                return False
            else:
                print(f"[{plat}] Error Player: {e}")
                return False
        except RequestException as e:
            handle_transient_request_error(conn, plat, "puuid", pid, e)
            return False

    return False # Nothing to do

def main():
    ensure_watchers()
    init_db()
    conn = connect_sqlite(DB_PATH)
    ensure_collector_state_schema(conn)
    compact_writer = CompactParquetWriter(COMPACT_DATASET_DIR) if STORAGE_MODE == "compact" else None
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
                did_work = process_region(conn, target, compact_writer=compact_writer)

                if did_work:
                    ops += 1

            # Save periodically
            if ops >= BATCH_SIZE:
                flush_compact_writer(compact_writer)
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
        flush_compact_writer(compact_writer)
        conn.commit()
        print_status(conn, started_at, final=True)
        conn.close()
        print("Exited.")

if __name__ == "__main__":
    main()
