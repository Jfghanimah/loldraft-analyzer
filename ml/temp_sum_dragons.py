import argparse
import json
import time

from ml.data.match_storage import connect_sqlite
from ml.runtime_config import get_db_path, load_runtime_env

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None


PROGRESS_EVERY = 1000


def _loads_json(payload):
    if orjson is not None:
        return orjson.loads(payload)
    return json.loads(payload)


def _configure_read_connection(conn):
    pragmas = (
        "PRAGMA cache_size = -65536",
        "PRAGMA mmap_size = 0",
        "PRAGMA read_uncommitted = ON",
        "PRAGMA temp_store = MEMORY",
    )
    for pragma in pragmas:
        try:
            conn.execute(pragma)
        except Exception:
            continue


def _count_matches(conn, queue_id):
    if queue_id is None:
        row = conn.execute(
            """
            SELECT count(*)
            FROM matches
            WHERE raw_match_json IS NOT NULL
            """
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT count(*)
            FROM matches
            WHERE raw_match_json IS NOT NULL
              AND queue_id = ?
            """,
            (queue_id,),
        ).fetchone()
    return int(row[0] or 0) if row else 0


def _iter_raw_matches(conn, queue_id):
    if queue_id is None:
        return conn.execute(
            """
            SELECT match_id, raw_match_json
            FROM matches
            WHERE raw_match_json IS NOT NULL
            ORDER BY game_creation, match_id
            """
        )
    return conn.execute(
        """
        SELECT match_id, raw_match_json
        FROM matches
        WHERE raw_match_json IS NOT NULL
          AND queue_id = ?
        ORDER BY game_creation, match_id
        """,
        (queue_id,),
    )


def main():
    parser = argparse.ArgumentParser(description="Temporary read-only dragon counter for the training DB.")
    parser.add_argument("--db-path", default=None, help="SQLite DB path (defaults to LOL_DRAFT_DB_PATH or league_data.db)")
    parser.add_argument("--queue-id", type=int, default=420, help="Queue ID filter (default: 420, use no flag for solo queue)")
    parser.add_argument("--progress-every", type=int, default=PROGRESS_EVERY, help="Print progress every N matches")
    args = parser.parse_args()

    load_runtime_env()
    db_path = args.db_path or get_db_path()
    conn = connect_sqlite(db_path, read_only=True)
    _configure_read_connection(conn)

    count_started_at = time.time()
    total_matches = _count_matches(conn, args.queue_id)
    count_elapsed = time.time() - count_started_at

    cursor_started_at = time.time()
    cursor = _iter_raw_matches(conn, args.queue_id)
    cursor_elapsed = time.time() - cursor_started_at

    blue_dragons = 0
    red_dragons = 0
    parse_failures = 0
    started_at = time.time()
    fetch_elapsed = 0.0
    parse_elapsed = 0.0
    objective_elapsed = 0.0

    print(f"db={db_path}")
    print(f"queue_id={args.queue_id}")
    print(f"matches_with_raw_json={total_matches:,}")
    print(f"count_seconds={count_elapsed:.4f}")
    print(f"cursor_open_seconds={cursor_elapsed:.4f}")
    print("starting dragon scan...")

    index = 0
    cursor_iter = iter(cursor)
    while True:
        fetch_started_at = time.time()
        try:
            match_id, raw_match_json = next(cursor_iter)
        except StopIteration:
            fetch_elapsed += time.time() - fetch_started_at
            break
        fetch_elapsed += time.time() - fetch_started_at
        index += 1

        parse_started_at = time.time()
        try:
            match = _loads_json(raw_match_json)
        except Exception:
            parse_elapsed += time.time() - parse_started_at
            parse_failures += 1
            continue
        parse_elapsed += time.time() - parse_started_at

        objective_started_at = time.time()
        teams = {team.get("teamId"): team for team in match.get("info", {}).get("teams", [])}
        blue_dragons += int(teams.get(100, {}).get("objectives", {}).get("dragon", {}).get("kills", 0) or 0)
        red_dragons += int(teams.get(200, {}).get("objectives", {}).get("dragon", {}).get("kills", 0) or 0)
        objective_elapsed += time.time() - objective_started_at

        if index % args.progress_every == 0 or index == total_matches:
            elapsed = time.time() - started_at
            rate = index / elapsed if elapsed > 0 else 0.0
            print(
                f"processed={index:,}/{total_matches:,} "
                f"elapsed={elapsed:.1f}s "
                f"rate={rate:,.1f} matches/s "
                f"fetch={fetch_elapsed:.1f}s "
                f"parse={parse_elapsed:.1f}s "
                f"objectives={objective_elapsed:.1f}s "
                f"blue_dragons={blue_dragons:,} "
                f"red_dragons={red_dragons:,} "
                f"failures={parse_failures:,}"
            )

    elapsed = time.time() - started_at
    total_dragons = blue_dragons + red_dragons
    rate = total_matches / elapsed if elapsed > 0 else 0.0
    print("")
    print("done")
    print(f"elapsed_seconds={elapsed:.2f}")
    print(f"matches_per_second={rate:,.1f}")
    print(f"sqlite_fetch_seconds={fetch_elapsed:.2f}")
    print(f"json_parse_seconds={parse_elapsed:.2f}")
    print(f"objective_extract_seconds={objective_elapsed:.2f}")
    print(f"blue_dragons={blue_dragons:,}")
    print(f"red_dragons={red_dragons:,}")
    print(f"total_dragons={total_dragons:,}")
    print(f"parse_failures={parse_failures:,}")

    conn.close()


if __name__ == "__main__":
    main()
