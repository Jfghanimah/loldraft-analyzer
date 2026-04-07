import argparse
import json
import time

from ml.data.match_storage import connect_sqlite
from ml.runtime_config import get_db_path, load_runtime_env

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None


DEFAULT_PROGRESS_EVERY = 1000
DEFAULT_QUEUE_ID = 420


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


def _format_seconds(seconds):
    return f"{seconds:.2f}s"


def _print_summary(name, rows, started_at, fetch_elapsed, parse_elapsed, work_elapsed, extra=None):
    elapsed = time.time() - started_at
    rate = rows / elapsed if elapsed > 0 else 0.0
    print("")
    print(f"[{name}] done")
    print(f"[{name}] rows={rows:,}")
    print(f"[{name}] elapsed={_format_seconds(elapsed)}")
    print(f"[{name}] rows_per_second={rate:,.1f}")
    print(f"[{name}] sqlite_fetch_seconds={fetch_elapsed:.2f}")
    print(f"[{name}] json_parse_seconds={parse_elapsed:.2f}")
    print(f"[{name}] work_seconds={work_elapsed:.2f}")
    if extra:
        for key, value in extra.items():
            print(f"[{name}] {key}={value}")


def _progress(name, rows, total_rows, started_at, fetch_elapsed, parse_elapsed, work_elapsed, progress_every):
    if rows % progress_every != 0 and rows != total_rows:
        return
    elapsed = time.time() - started_at
    rate = rows / elapsed if elapsed > 0 else 0.0
    print(
        f"[{name}] processed={rows:,}/{total_rows:,} "
        f"elapsed={elapsed:.1f}s "
        f"rate={rate:,.1f} rows/s "
        f"fetch={fetch_elapsed:.1f}s "
        f"parse={parse_elapsed:.1f}s "
        f"work={work_elapsed:.1f}s"
    )


def _count_rows(conn, where_sql="", params=()):
    query = "SELECT count(*) FROM matches"
    if where_sql:
        query += f" WHERE {where_sql}"
    row = conn.execute(query, params).fetchone()
    return int(row[0] or 0) if row else 0


def _count_participant_rows(conn, queue_id):
    if queue_id is None:
        row = conn.execute("SELECT count(*) FROM participant_history").fetchone()
    else:
        row = conn.execute("SELECT count(*) FROM participant_history WHERE queue_id = ?", (queue_id,)).fetchone()
    return int(row[0] or 0) if row else 0


def benchmark_raw_json_parse(conn, queue_id, progress_every):
    if queue_id is None:
        total_rows = _count_rows(conn, "raw_match_json IS NOT NULL")
        cursor = conn.execute(
            """
            SELECT raw_match_json
            FROM matches
            WHERE raw_match_json IS NOT NULL
            ORDER BY game_creation, match_id
            """
        )
    else:
        total_rows = _count_rows(conn, "raw_match_json IS NOT NULL AND queue_id = ?", (queue_id,))
        cursor = conn.execute(
            """
            SELECT raw_match_json
            FROM matches
            WHERE raw_match_json IS NOT NULL
              AND queue_id = ?
            ORDER BY game_creation, match_id
            """,
            (queue_id,),
        )

    rows = 0
    fetch_elapsed = 0.0
    parse_elapsed = 0.0
    work_elapsed = 0.0
    blue_dragons = 0
    red_dragons = 0
    started_at = time.time()
    cursor_iter = iter(cursor)

    while True:
        fetch_started_at = time.time()
        try:
            (raw_match_json,) = next(cursor_iter)
        except StopIteration:
            fetch_elapsed += time.time() - fetch_started_at
            break
        fetch_elapsed += time.time() - fetch_started_at
        rows += 1

        parse_started_at = time.time()
        match = _loads_json(raw_match_json)
        parse_elapsed += time.time() - parse_started_at

        work_started_at = time.time()
        teams = {team.get("teamId"): team for team in match.get("info", {}).get("teams", [])}
        blue_dragons += int(teams.get(100, {}).get("objectives", {}).get("dragon", {}).get("kills", 0) or 0)
        red_dragons += int(teams.get(200, {}).get("objectives", {}).get("dragon", {}).get("kills", 0) or 0)
        work_elapsed += time.time() - work_started_at
        _progress("raw_json_parse", rows, total_rows, started_at, fetch_elapsed, parse_elapsed, work_elapsed, progress_every)

    _print_summary(
        "raw_json_parse",
        rows,
        started_at,
        fetch_elapsed,
        parse_elapsed,
        work_elapsed,
        extra={
            "blue_dragons": f"{blue_dragons:,}",
            "red_dragons": f"{red_dragons:,}",
        },
    )


def benchmark_ordered_json_parse(conn, queue_id, progress_every):
    if queue_id is None:
        total_rows = _count_rows(conn, "ordered_match_json IS NOT NULL")
        cursor = conn.execute(
            """
            SELECT ordered_match_json, region, game_version
            FROM matches
            WHERE ordered_match_json IS NOT NULL
            ORDER BY game_creation, match_id
            """
        )
    else:
        total_rows = _count_rows(conn, "ordered_match_json IS NOT NULL AND queue_id = ?", (queue_id,))
        cursor = conn.execute(
            """
            SELECT ordered_match_json, region, game_version
            FROM matches
            WHERE ordered_match_json IS NOT NULL
              AND queue_id = ?
            ORDER BY game_creation, match_id
            """,
            (queue_id,),
        )

    rows = 0
    fetch_elapsed = 0.0
    parse_elapsed = 0.0
    work_elapsed = 0.0
    champion_slots = 0
    region_nonempty = 0
    started_at = time.time()
    cursor_iter = iter(cursor)

    while True:
        fetch_started_at = time.time()
        try:
            ordered_match_json, region, game_version = next(cursor_iter)
        except StopIteration:
            fetch_elapsed += time.time() - fetch_started_at
            break
        fetch_elapsed += time.time() - fetch_started_at
        rows += 1

        parse_started_at = time.time()
        ordered = _loads_json(ordered_match_json)
        parse_elapsed += time.time() - parse_started_at

        work_started_at = time.time()
        champion_slots += len(ordered.get("champions", []))
        if region:
            region_nonempty += 1
        _ = game_version or ordered.get("game_version", "")
        work_elapsed += time.time() - work_started_at
        _progress("ordered_json_parse", rows, total_rows, started_at, fetch_elapsed, parse_elapsed, work_elapsed, progress_every)

    _print_summary(
        "ordered_json_parse",
        rows,
        started_at,
        fetch_elapsed,
        parse_elapsed,
        work_elapsed,
        extra={
            "champion_slots_seen": f"{champion_slots:,}",
            "rows_with_region": f"{region_nonempty:,}",
        },
    )


def benchmark_compact_match_columns(conn, queue_id, progress_every):
    if queue_id is None:
        total_rows = _count_rows(conn, "ordered_match_json IS NOT NULL")
        cursor = conn.execute(
            """
            SELECT region, game_version, game_creation, game_end_timestamp,
                   blue_first_blood, blue_first_tower, blue_dragon_share, blue_gold_share
            FROM matches
            WHERE ordered_match_json IS NOT NULL
            ORDER BY game_creation, match_id
            """
        )
    else:
        total_rows = _count_rows(conn, "ordered_match_json IS NOT NULL AND queue_id = ?", (queue_id,))
        cursor = conn.execute(
            """
            SELECT region, game_version, game_creation, game_end_timestamp,
                   blue_first_blood, blue_first_tower, blue_dragon_share, blue_gold_share
            FROM matches
            WHERE ordered_match_json IS NOT NULL
              AND queue_id = ?
            ORDER BY game_creation, match_id
            """,
            (queue_id,),
        )

    rows = 0
    fetch_elapsed = 0.0
    parse_elapsed = 0.0
    work_elapsed = 0.0
    total_duration_minutes = 0.0
    started_at = time.time()
    cursor_iter = iter(cursor)

    while True:
        fetch_started_at = time.time()
        try:
            region, game_version, game_creation, game_end_timestamp, blue_first_blood, blue_first_tower, blue_dragon_share, blue_gold_share = next(cursor_iter)
        except StopIteration:
            fetch_elapsed += time.time() - fetch_started_at
            break
        fetch_elapsed += time.time() - fetch_started_at
        rows += 1

        work_started_at = time.time()
        total_duration_minutes += max(float(((game_end_timestamp or 0) - (game_creation or 0)) / 60_000.0), 1.0)
        _ = (region, game_version, blue_first_blood, blue_first_tower, blue_dragon_share, blue_gold_share)
        work_elapsed += time.time() - work_started_at
        _progress("compact_match_columns", rows, total_rows, started_at, fetch_elapsed, parse_elapsed, work_elapsed, progress_every)

    _print_summary(
        "compact_match_columns",
        rows,
        started_at,
        fetch_elapsed,
        parse_elapsed,
        work_elapsed,
        extra={
            "total_duration_minutes": f"{total_duration_minutes:,.1f}",
        },
    )


def benchmark_participant_history_stream(conn, queue_id, progress_every):
    if queue_id is None:
        total_rows = _count_participant_rows(conn, None)
        cursor = conn.execute(
            """
            SELECT match_id, puuid, champion_name, role, win, kills, deaths, assists,
                   vision_score, damage_to_champions, healing, gold_earned, cs, game_creation, team_id
            FROM participant_history
            ORDER BY game_creation, match_id, team_id
            """
        )
    else:
        total_rows = _count_participant_rows(conn, queue_id)
        cursor = conn.execute(
            """
            SELECT match_id, puuid, champion_name, role, win, kills, deaths, assists,
                   vision_score, damage_to_champions, healing, gold_earned, cs, game_creation, team_id
            FROM participant_history
            WHERE queue_id = ?
            ORDER BY game_creation, match_id, team_id
            """,
            (queue_id,),
        )

    rows = 0
    fetch_elapsed = 0.0
    parse_elapsed = 0.0
    work_elapsed = 0.0
    total_gold = 0.0
    unique_matches = 0
    last_match_id = None
    started_at = time.time()
    cursor_iter = iter(cursor)

    while True:
        fetch_started_at = time.time()
        try:
            row = next(cursor_iter)
        except StopIteration:
            fetch_elapsed += time.time() - fetch_started_at
            break
        fetch_elapsed += time.time() - fetch_started_at
        rows += 1

        work_started_at = time.time()
        match_id = row[0]
        if match_id != last_match_id:
            unique_matches += 1
            last_match_id = match_id
        total_gold += float(row[11] or 0.0)
        work_elapsed += time.time() - work_started_at
        _progress("participant_history_stream", rows, total_rows, started_at, fetch_elapsed, parse_elapsed, work_elapsed, progress_every)

    _print_summary(
        "participant_history_stream",
        rows,
        started_at,
        fetch_elapsed,
        parse_elapsed,
        work_elapsed,
        extra={
            "unique_matches": f"{unique_matches:,}",
            "total_gold": f"{total_gold:,.1f}",
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Temporary read-only benchmark for ML training I/O paths.")
    parser.add_argument("--db-path", default=None, help="SQLite DB path (defaults to LOL_DRAFT_DB_PATH or league_data.db)")
    parser.add_argument("--queue-id", type=int, default=DEFAULT_QUEUE_ID, help="Queue ID filter (default: 420)")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=("raw_json_parse", "ordered_json_parse", "compact_match_columns", "participant_history_stream"),
        choices=("raw_json_parse", "ordered_json_parse", "compact_match_columns", "participant_history_stream"),
        help="Benchmark groups to run",
    )
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY, help="Print progress every N rows")
    args = parser.parse_args()

    load_runtime_env()
    db_path = args.db_path or get_db_path()
    conn = connect_sqlite(db_path, read_only=True)
    _configure_read_connection(conn)

    print(f"db={db_path}")
    print(f"queue_id={args.queue_id}")
    print(f"benchmarks={','.join(args.benchmarks)}")

    if "raw_json_parse" in args.benchmarks:
        benchmark_raw_json_parse(conn, args.queue_id, args.progress_every)
    if "ordered_json_parse" in args.benchmarks:
        benchmark_ordered_json_parse(conn, args.queue_id, args.progress_every)
    if "compact_match_columns" in args.benchmarks:
        benchmark_compact_match_columns(conn, args.queue_id, args.progress_every)
    if "participant_history_stream" in args.benchmarks:
        benchmark_participant_history_stream(conn, args.queue_id, args.progress_every)

    conn.close()


if __name__ == "__main__":
    main()
