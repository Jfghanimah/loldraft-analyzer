import argparse
import importlib.util
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ml.data.match_storage import connect_sqlite
from ml.runtime_config import get_db_path, load_runtime_env

READ_PRAGMAS = (
    "PRAGMA cache_size = -65536",
    "PRAGMA mmap_size = 0",
    "PRAGMA read_uncommitted = ON",
    "PRAGMA temp_store = MEMORY",
)
DATASET_SCHEMA_VERSION = 1
DEFAULT_MATCH_CHUNK_ROWS = 25_000
DEFAULT_PARTICIPANT_CHUNK_ROWS = 250_000
DEFAULT_MAX_WORKERS = max(1, min(24, os.cpu_count() or 1))
MATCH_CHAMPION_COLUMNS = [f"champion_{slot}" for slot in range(10)]

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None


def _loads_json(payload):
    if orjson is not None:
        return orjson.loads(payload)
    return json.loads(payload)


def _configure_read_connection(conn):
    for pragma in READ_PRAGMAS:
        try:
            conn.execute(pragma)
        except Exception:
            continue


def _detect_parquet_engine():
    if importlib.util.find_spec("pyarrow"):
        return "pyarrow"
    if importlib.util.find_spec("fastparquet"):
        return "fastparquet"
    raise RuntimeError(
        "Parquet support is not installed. Install 'pyarrow' (preferred) or 'fastparquet' "
        "before running ml.data.export_training_parquet."
    )


def _where_clause(queue_id, *, require_ordered_match_json=False):
    clauses = []
    params = []
    if require_ordered_match_json:
        clauses.append("ordered_match_json IS NOT NULL")
    if queue_id is not None:
        clauses.append("queue_id = ?")
        params.append(queue_id)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where_sql, tuple(params)


def _get_rowid_bounds(conn, table_name, where_sql="", params=()):
    query = f"SELECT MIN(rowid), MAX(rowid), COUNT(*) FROM {table_name} {where_sql}"
    row = conn.execute(query, params).fetchone()
    if not row or row[2] == 0:
        return None, None, 0
    return int(row[0]), int(row[1]), int(row[2])


def _build_rowid_ranges(min_rowid, max_rowid, chunk_size):
    if min_rowid is None or max_rowid is None:
        return []
    return [
        (start, min(start + chunk_size - 1, max_rowid))
        for start in range(min_rowid, max_rowid + 1, chunk_size)
    ]


def _match_record_from_row(row):
    ordered = _loads_json(row[1])
    champions = list(ordered["champions"])
    if len(champions) != 10:
        raise ValueError(f"Expected 10 champions in ordered_match_json for match_id={row[0]}")

    record = {
        "match_id": row[0],
        "region": row[2] or "",
        "game_version": row[3] or "",
        "queue_id": int(row[4] or 0),
        "game_creation": int(row[5] or 0),
        "game_end_timestamp": int(row[6] or 0),
        "label": int(bool(ordered["blue_win"])),
        "target_blue_dragons": float(row[7]) if row[7] is not None else None,
        "target_red_dragons": float(row[8]) if row[8] is not None else None,
        "target_gold_diff": float(row[9]) if row[9] is not None else None,
        "target_game_length_minutes": float(row[10]) if row[10] is not None else None,
    }
    for slot, champion_name in enumerate(champions):
        record[MATCH_CHAMPION_COLUMNS[slot]] = champion_name
    return record


def _participant_history_record_from_row(row):
    return {
        "match_id": row[0],
        "puuid": row[1],
        "queue_id": int(row[2] or 0),
        "game_creation": int(row[3] or 0),
        "champion_name": row[4] or "",
        "role": row[5] or "",
        "team_id": int(row[6] or 0),
        "win": int(row[7] or 0),
        "kills": int(row[8] or 0),
        "deaths": int(row[9] or 0),
        "assists": int(row[10] or 0),
        "vision_score": float(row[11] or 0.0),
        "damage_to_champions": float(row[12] or 0.0),
        "healing": float(row[13] or 0.0),
        "gold_earned": float(row[14] or 0.0),
        "cs": float(row[15] or 0.0),
        "game_version": row[16] or "",
    }


def _write_parquet_records(records, output_path, engine):
    df = pd.DataFrame.from_records(records)
    df.to_parquet(output_path, index=False, engine=engine)


def _export_matches_shard(db_path, queue_id, rowid_start, rowid_end, output_path, engine):
    conn = connect_sqlite(db_path, read_only=True)
    _configure_read_connection(conn)
    params = [rowid_start, rowid_end]
    queue_filter = ""
    if queue_id is not None:
        queue_filter = "AND queue_id = ?"
        params.append(queue_id)

    cursor = conn.execute(
        f"""
        SELECT match_id, ordered_match_json, region, game_version, queue_id, game_creation, game_end_timestamp,
               blue_dragons, red_dragons, gold_diff, game_length_minutes
        FROM matches
        WHERE rowid BETWEEN ? AND ?
          AND ordered_match_json IS NOT NULL
          {queue_filter}
        ORDER BY game_creation, match_id
        """,
        tuple(params),
    )

    records = []
    parse_seconds = 0.0
    started_at = time.time()
    for row in cursor:
        parse_started_at = time.time()
        records.append(_match_record_from_row(row))
        parse_seconds += time.time() - parse_started_at

    conn.close()
    write_started_at = time.time()
    if records:
        _write_parquet_records(records, output_path, engine)
    write_seconds = time.time() - write_started_at
    return {
        "kind": "matches",
        "rowid_start": rowid_start,
        "rowid_end": rowid_end,
        "rows": len(records),
        "parse_seconds": parse_seconds,
        "write_seconds": write_seconds,
        "elapsed_seconds": time.time() - started_at,
        "output_path": str(output_path),
    }


def _export_participant_history_shard(db_path, queue_id, rowid_start, rowid_end, output_path, engine):
    conn = connect_sqlite(db_path, read_only=True)
    _configure_read_connection(conn)
    params = [rowid_start, rowid_end]
    queue_filter = ""
    if queue_id is not None:
        queue_filter = "AND queue_id = ?"
        params.append(queue_id)

    cursor = conn.execute(
        f"""
        SELECT match_id, puuid, queue_id, game_creation, champion_name, role, team_id, win,
               kills, deaths, assists, vision_score, damage_to_champions, healing, gold_earned, cs, game_version
        FROM participant_history
        WHERE rowid BETWEEN ? AND ?
          {queue_filter}
        ORDER BY game_creation, match_id, team_id
        """,
        tuple(params),
    )

    records = []
    started_at = time.time()
    for row in cursor:
        records.append(_participant_history_record_from_row(row))

    conn.close()
    write_started_at = time.time()
    if records:
        _write_parquet_records(records, output_path, engine)
    write_seconds = time.time() - write_started_at
    return {
        "kind": "participant_history",
        "rowid_start": rowid_start,
        "rowid_end": rowid_end,
        "rows": len(records),
        "parse_seconds": 0.0,
        "write_seconds": write_seconds,
        "elapsed_seconds": time.time() - started_at,
        "output_path": str(output_path),
    }


def _print_completed_shard(result, completed, total):
    print(
        f"[parquet] {result['kind']} shard {completed}/{total} "
        f"rows={result['rows']:,} "
        f"elapsed={result['elapsed_seconds']:.1f}s "
        f"parse={result['parse_seconds']:.1f}s "
        f"write={result['write_seconds']:.1f}s "
        f"path={result['output_path']}"
    )


def _run_parallel_exports(kind, jobs, worker_fn, max_workers):
    if not jobs:
        return []

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_fn, *job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            _print_completed_shard(result, completed, len(jobs))
    return sorted(results, key=lambda item: item["rowid_start"])


def _write_manifest(output_dir, payload):
    manifest_path = Path(output_dir) / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _prepare_output_dir(output_dir, overwrite=False):
    output_path = Path(output_dir)
    if output_path.exists():
        if any(output_path.iterdir()) and not overwrite:
            raise ValueError(
                f"Output directory '{output_path}' is not empty. Pass --overwrite to reuse it."
            )
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "matches").mkdir(parents=True, exist_ok=True)
    (output_path / "participant_history").mkdir(parents=True, exist_ok=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export compact ML training parquet shards from the source SQLite DB.")
    parser.add_argument("--db-path", default=None, help="SQLite DB path (defaults to LOL_DRAFT_DB_PATH or league_data.db)")
    parser.add_argument("--output-dir", default="ml/save_data/prepared_training_dataset", help="Output directory for parquet shards")
    parser.add_argument("--queue-id", type=int, default=420, help="Queue ID filter (default: 420)")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Number of parallel read workers")
    parser.add_argument("--match-chunk-rows", type=int, default=DEFAULT_MATCH_CHUNK_ROWS, help="Approximate rowid span per matches shard")
    parser.add_argument("--participant-chunk-rows", type=int, default=DEFAULT_PARTICIPANT_CHUNK_ROWS, help="Approximate rowid span per participant_history shard")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty output directory")
    args = parser.parse_args()

    load_runtime_env()
    db_path = args.db_path or get_db_path()
    engine = _detect_parquet_engine()
    output_dir = _prepare_output_dir(args.output_dir, overwrite=args.overwrite)

    conn = connect_sqlite(db_path, read_only=True)
    _configure_read_connection(conn)
    matches_where_sql, matches_params = _where_clause(args.queue_id, require_ordered_match_json=True)
    matches_min_rowid, matches_max_rowid, matches_count = _get_rowid_bounds(conn, "matches", matches_where_sql, matches_params)
    participant_where_sql, participant_params = _where_clause(args.queue_id, require_ordered_match_json=False)
    participant_min_rowid, participant_max_rowid, participant_count = _get_rowid_bounds(
        conn,
        "participant_history",
        participant_where_sql,
        participant_params,
    )
    conn.close()

    match_ranges = _build_rowid_ranges(matches_min_rowid, matches_max_rowid, args.match_chunk_rows)
    participant_ranges = _build_rowid_ranges(participant_min_rowid, participant_max_rowid, args.participant_chunk_rows)

    match_jobs = [
        (
            db_path,
            args.queue_id,
            rowid_start,
            rowid_end,
            output_dir / "matches" / f"part-{index:05d}.parquet",
            engine,
        )
        for index, (rowid_start, rowid_end) in enumerate(match_ranges)
    ]
    participant_jobs = [
        (
            db_path,
            args.queue_id,
            rowid_start,
            rowid_end,
            output_dir / "participant_history" / f"part-{index:05d}.parquet",
            engine,
        )
        for index, (rowid_start, rowid_end) in enumerate(participant_ranges)
    ]

    started_at = time.time()
    print(f"[parquet] db={db_path}")
    print(f"[parquet] output_dir={output_dir}")
    print(f"[parquet] queue_id={args.queue_id}")
    print(f"[parquet] engine={engine}")
    print(f"[parquet] workers={args.max_workers}")
    print(f"[parquet] matches={matches_count:,} rows across {len(match_jobs):,} shards")
    print(f"[parquet] participant_history={participant_count:,} rows across {len(participant_jobs):,} shards")

    match_results = _run_parallel_exports("matches", match_jobs, _export_matches_shard, args.max_workers)
    participant_results = _run_parallel_exports(
        "participant_history",
        participant_jobs,
        _export_participant_history_shard,
        args.max_workers,
    )

    manifest = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_db_path": db_path,
        "queue_id": args.queue_id,
        "parquet_engine": engine,
        "max_workers": args.max_workers,
        "matches": {
            "row_count": matches_count,
            "chunk_rows": args.match_chunk_rows,
            "shards": match_results,
        },
        "participant_history": {
            "row_count": participant_count,
            "chunk_rows": args.participant_chunk_rows,
            "shards": participant_results,
        },
        "elapsed_seconds": time.time() - started_at,
    }
    _write_manifest(output_dir, manifest)
    print(f"[parquet] manifest={output_dir / 'manifest.json'}")
    print(f"[parquet] total_elapsed={manifest['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
