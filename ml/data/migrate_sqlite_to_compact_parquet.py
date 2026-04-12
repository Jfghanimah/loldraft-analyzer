import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from ml.data.compact_parquet import CompactParquetWriter, get_compact_dataset_dir
from ml.data.compact_records import match_record_from_ordered_json_row, participant_record_from_history_row
from ml.data.match_storage import connect_sqlite
from ml.runtime_config import get_db_path, load_runtime_env


DEFAULT_MATCH_BATCH_ROWS = 50_000
DEFAULT_PARTICIPANT_BATCH_ROWS = 500_000
DEFAULT_MAX_WORKERS = 1


def _iter_match_rows(conn, queue_id):
    params = []
    queue_filter = ""
    if queue_id is not None:
        queue_filter = "AND queue_id = ?"
        params.append(queue_id)
    return conn.execute(
        f"""
        SELECT match_id, ordered_match_json, region, game_version, queue_id, game_creation, game_end_timestamp,
               blue_first_blood, blue_first_tower, blue_dragon_share, blue_gold_share,
               blue_dragons, red_dragons, gold_diff, game_length_minutes, collector_id, last_updated_ts
        FROM matches
        WHERE ordered_match_json IS NOT NULL
          {queue_filter}
        ORDER BY game_creation, match_id
        """,
        tuple(params),
    )


def _iter_match_rows_by_rowid(conn, queue_id, rowid_start, rowid_end):
    params = [rowid_start, rowid_end]
    queue_filter = ""
    if queue_id is not None:
        queue_filter = "AND queue_id = ?"
        params.append(queue_id)
    return conn.execute(
        f"""
        SELECT match_id, ordered_match_json, region, game_version, queue_id, game_creation, game_end_timestamp,
               blue_first_blood, blue_first_tower, blue_dragon_share, blue_gold_share,
               blue_dragons, red_dragons, gold_diff, game_length_minutes, collector_id, last_updated_ts
        FROM matches
        WHERE rowid BETWEEN ? AND ?
          AND ordered_match_json IS NOT NULL
          {queue_filter}
        ORDER BY game_creation, match_id
        """,
        tuple(params),
    )


def _iter_participant_rows(conn, queue_id):
    params = []
    queue_filter = ""
    if queue_id is not None:
        queue_filter = "WHERE ph.queue_id = ?"
        params.append(queue_id)
    return conn.execute(
        f"""
        SELECT ph.match_id, m.region, ph.queue_id, ph.game_creation, m.game_end_timestamp, ph.game_version,
               ph.puuid, ph.champion_name, ph.role, ph.team_id, ph.win, ph.kills, ph.deaths, ph.assists,
               ph.vision_score, ph.damage_to_champions, ph.healing, ph.gold_earned, ph.cs
        FROM participant_history ph
        LEFT JOIN matches m ON m.match_id = ph.match_id
        {queue_filter}
        ORDER BY ph.game_creation, ph.match_id, ph.team_id
        """,
        tuple(params),
    )


def _iter_participant_rows_by_rowid(conn, queue_id, rowid_start, rowid_end):
    params = [rowid_start, rowid_end]
    queue_filter = ""
    if queue_id is not None:
        queue_filter = "AND ph.queue_id = ?"
        params.append(queue_id)
    return conn.execute(
        f"""
        SELECT ph.match_id, m.region, ph.queue_id, ph.game_creation, m.game_end_timestamp, ph.game_version,
               ph.puuid, ph.champion_name, ph.role, ph.team_id, ph.win, ph.kills, ph.deaths, ph.assists,
               ph.vision_score, ph.damage_to_champions, ph.healing, ph.gold_earned, ph.cs
        FROM participant_history ph
        LEFT JOIN matches m ON m.match_id = ph.match_id
        WHERE ph.rowid BETWEEN ? AND ?
          {queue_filter}
        ORDER BY ph.game_creation, ph.match_id, ph.team_id
        """,
        tuple(params),
    )


def _rowid_bounds(conn, table_name, queue_id):
    where_sql = ""
    params = ()
    if queue_id is not None:
        where_sql = "WHERE queue_id = ?"
        params = (queue_id,)
    row = conn.execute(f"SELECT MIN(rowid), MAX(rowid), COUNT(*) FROM {table_name} {where_sql}", params).fetchone()
    if not row or not row[2]:
        return None, None, 0
    return int(row[0]), int(row[1]), int(row[2])


def _rowid_ranges(min_rowid, max_rowid, chunk_rows):
    if min_rowid is None or max_rowid is None:
        return []
    return [(start, min(start + chunk_rows - 1, max_rowid)) for start in range(min_rowid, max_rowid + 1, chunk_rows)]


def _flush_kind(writer, kind, records, total, started_at):
    if not records:
        return total
    writer.add_records(kind, records)
    written = writer.flush()
    total += len(records)
    elapsed = time.time() - started_at
    rate = total / elapsed if elapsed > 0 else 0.0
    files = sum(1 for item in written if item["kind"] == kind)
    print(f"[compact-migrate] {kind}: rows={total:,} files+={files} rate={rate:,.1f}/s", flush=True)
    return total


def _write_manifest(output_dir, rows):
    import json
    import os

    if not rows:
        return
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = f"{output_dir}/manifest.jsonl"
    with open(manifest_path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"written_at_ts": int(time.time()), **row}, sort_keys=True) + "\n")


def _migrate_match_range(db_path, output_dir, queue_id, rowid_start, rowid_end):
    conn = connect_sqlite(db_path, read_only=True)
    writer = CompactParquetWriter(output_dir, write_manifest=False)
    records = [match_record_from_ordered_json_row(row) for row in _iter_match_rows_by_rowid(conn, queue_id, rowid_start, rowid_end)]
    conn.close()
    writer.add_records("matches", records)
    written = writer.flush()
    return {"kind": "matches", "rows": len(records), "written": written, "rowid_start": rowid_start, "rowid_end": rowid_end}


def _migrate_participant_range(db_path, output_dir, queue_id, rowid_start, rowid_end):
    conn = connect_sqlite(db_path, read_only=True)
    writer = CompactParquetWriter(output_dir, write_manifest=False)
    records = [participant_record_from_history_row(row) for row in _iter_participant_rows_by_rowid(conn, queue_id, rowid_start, rowid_end)]
    conn.close()
    writer.add_records("participants", records)
    written = writer.flush()
    return {"kind": "participants", "rows": len(records), "written": written, "rowid_start": rowid_start, "rowid_end": rowid_end}


def _run_parallel_jobs(jobs, max_workers, total_jobs, started_at, output_dir):
    totals = {"matches": 0, "participants": 0}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fn, *args) for fn, args in jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            totals[result["kind"]] += result["rows"]
            _write_manifest(output_dir, result["written"])
            elapsed = time.time() - started_at
            rate = sum(totals.values()) / elapsed if elapsed > 0 else 0.0
            print(
                f"[compact-migrate] job {completed:,}/{total_jobs:,} {result['kind']} "
                f"rows={result['rows']:,} total={totals[result['kind']]:,} rate={rate:,.1f}/s",
                flush=True,
            )
    return totals


def migrate_sqlite_to_compact_parquet(
    *,
    db_path,
    output_dir,
    queue_id=420,
    match_batch_rows=DEFAULT_MATCH_BATCH_ROWS,
    participant_batch_rows=DEFAULT_PARTICIPANT_BATCH_ROWS,
    limit_matches=None,
    limit_participants=None,
    max_workers=DEFAULT_MAX_WORKERS,
):
    conn = connect_sqlite(db_path, read_only=True)
    if max_workers > 1 and limit_matches is None and limit_participants is None:
        matches_min, matches_max, matches_count = _rowid_bounds(conn, "matches", queue_id)
        participants_min, participants_max, participants_count = _rowid_bounds(conn, "participant_history", queue_id)
        conn.close()
        started_at = time.time()
        match_ranges = _rowid_ranges(matches_min, matches_max, match_batch_rows)
        participant_ranges = _rowid_ranges(participants_min, participants_max, participant_batch_rows)
        jobs = [
            (_migrate_match_range, (db_path, output_dir, queue_id, rowid_start, rowid_end))
            for rowid_start, rowid_end in match_ranges
        ]
        jobs.extend(
            (_migrate_participant_range, (db_path, output_dir, queue_id, rowid_start, rowid_end))
            for rowid_start, rowid_end in participant_ranges
        )
        print(
            f"[compact-migrate] parallel workers={max_workers} "
            f"matches={matches_count:,} participants={participants_count:,} jobs={len(jobs):,}",
            flush=True,
        )
        totals = _run_parallel_jobs(jobs, max_workers, len(jobs), started_at, output_dir)
        elapsed = time.time() - started_at
        print(
            f"[compact-migrate] complete matches={totals['matches']:,} participants={totals['participants']:,} "
            f"elapsed={elapsed:.1f}s output_dir={output_dir}",
            flush=True,
        )
        return {"matches": totals["matches"], "participants": totals["participants"], "elapsed_seconds": elapsed}

    writer = CompactParquetWriter(output_dir)
    started_at = time.time()

    total_matches = 0
    batch = []
    for row in _iter_match_rows(conn, queue_id):
        if limit_matches is not None and total_matches + len(batch) >= limit_matches:
            break
        batch.append(match_record_from_ordered_json_row(row))
        if len(batch) >= match_batch_rows:
            total_matches = _flush_kind(writer, "matches", batch, total_matches, started_at)
            batch = []
    total_matches = _flush_kind(writer, "matches", batch, total_matches, started_at)

    total_participants = 0
    batch = []
    for row in _iter_participant_rows(conn, queue_id):
        if limit_participants is not None and total_participants + len(batch) >= limit_participants:
            break
        batch.append(participant_record_from_history_row(row))
        if len(batch) >= participant_batch_rows:
            total_participants = _flush_kind(writer, "participants", batch, total_participants, started_at)
            batch = []
    total_participants = _flush_kind(writer, "participants", batch, total_participants, started_at)
    conn.close()

    elapsed = time.time() - started_at
    print(
        f"[compact-migrate] complete matches={total_matches:,} participants={total_participants:,} "
        f"elapsed={elapsed:.1f}s output_dir={output_dir}",
        flush=True,
    )
    return {"matches": total_matches, "participants": total_participants, "elapsed_seconds": elapsed}


def main():
    parser = argparse.ArgumentParser(description="Migrate structured SQLite match facts to compact partitioned Parquet.")
    parser.add_argument("--db-path", default=None, help="SQLite DB path. Defaults to LOL_DRAFT_DB_PATH.")
    parser.add_argument("--output-dir", default=None, help="Compact dataset root. Defaults to LOL_DRAFT_DATASET_DIR.")
    parser.add_argument("--queue-id", type=int, default=420, help="Queue ID to migrate. Use --queue-id -1 for all queues.")
    parser.add_argument("--match-batch-rows", type=int, default=DEFAULT_MATCH_BATCH_ROWS)
    parser.add_argument("--participant-batch-rows", type=int, default=DEFAULT_PARTICIPANT_BATCH_ROWS)
    parser.add_argument("--limit-matches", type=int, default=None, help="Optional smoke-test limit for match rows")
    parser.add_argument("--limit-participants", type=int, default=None, help="Optional smoke-test limit for participant rows")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Parallel workers for full migrations without limits")
    args = parser.parse_args()

    load_runtime_env()
    queue_id = None if args.queue_id == -1 else args.queue_id
    migrate_sqlite_to_compact_parquet(
        db_path=args.db_path or get_db_path(),
        output_dir=args.output_dir or get_compact_dataset_dir(),
        queue_id=queue_id,
        match_batch_rows=args.match_batch_rows,
        participant_batch_rows=args.participant_batch_rows,
        limit_matches=args.limit_matches,
        limit_participants=args.limit_participants,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
