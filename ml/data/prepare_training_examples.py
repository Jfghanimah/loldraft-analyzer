import argparse
import json
import os
import shutil
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ml.data.compact_parquet import DEFAULT_COMPRESSION, _require_pyarrow, get_compact_dataset_dir
from ml.data.match_format import ROLE_ORDER
from ml.features.recent_history import QUEUE_ID_SOLO, RecentHistoryStore, dense_feature_columns, parse_patch
from ml.runtime_config import load_runtime_env
from ml.trainer.feature_pipeline import AUX_TARGET_COLUMNS, CHAMPION_COLUMNS, CHAMPION_LIST_PATH, REGION_LIST_PATH


TRAINING_EXAMPLE_SCHEMA_VERSION = 1
DEFAULT_TRAINING_EXAMPLE_ROWS_PER_FILE = 250_000
DEFAULT_MAX_WORKERS = 1
TRAINING_EXAMPLE_METADATA_COLUMNS = (
    "training_example_schema_version",
    "match_id",
    "platform",
    "queue_id",
    "game_date",
    "game_creation",
)
TRAINING_EXAMPLE_MODEL_COLUMNS = (
    "label",
    *CHAMPION_COLUMNS,
    "region_id",
    *AUX_TARGET_COLUMNS,
    *dense_feature_columns(ROLE_ORDER),
)
TRAINING_EXAMPLE_COLUMNS = (*TRAINING_EXAMPLE_METADATA_COLUMNS, *TRAINING_EXAMPLE_MODEL_COLUMNS)


def _load_id_mapping(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_id_mapping(mapping, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ordered = dict(sorted(mapping.items(), key=lambda item: item[1]))
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(ordered, handle, indent=4)


def _id_for(mapping, key):
    if key not in mapping:
        mapping[key] = max(mapping.values(), default=-1) + 1
    return mapping[key]


def _require_columns(df, columns, label):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} dataset is missing required columns: {', '.join(missing)}")


def _dataset_files(dataset_dir, kind):
    root = Path(dataset_dir) / kind
    return sorted(root.glob("queue_id=*/platform=*/game_date=*/part-*.parquet"))


def _available_partitions(dataset_dir, queue_id):
    partitions = set()
    root = Path(dataset_dir) / "matches"
    if not root.exists():
        return []
    queue_roots = [root / f"queue_id={queue_id}"] if queue_id is not None else sorted(root.glob("queue_id=*"))
    for queue_root in queue_roots:
        if not queue_root.exists():
            continue
        for date_dir in queue_root.glob("platform=*/game_date=*"):
            platform = date_dir.parent.name.split("=", 1)[1]
            game_date = date_dir.name.split("=", 1)[1]
            partitions.add((platform, game_date))
    return sorted(partitions)


def _available_game_dates(dataset_dir, queue_id, platform=None):
    return sorted({game_date for part_platform, game_date in _available_partitions(dataset_dir, queue_id) if platform is None or part_platform == platform})


def _available_platforms(dataset_dir, queue_id):
    return sorted({platform for platform, _ in _available_partitions(dataset_dir, queue_id)})


def _read_partition(dataset_dir, kind, game_date, queue_id, platform=None):
    pa, pq = _require_pyarrow()
    files = []
    root = Path(dataset_dir) / kind
    queue_roots = [root / f"queue_id={queue_id}"] if queue_id is not None else sorted(root.glob("queue_id=*"))
    for queue_root in queue_roots:
        if platform is None:
            files.extend(sorted(queue_root.glob(f"platform=*/game_date={game_date}/part-*.parquet")))
        else:
            files.extend(sorted(queue_root.glob(f"platform={platform}/game_date={game_date}/part-*.parquet")))
    if not files:
        return pd.DataFrame()
    tables = [pq.ParquetFile(file_path).read() for file_path in files]
    return pa.concat_tables(tables, promote_options="default").to_pandas()


def _participant_rows_by_match(participants_df):
    if participants_df.empty:
        return {}
    participants_df = participants_df.sort_values(["match_id", "slot"])
    grouped = {}
    for match_id, group in participants_df.groupby("match_id", sort=False):
        grouped[match_id] = [_participant_history_row(row) for row in group.to_dict("records")]
    return grouped


def _participant_history_row(row):
    return {
        "puuid": row["puuid"],
        "champion_name": row["champion_name"],
        "role": row["role"],
        "win": int(row["win"] or 0),
        "kills": int(row["kills"] or 0),
        "deaths": int(row["deaths"] or 0),
        "assists": int(row["assists"] or 0),
        "vision_score": float(row["vision_score"] or 0.0),
        "damage_to_champions": float(row["damage_to_champions"] or 0.0),
        "healing": float(row["healing"] or 0.0),
        "gold_earned": float(row["gold_earned"] or 0.0),
        "cs": float(row["cs"] or 0.0),
        "game_creation": int(row["game_creation"] or 0),
        "team_id": int(row["team_id"] or 0),
        "duration_minutes": float(row["duration_minutes"] or 1.0),
        "kda_value": float(row["kda_value"] or 0.0),
        "dpm_value": float(row["dpm_value"] or 0.0),
        "gpm_value": float(row["gpm_value"] or 0.0),
        "cspm_value": float(row["cspm_value"] or 0.0),
        "vspm_value": float(row["vspm_value"] or 0.0),
        "hpm_value": float(row["hpm_value"] or 0.0),
    }


def _build_example(match_row, participant_rows, history_store, champion_list, region_list):
    game_creation = int(match_row["game_creation"] or 0)
    game_version = match_row.get("game_version") or ""
    feature_values = []
    for participant in participant_rows:
        feature_values.extend(
            history_store.feature_vector(
                participant["puuid"],
                participant["champion_name"],
                participant["role"],
                game_creation,
            )
        )
    feature_values.extend(parse_patch(game_version))

    champion_ids = []
    for slot in range(10):
        champion_name = match_row.get(f"champion_{slot}") or participant_rows[slot]["champion_name"]
        champion_ids.append(_id_for(champion_list, champion_name))

    region_key = match_row.get("platform") or ""
    region_id = _id_for(region_list, region_key) if region_key else 0
    values = {
        "training_example_schema_version": TRAINING_EXAMPLE_SCHEMA_VERSION,
        "match_id": match_row["match_id"],
        "platform": region_key,
        "queue_id": int(match_row["queue_id"] or 0),
        "game_date": match_row.get("game_date") or "unknown",
        "game_creation": game_creation,
        "label": int(match_row["label"] or 0),
        "region_id": region_id,
        "target_gold_diff": float(match_row.get("gold_diff") or 0.0),
        "target_blue_dragons": float(match_row.get("blue_dragons") or 0.0),
        "target_red_dragons": float(match_row.get("red_dragons") or 0.0),
        "target_game_length_minutes": float(match_row.get("duration_minutes") or 1.0),
    }
    for column, champion_id in zip(CHAMPION_COLUMNS, champion_ids):
        values[column] = champion_id
    for column, value in zip(dense_feature_columns(ROLE_ORDER), feature_values):
        values[column] = float(value)
    return {column: values.get(column) for column in TRAINING_EXAMPLE_COLUMNS}


def _write_training_examples(output_dir, records, compression=DEFAULT_COMPRESSION, rows_per_file=DEFAULT_TRAINING_EXAMPLE_ROWS_PER_FILE):
    if not records:
        return []
    pa, pq = _require_pyarrow()
    written = []
    output_root = Path(output_dir) / "training_examples"
    grouped = {}
    for record in records:
        key = (record["queue_id"], record["platform"] or "unknown", record["game_date"] or "unknown")
        grouped.setdefault(key, []).append(record)
    for (queue_id, platform, game_date), partition_records in grouped.items():
        partition_dir = output_root / f"queue_id={queue_id}" / f"platform={platform}" / f"game_date={game_date}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(partition_records), rows_per_file):
            chunk = partition_records[start:start + rows_per_file]
            table = pa.Table.from_pylist([{column: row.get(column) for column in TRAINING_EXAMPLE_COLUMNS} for row in chunk])
            path = partition_dir / f"part-{int(time.time() * 1000)}-{uuid.uuid4().hex[:10]}-{start:06d}.parquet"
            pq.write_table(table, path, compression=compression, use_dictionary=True, write_statistics=True)
            written.append({"path": str(path), "rows": len(chunk), "queue_id": queue_id, "platform": platform, "game_date": game_date})
    return written


def _training_example_files(dataset_dir, queue_id):
    root = Path(dataset_dir) / "training_examples"
    if queue_id is None:
        return sorted(root.glob("queue_id=*/platform=*/game_date=*/part-*.parquet"))
    return sorted(root.glob(f"queue_id={queue_id}/platform=*/game_date=*/part-*.parquet"))


def load_training_examples_dataframe(dataset_dir, *, champion_path=CHAMPION_LIST_PATH, queue_id=QUEUE_ID_SOLO):
    pa, pq = _require_pyarrow()
    files = _training_example_files(dataset_dir, queue_id)
    if not files:
        raise ValueError(f"No prepared training_examples Parquet files found in {dataset_dir}")
    tables = [pq.ParquetFile(file_path).read(columns=list(TRAINING_EXAMPLE_MODEL_COLUMNS)) for file_path in files]
    dataframe = pa.concat_tables(tables, promote_options="default").to_pandas()
    return dataframe.sample(frac=1, random_state=42).reset_index(drop=True), _load_id_mapping(champion_path)


def _append_manifest(output_dir, manifest_rows):
    if not manifest_rows:
        return
    manifest_path = Path(output_dir) / "training_examples_manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps({"written_at_ts": int(time.time()), **row}, sort_keys=True) + "\n")


def _sync_mappings_from_compact_matches(dataset_dir, queue_id, champion_path, region_path):
    pa, pq = _require_pyarrow()
    champion_list = _load_id_mapping(champion_path)
    region_list = _load_id_mapping(region_path)
    original_champion_list = dict(champion_list)
    original_region_list = dict(region_list)
    columns = [f"champion_{slot}" for slot in range(10)] + ["platform"]
    files = _dataset_files(dataset_dir, "matches")
    for file_path in files:
        if queue_id is not None and f"queue_id={queue_id}" not in str(file_path):
            continue
        table = pq.ParquetFile(file_path).read(columns=columns)
        for row in table.to_pylist():
            if row.get("platform"):
                _id_for(region_list, row["platform"])
            for slot in range(10):
                champion_name = row.get(f"champion_{slot}")
                if champion_name:
                    _id_for(champion_list, champion_name)
    if champion_list != original_champion_list:
        _write_id_mapping(champion_list, champion_path)
    if region_list != original_region_list:
        _write_id_mapping(region_list, region_path)
    return champion_list, region_list


def _prepare_platform_training_examples(dataset_dir, output_dir, queue_id, platform, dates, champion_list, region_list, rows_per_file):
    history_store = RecentHistoryStore()
    total_examples = 0
    written_manifest_rows = []
    started_at = time.time()
    for game_date in dates:
        matches_df = _read_partition(dataset_dir, "matches", game_date, queue_id, platform=platform)
        participants_df = _read_partition(dataset_dir, "participants", game_date, queue_id, platform=platform)
        if matches_df.empty or participants_df.empty:
            continue
        _require_columns(matches_df, ("match_id", "label", "game_creation", "platform", "queue_id", "game_date"), "matches")
        _require_columns(participants_df, ("match_id", "puuid", "champion_name", "role", "slot"), "participants")
        participants_by_match = _participant_rows_by_match(participants_df)
        examples = []
        matches_df = matches_df.sort_values(["game_creation", "match_id"])
        for match_row in matches_df.to_dict("records"):
            participant_rows = participants_by_match.get(match_row["match_id"], [])
            if len(participant_rows) != 10:
                continue
            examples.append(_build_example(match_row, participant_rows, history_store, champion_list, region_list))
            history_store.add_match_rows(participant_rows)
        written = _write_training_examples(output_dir, examples, rows_per_file=rows_per_file)
        written_manifest_rows.extend(written)
        total_examples += len(examples)
        elapsed = time.time() - started_at
        rate = total_examples / elapsed if elapsed > 0 else 0.0
        print(
            f"[training-examples] platform={platform} date={game_date} "
            f"examples={total_examples:,} files+={len(written)} rate={rate:,.1f}/s",
            flush=True,
        )
    return {"platform": platform, "examples": total_examples, "manifest_rows": written_manifest_rows}


def prepare_training_examples(
    *,
    dataset_dir,
    output_dir=None,
    champion_path=CHAMPION_LIST_PATH,
    region_path=REGION_LIST_PATH,
    queue_id=QUEUE_ID_SOLO,
    limit_dates=None,
    rows_per_file=DEFAULT_TRAINING_EXAMPLE_ROWS_PER_FILE,
    overwrite=False,
    max_workers=DEFAULT_MAX_WORKERS,
):
    output_dir = output_dir or dataset_dir
    existing_files = _training_example_files(output_dir, queue_id)
    if existing_files and not overwrite:
        raise ValueError(
            f"Found existing training_examples in {output_dir}. "
            "Pass overwrite=True or --overwrite to replace them."
        )
    if existing_files and overwrite:
        shutil.rmtree(Path(output_dir) / "training_examples")
        manifest_path = Path(output_dir) / "training_examples_manifest.jsonl"
        if manifest_path.exists():
            manifest_path.unlink()

    champion_list, region_list = _sync_mappings_from_compact_matches(dataset_dir, queue_id, champion_path, region_path)
    total_examples = 0
    started_at = time.time()
    written_manifest_rows = []

    platforms = _available_platforms(dataset_dir, queue_id)
    jobs = []
    for platform in platforms:
        dates = _available_game_dates(dataset_dir, queue_id, platform=platform)
        if limit_dates is not None:
            dates = dates[:limit_dates]
        if dates:
            jobs.append((platform, dates))

    if max_workers > 1 and len(jobs) > 1:
        worker_count = min(max_workers, len(jobs))
        print(f"[training-examples] parallel platforms={len(jobs)} workers={worker_count}", flush=True)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _prepare_platform_training_examples,
                    dataset_dir,
                    output_dir,
                    queue_id,
                    platform,
                    dates,
                    champion_list,
                    region_list,
                    rows_per_file,
                )
                for platform, dates in jobs
            ]
            for future in as_completed(futures):
                result = future.result()
                written_manifest_rows.extend(result["manifest_rows"])
                total_examples += result["examples"]
                _append_manifest(output_dir, result["manifest_rows"])
                print(
                    f"[training-examples] platform={result['platform']} complete examples={result['examples']:,}",
                    flush=True,
                )
    else:
        for platform, dates in jobs:
            result = _prepare_platform_training_examples(
                dataset_dir,
                output_dir,
                queue_id,
                platform,
                dates,
                champion_list,
                region_list,
                rows_per_file,
            )
            written_manifest_rows.extend(result["manifest_rows"])
            total_examples += result["examples"]
            _append_manifest(output_dir, result["manifest_rows"])

    print(
        f"[training-examples] complete examples={total_examples:,} files={len(written_manifest_rows):,} "
        f"elapsed={time.time() - started_at:.1f}s output_dir={output_dir}",
        flush=True,
    )
    return {"examples": total_examples, "files": len(written_manifest_rows), "output_dir": str(output_dir)}


def main():
    parser = argparse.ArgumentParser(description="Build model-ready training_examples Parquet from compact match facts.")
    parser.add_argument("--dataset-dir", default=None, help="Compact dataset root. Defaults to LOL_DRAFT_DATASET_DIR.")
    parser.add_argument("--output-dir", default=None, help="Output root. Defaults to dataset-dir.")
    parser.add_argument("--queue-id", type=int, default=QUEUE_ID_SOLO, help="Queue ID to prepare. Use --queue-id -1 for all queues.")
    parser.add_argument("--limit-dates", type=int, default=None, help="Optional smoke-test limit for date partitions")
    parser.add_argument("--rows-per-file", type=int, default=DEFAULT_TRAINING_EXAMPLE_ROWS_PER_FILE)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing training_examples output")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Parallel workers by platform")
    args = parser.parse_args()

    load_runtime_env()
    prepare_training_examples(
        dataset_dir=args.dataset_dir or get_compact_dataset_dir(),
        output_dir=args.output_dir,
        queue_id=None if args.queue_id == -1 else args.queue_id,
        limit_dates=args.limit_dates,
        rows_per_file=args.rows_per_file,
        overwrite=args.overwrite,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
