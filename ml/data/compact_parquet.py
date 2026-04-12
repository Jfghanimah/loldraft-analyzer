import json
import os
import time
import uuid
from collections import defaultdict
from pathlib import Path

from ml.data.compact_records import (
    MATCH_COLUMNS,
    PARTICIPANT_COLUMNS,
    normalize_match_record,
    normalize_participant_record,
)


DEFAULT_COMPACT_DATASET_DIR = "ml/save_data/lol_dataset_v1"
DEFAULT_COMPRESSION = "zstd"
DEFAULT_MATCH_ROWS_PER_FILE = 250_000
DEFAULT_PARTICIPANT_ROWS_PER_FILE = 1_000_000
KIND_COLUMNS = {
    "matches": MATCH_COLUMNS,
    "participants": PARTICIPANT_COLUMNS,
}


def get_compact_dataset_dir(default=DEFAULT_COMPACT_DATASET_DIR):
    try:
        from ml.runtime_config import get_compact_dataset_dir as get_configured_compact_dataset_dir
    except ImportError:
        return os.getenv("LOL_DRAFT_DATASET_DIR", default)
    return get_configured_compact_dataset_dir()


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("Compact Parquet datasets require pyarrow. Install requirements.txt first.") from exc
    return pa, pq


def _partition_key(record):
    return (
        int(record.get("queue_id") or 0),
        str(record.get("platform") or "unknown").lower(),
        str(record.get("game_date") or "unknown"),
    )


def partition_path(root, kind, record):
    queue_id, platform, game_date = _partition_key(record)
    return Path(root) / kind / f"queue_id={queue_id}" / f"platform={platform}" / f"game_date={game_date}"


def group_records_by_partition(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[_partition_key(record)].append(record)
    return dict(grouped)


def _normalize_records(kind, records):
    if kind == "matches":
        return [normalize_match_record(record) for record in records]
    if kind == "participants":
        return [normalize_participant_record(record) for record in records]
    raise ValueError(f"Unsupported compact dataset kind: {kind}")


class CompactParquetWriter:
    def __init__(
        self,
        dataset_dir=DEFAULT_COMPACT_DATASET_DIR,
        *,
        compression=DEFAULT_COMPRESSION,
        rows_per_file_by_kind=None,
        write_manifest=True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.compression = compression
        self.write_manifest = write_manifest
        self.rows_per_file_by_kind = {
            "matches": DEFAULT_MATCH_ROWS_PER_FILE,
            "participants": DEFAULT_PARTICIPANT_ROWS_PER_FILE,
        }
        if rows_per_file_by_kind:
            self.rows_per_file_by_kind.update(rows_per_file_by_kind)
        self.buffers = {kind: [] for kind in KIND_COLUMNS}

    def add_batch(self, compact_batch):
        self.buffers["matches"].append(compact_batch.match)
        self.buffers["participants"].extend(compact_batch.participants)

    def add_records(self, kind, records):
        if kind not in self.buffers:
            raise ValueError(f"Unsupported compact dataset kind: {kind}")
        self.buffers[kind].extend(records)

    def flush(self):
        written = []
        for kind, records in self.buffers.items():
            if not records:
                continue
            written.extend(self._write_kind(kind, records))
            self.buffers[kind] = []
        if written and self.write_manifest:
            self._append_manifest(written)
        return written

    def _write_kind(self, kind, records):
        _, pq = _require_pyarrow()
        normalized = _normalize_records(kind, records)
        written = []
        rows_per_file = self.rows_per_file_by_kind[kind]
        for key, partition_records in group_records_by_partition(normalized).items():
            queue_id, platform, game_date = key
            output_dir = self.dataset_dir / kind / f"queue_id={queue_id}" / f"platform={platform}" / f"game_date={game_date}"
            output_dir.mkdir(parents=True, exist_ok=True)
            for start in range(0, len(partition_records), rows_per_file):
                chunk = partition_records[start:start + rows_per_file]
                file_path = output_dir / f"part-{int(time.time() * 1000)}-{uuid.uuid4().hex[:10]}.parquet"
                table = _table_from_records(kind, chunk)
                pq.write_table(table, file_path, compression=self.compression, use_dictionary=True, write_statistics=True)
                written.append(
                    {
                        "kind": kind,
                        "path": str(file_path),
                        "rows": len(chunk),
                        "queue_id": queue_id,
                        "platform": platform,
                        "game_date": game_date,
                        "compression": self.compression,
                    }
                )
        return written

    def _append_manifest(self, written_files):
        manifest_path = self.dataset_dir / "manifest.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("a", encoding="utf-8") as handle:
            for item in written_files:
                payload = {"written_at_ts": int(time.time()), **item}
                handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _table_from_records(kind, records):
    pa, _ = _require_pyarrow()
    columns = KIND_COLUMNS[kind]
    normalized = [{column: record.get(column) for column in columns} for record in records]
    return pa.Table.from_pylist(normalized)
