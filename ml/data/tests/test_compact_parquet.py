import pytest

from ml.data.compact_parquet import CompactParquetWriter, group_records_by_partition, partition_path
from ml.data.compact_records import CompactMatchBatch, normalize_match_record, normalize_participant_record


def test_group_records_by_partition_groups_by_queue_platform_and_date():
    records = [
        {"queue_id": 420, "platform": "na1", "game_date": "2026-04-01", "match_id": "NA1_1"},
        {"queue_id": 420, "platform": "na1", "game_date": "2026-04-01", "match_id": "NA1_2"},
        {"queue_id": 420, "platform": "euw1", "game_date": "2026-04-01", "match_id": "EUW1_1"},
    ]

    grouped = group_records_by_partition(records)

    assert len(grouped) == 2
    assert len(grouped[(420, "na1", "2026-04-01")]) == 2
    assert len(grouped[(420, "euw1", "2026-04-01")]) == 1


def test_partition_path_uses_hive_style_directories(tmp_path):
    path = partition_path(
        tmp_path,
        "matches",
        {"queue_id": 420, "platform": "na1", "game_date": "2026-04-01"},
    )

    assert path == tmp_path / "matches" / "queue_id=420" / "platform=na1" / "game_date=2026-04-01"


def test_compact_parquet_writer_flushes_partitioned_files(tmp_path):
    pytest.importorskip("pyarrow")
    batch = CompactMatchBatch(
        match=normalize_match_record(
            {
                "match_id": "NA1_1",
                "queue_id": 420,
                "platform": "na1",
                "game_date": "2026-04-01",
                "label": 1,
            }
        ),
        participants=[
            normalize_participant_record(
                {
                    "match_id": "NA1_1",
                    "queue_id": 420,
                    "platform": "na1",
                    "game_date": "2026-04-01",
                    "puuid": "p1",
                    "slot": 0,
                }
            )
        ],
    )

    writer = CompactParquetWriter(tmp_path)
    writer.add_batch(batch)
    written = writer.flush()

    assert len(written) == 2
    assert (tmp_path / "manifest.jsonl").exists()
    assert list((tmp_path / "matches").glob("queue_id=420/platform=na1/game_date=*/part-*.parquet"))
    assert list((tmp_path / "participants").glob("queue_id=420/platform=na1/game_date=*/part-*.parquet"))
