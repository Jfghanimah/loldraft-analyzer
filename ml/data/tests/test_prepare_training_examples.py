import json

import pytest

from ml.data.compact_parquet import CompactParquetWriter
from ml.data.compact_records import extract_compact_records
from ml.data.prepare_training_examples import load_training_examples_dataframe, prepare_training_examples
from ml.features.recent_history import dense_feature_columns
from ml.data.match_format import ROLE_ORDER


def _participant(team_id, puuid, champion_name, champion_id, role, kills, deaths, assists):
    return {
        "teamId": team_id,
        "puuid": puuid,
        "championName": champion_name,
        "championId": champion_id,
        "teamPosition": role,
        "individualPosition": role,
        "kills": kills,
        "deaths": deaths,
        "assists": assists,
        "visionScore": 20,
        "totalDamageDealtToChampions": 20000,
        "totalHeal": 100,
        "totalHealsOnTeammates": 50,
        "goldEarned": 12000,
        "totalMinionsKilled": 200,
        "neutralMinionsKilled": 20,
    }


def _match(match_id, game_creation):
    return {
        "metadata": {"matchId": match_id},
        "info": {
            "queueId": 420,
            "gameVersion": "15.5.1",
            "gameCreation": game_creation,
            "gameEndTimestamp": game_creation + 1_800_000,
            "teams": [
                {"teamId": 100, "win": True, "objectives": {"dragon": {"kills": 3}}},
                {"teamId": 200, "win": False, "objectives": {"dragon": {"kills": 1}}},
            ],
            "participants": [
                _participant(100, "p1", "Aatrox", 266, "TOP", 5, 2, 7),
                _participant(100, "p2", "Amumu", 32, "JUNGLE", 4, 3, 8),
                _participant(100, "p3", "Ahri", 103, "MIDDLE", 6, 1, 6),
                _participant(100, "p4", "Ashe", 22, "BOTTOM", 8, 2, 5),
                _participant(100, "p5", "Braum", 201, "UTILITY", 1, 4, 15),
                _participant(200, "p6", "Renekton", 58, "TOP", 3, 5, 4),
                _participant(200, "p7", "LeeSin", 64, "JUNGLE", 2, 5, 8),
                _participant(200, "p8", "Lux", 99, "MIDDLE", 5, 5, 5),
                _participant(200, "p9", "Jinx", 222, "BOTTOM", 7, 4, 3),
                _participant(200, "p10", "Nami", 267, "UTILITY", 0, 6, 12),
            ],
        },
    }


def _read_training_examples(output_dir):
    import pyarrow.parquet as pq

    records = []
    for path in sorted((output_dir / "training_examples").glob("queue_id=*/platform=*/game_date=*/part-*.parquet")):
        records.extend(pq.ParquetFile(path).read().to_pylist())
    return sorted(records, key=lambda row: row["game_creation"])


def test_prepare_training_examples_builds_prior_history_features(tmp_path):
    pytest.importorskip("pyarrow")
    dataset_dir = tmp_path / "compact"
    writer = CompactParquetWriter(dataset_dir)
    for match in (_match("NA1_1", 1_700_000_000_000), _match("NA1_2", 1_700_100_000_000)):
        batch, reason = extract_compact_records(match, "na1", collector_id="tester", collected_at_ts=123)
        assert reason is None
        writer.add_batch(batch)
    writer.flush()

    champion_path = tmp_path / "champion_list.json"
    region_path = tmp_path / "region_list.json"
    result = prepare_training_examples(
        dataset_dir=dataset_dir,
        champion_path=champion_path,
        region_path=region_path,
        rows_per_file=10,
    )
    records = _read_training_examples(dataset_dir)
    games_played_col = dense_feature_columns(ROLE_ORDER)[0]

    assert result["examples"] == 2
    assert len(records) == 2
    assert records[0][games_played_col] == 0.0
    assert records[1][games_played_col] > 0.0
    assert records[0]["champion_0"] == records[1]["champion_0"]
    assert json.loads(champion_path.read_text(encoding="utf-8"))["Aatrox"] == records[0]["champion_0"]

    dataframe, champion_list = load_training_examples_dataframe(dataset_dir, champion_path=champion_path)
    assert "match_id" not in dataframe.columns
    assert len(dataframe) == 2
    assert champion_list["Aatrox"] == records[0]["champion_0"]
