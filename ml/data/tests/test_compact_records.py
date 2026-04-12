from ml.data.compact_records import (
    MATCH_COLUMNS,
    PARTICIPANT_COLUMNS,
    extract_compact_records,
    game_date_from_creation,
    match_record_from_ordered_json_row,
    participant_record_from_history_row,
)


def _sample_match():
    return {
        "metadata": {"matchId": "NA1_1"},
        "info": {
            "queueId": 420,
            "gameVersion": "15.5.1",
            "gameCreation": 1_700_000_000_000,
            "gameEndTimestamp": 1_700_001_800_000,
            "teams": [
                {
                    "teamId": 100,
                    "win": True,
                    "objectives": {
                        "champion": {"first": True, "kills": 30},
                        "tower": {"first": True, "kills": 8},
                        "dragon": {"kills": 3},
                        "baron": {"kills": 1},
                        "riftHerald": {"kills": 1},
                        "inhibitor": {"kills": 1},
                    },
                },
                {
                    "teamId": 200,
                    "win": False,
                    "objectives": {
                        "champion": {"first": False, "kills": 20},
                        "tower": {"first": False, "kills": 3},
                        "dragon": {"kills": 1},
                        "baron": {"kills": 0},
                        "riftHerald": {"kills": 0},
                        "inhibitor": {"kills": 0},
                    },
                },
            ],
            "participants": [
                _participant(100, "p1", "Aatrox", 266, "TOP", 5, 2, 7, 12000, 220),
                _participant(100, "p2", "Amumu", 32, "JUNGLE", 4, 3, 8, 11000, 160),
                _participant(100, "p3", "Ahri", 103, "MIDDLE", 6, 1, 6, 13000, 230),
                _participant(100, "p4", "Ashe", 22, "BOTTOM", 8, 2, 5, 14000, 260),
                _participant(100, "p5", "Braum", 201, "UTILITY", 1, 4, 15, 9000, 45),
                _participant(200, "p6", "Renekton", 58, "TOP", 3, 5, 4, 10000, 200),
                _participant(200, "p7", "LeeSin", 64, "JUNGLE", 2, 5, 8, 9500, 150),
                _participant(200, "p8", "Lux", 99, "MIDDLE", 5, 5, 5, 10500, 210),
                _participant(200, "p9", "Jinx", 222, "BOTTOM", 7, 4, 3, 12000, 250),
                _participant(200, "p10", "Nami", 267, "UTILITY", 0, 6, 12, 8000, 35),
            ],
        },
    }


def _participant(team_id, puuid, champion_name, champion_id, role, kills, deaths, assists, gold, cs):
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
        "goldEarned": gold,
        "totalMinionsKilled": cs,
        "neutralMinionsKilled": 0,
    }


def test_extract_compact_records_preserves_training_facts_without_raw_json():
    batch, reason = extract_compact_records(_sample_match(), "na1", collector_id="tester", collected_at_ts=123)

    assert reason is None
    assert set(batch.match) == set(MATCH_COLUMNS)
    assert len(batch.participants) == 10
    assert set(batch.participants[0]) == set(PARTICIPANT_COLUMNS)
    assert "raw_match_json" not in batch.match
    assert batch.match["match_id"] == "NA1_1"
    assert batch.match["label"] == 1
    assert batch.match["champion_0"] == "Aatrox"
    assert batch.match["champion_id_9"] == 267
    assert batch.match["blue_dragons"] == 3
    assert batch.match["red_dragons"] == 1
    assert batch.match["gold_diff"] == 9000.0
    assert batch.participants[0]["slot"] == 0
    assert batch.participants[0]["role"] == "TOP"
    assert batch.participants[9]["side"] == "red"


def test_game_date_from_creation_uses_utc_date():
    assert game_date_from_creation(1_700_000_000_000) == "2023-11-14"


def test_match_record_from_ordered_json_row_uses_structured_sqlite_columns():
    row = (
        "NA1_1",
        '{"blue_win": true, "blue_side": 1, "position_source": "teamPosition", "champions": ["Aatrox", "Amumu", "Ahri", "Ashe", "Braum", "Renekton", "LeeSin", "Lux", "Jinx", "Nami"]}',
        "na1",
        "15.5.1",
        420,
        1_700_000_000_000,
        1_700_001_800_000,
        1,
        1,
        0.75,
        0.55,
        3,
        1,
        10000.0,
        30.0,
        "tester",
        123,
    )

    record = match_record_from_ordered_json_row(row)

    assert record["match_id"] == "NA1_1"
    assert record["champion_9"] == "Nami"
    assert record["duration_minutes"] == 30.0
    assert record["blue_gold_share"] == 0.55


def test_participant_record_from_history_row_derives_slot_and_rates():
    row = (
        "NA1_1",
        "na1",
        420,
        1_700_000_000_000,
        1_700_001_800_000,
        "15.5.1",
        "p1",
        "Aatrox",
        "TOP",
        100,
        1,
        5,
        2,
        7,
        20.0,
        20000.0,
        150.0,
        12000.0,
        220.0,
    )

    record = participant_record_from_history_row(row)

    assert record["slot"] == 0
    assert record["side"] == "blue"
    assert record["duration_minutes"] == 30.0
    assert record["kda_value"] == 6.0
    assert record["gpm_value"] == 400.0
