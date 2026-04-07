from ml.data.export_training_parquet import (
    _build_rowid_ranges,
    _match_record_from_row,
    _participant_history_record_from_row,
)


def test_build_rowid_ranges_handles_empty_and_non_empty_bounds():
    assert _build_rowid_ranges(None, None, 100) == []
    assert _build_rowid_ranges(1, 5, 2) == [(1, 2), (3, 4), (5, 5)]


def test_match_record_from_row_extracts_compact_training_fields():
    row = (
        "NA1_1",
        '{"blue_win": true, "champions": ["Aatrox", "Amumu", "Ahri", "Ashe", "Braum", "Renekton", "LeeSin", "Lux", "Jinx", "Nami"]}',
        "na1",
        "15.5.1",
        420,
        123456,
        125256,
        3,
        1,
        24500.0,
        30.0,
    )

    record = _match_record_from_row(row)

    assert record["match_id"] == "NA1_1"
    assert record["label"] == 1
    assert record["region"] == "na1"
    assert record["target_blue_dragons"] == 3.0
    assert record["target_red_dragons"] == 1.0
    assert record["target_gold_diff"] == 24500.0
    assert record["target_game_length_minutes"] == 30.0
    assert record["champion_0"] == "Aatrox"
    assert record["champion_9"] == "Nami"


def test_participant_history_record_from_row_preserves_compact_columns():
    row = (
        "NA1_1",
        "puuid-1",
        420,
        123456,
        "Aatrox",
        "TOP",
        100,
        1,
        5,
        2,
        7,
        20.0,
        2000.0,
        100.0,
        12000.0,
        220.0,
        "15.5.1",
    )

    record = _participant_history_record_from_row(row)

    assert record["match_id"] == "NA1_1"
    assert record["puuid"] == "puuid-1"
    assert record["queue_id"] == 420
    assert record["champion_name"] == "Aatrox"
    assert record["role"] == "TOP"
    assert record["team_id"] == 100
    assert record["gold_earned"] == 12000.0
    assert record["cs"] == 220.0
