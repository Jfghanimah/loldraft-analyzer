import json
import sqlite3

from ml.features.recent_history import PARTICIPANT_FEATURES
from ml.trainer.feature_pipeline import build_dense_features_for_prediction, build_rich_feature_dataframe
from ml.data.match_storage import ensure_match_schema


def _make_match(match_id, game_creation, blue_win, participants):
    return {
        "metadata": {"matchId": match_id},
        "info": {
            "queueId": 420,
            "gameVersion": "15.5.1",
            "gameCreation": game_creation,
            "gameEndTimestamp": game_creation + 1800000,
            "teams": [{"teamId": 100, "win": blue_win}, {"teamId": 200, "win": not blue_win}],
            "participants": participants,
        },
    }


def _ordered_record(blue_win, champions):
    return {
        "format": "role_order_v1",
        "blue_win": blue_win,
        "blue_side": 1,
        "champions": champions,
        "role_order": ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"],
        "position_source": "teamPosition",
        "game_version": "15.5.1",
        "queue_id": 420,
    }


def _participant(team_id, puuid, champion, role, win_stats_seed):
    return {
        "teamId": team_id,
        "puuid": puuid,
        "championName": champion,
        "teamPosition": role,
        "individualPosition": role,
        "kills": 2 + win_stats_seed,
        "deaths": 1,
        "assists": 3,
        "visionScore": 10 + win_stats_seed,
        "totalDamageDealtToChampions": 1000 + (100 * win_stats_seed),
        "totalHeal": 50 + win_stats_seed,
        "totalHealsOnTeammates": 5,
    }


def test_build_rich_feature_dataframe_generates_dense_features_from_prior_matches(tmp_path):
    db_path = tmp_path / "rich.db"
    champion_path = tmp_path / "champion_list.json"
    champion_path.write_text(json.dumps({}, indent=4), encoding="utf-8")

    conn = sqlite3.connect(db_path)
    ensure_match_schema(conn)

    roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    blue_puuids = [f"blue_{i}" for i in range(5)]
    red_puuids = [f"red_{i}" for i in range(5)]
    champs_one = ["Aatrox", "Amumu", "Ahri", "Ashe", "Braum", "Renekton", "LeeSin", "Lux", "Jinx", "Nami"]
    champs_two = ["Aatrox", "Amumu", "Ahri", "Ashe", "Braum", "Darius", "Vi", "Annie", "Caitlyn", "Soraka"]

    match_one_participants = [
        _participant(100, blue_puuids[i], champs_one[i], roles[i], i) for i in range(5)
    ] + [
        _participant(200, red_puuids[i], champs_one[i + 5], roles[i], i) for i in range(5)
    ]
    match_two_participants = [
        _participant(100, blue_puuids[i], champs_two[i], roles[i], i + 1) for i in range(5)
    ] + [
        _participant(200, red_puuids[i], champs_two[i + 5], roles[i], i + 1) for i in range(5)
    ]

    match_one = _make_match("NA1_1", 100, True, match_one_participants)
    match_two = _make_match("NA1_2", 200, False, match_two_participants)

    conn.execute(
        """
        INSERT INTO matches (
            match_id, match_data, region, raw_match_json, ordered_match_json,
            data_source, collector_id, game_version, queue_id, game_creation, game_end_timestamp, last_updated_ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "NA1_1",
            json.dumps(_ordered_record(True, champs_one)),
            "na1",
            json.dumps(match_one),
            json.dumps(_ordered_record(True, champs_one)),
            "riot_api_match_v5",
            "tester",
            "15.5.1",
            420,
            100,
            190,
            1,
        ),
    )
    conn.execute(
        """
        INSERT INTO matches (
            match_id, match_data, region, raw_match_json, ordered_match_json,
            data_source, collector_id, game_version, queue_id, game_creation, game_end_timestamp, last_updated_ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "NA1_2",
            json.dumps(_ordered_record(False, champs_two)),
            "na1",
            json.dumps(match_two),
            json.dumps(_ordered_record(False, champs_two)),
            "riot_api_match_v5",
            "tester",
            "15.5.1",
            420,
            200,
            290,
            2,
        ),
    )
    conn.executemany(
        """
        INSERT INTO participant_history (
            match_id, puuid, queue_id, game_creation, champion_name, role, team_id, win,
            kills, deaths, assists, vision_score, damage_to_champions, healing, game_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                match_id,
                p["puuid"],
                420,
                game_creation,
                p["championName"],
                p["teamPosition"],
                p["teamId"],
                1 if p["teamId"] == 100 else 0,
                p["kills"],
                p["deaths"],
                p["assists"],
                p["visionScore"],
                p["totalDamageDealtToChampions"],
                p["totalHeal"] + p["totalHealsOnTeammates"],
                "15.5.1",
            )
            for match_id, game_creation, participants in (
                ("NA1_1", 100, match_one_participants),
                ("NA1_2", 200, match_two_participants),
            )
            for p in participants
        ],
    )
    conn.commit()
    conn.close()

    df, champion_list = build_rich_feature_dataframe(
        db_path=str(db_path),
        champion_path=str(champion_path),
    )

    assert len(df) == 2
    assert len(champion_list) >= 12
    assert df.shape[1] == 164

    dense_columns = list(df.columns[12:])
    dense_sums = sorted(float(df.iloc[i][dense_columns].sum()) for i in range(len(df)))

    assert dense_sums[0] >= 0.0
    assert dense_sums[1] > dense_sums[0]
    assert f"slot_0_blue_top_{PARTICIPANT_FEATURES[0]}" in dense_columns
    assert "patch_major" in dense_columns
    assert "patch_minor" in dense_columns


def test_build_dense_features_for_prediction_uses_recent_prior_rows_only(tmp_path):
    db_path = tmp_path / "predict.db"
    conn = sqlite3.connect(db_path)
    ensure_match_schema(conn)

    conn.executemany(
        """
        INSERT INTO participant_history (
            match_id, puuid, queue_id, game_creation, champion_name, role, team_id, win,
            kills, deaths, assists, vision_score, damage_to_champions, healing, game_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("NA1_1", "p1", 420, 100, "Aatrox", "TOP", 100, 1, 4, 2, 5, 20.0, 2000.0, 80.0, "15.5.1"),
            ("NA1_2", "p1", 420, 200, "Garen", "TOP", 100, 0, 1, 4, 2, 10.0, 1000.0, 40.0, "15.5.1"),
            ("NA1_3", "p1", 420, 500, "Aatrox", "TOP", 100, 1, 6, 1, 7, 25.0, 2500.0, 120.0, "15.5.1"),
            ("NA1_4", "p1", 420, 900, "Darius", "TOP", 100, 1, 8, 3, 4, 30.0, 3000.0, 140.0, "15.5.1"),
        ],
    )

    features = build_dense_features_for_prediction(
        conn,
        champions=["Aatrox", "Amumu", "Ahri", "Ashe", "Braum", "Renekton", "LeeSin", "Lux", "Jinx", "Nami"],
        players=["p1", None, None, None, None, None, None, None, None, None],
        current_game_creation=700,
    )
    conn.close()

    assert len(features) == 152
    top_offset = 0
    games_played = features[top_offset]
    champ_games = features[top_offset + 2]
    games_last_3d = features[top_offset + 10]
    hours_since_last = features[top_offset + 12]
    unique_champions = features[top_offset + 13]
    patch_major = features[-2]
    patch_minor = features[-1]

    assert games_played > 0.0
    assert champ_games > 0.0
    assert games_last_3d > 0.0
    assert 0.0 < hours_since_last < 1.0
    assert unique_champions > 0.0
    assert patch_major == 0.0
    assert patch_minor == 0.0


def test_training_and_prediction_share_recent_history_feature_values(tmp_path):
    db_path = tmp_path / "shared.db"
    champion_path = tmp_path / "champion_list.json"
    champion_path.write_text(json.dumps({}, indent=4), encoding="utf-8")

    conn = sqlite3.connect(db_path)
    ensure_match_schema(conn)

    roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    blue_puuids = [f"blue_{i}" for i in range(5)]
    red_puuids = [f"red_{i}" for i in range(5)]
    champs_one = ["Aatrox", "Amumu", "Ahri", "Ashe", "Braum", "Renekton", "LeeSin", "Lux", "Jinx", "Nami"]
    champs_two = ["Aatrox", "Amumu", "Ahri", "Ashe", "Braum", "Darius", "Vi", "Annie", "Caitlyn", "Soraka"]

    match_one_participants = [
        _participant(100, blue_puuids[i], champs_one[i], roles[i], i) for i in range(5)
    ] + [
        _participant(200, red_puuids[i], champs_one[i + 5], roles[i], i) for i in range(5)
    ]
    match_two_participants = [
        _participant(100, blue_puuids[i], champs_two[i], roles[i], i + 1) for i in range(5)
    ] + [
        _participant(200, red_puuids[i], champs_two[i + 5], roles[i], i + 1) for i in range(5)
    ]

    match_one = _make_match("NA1_1", 100, True, match_one_participants)
    match_two = _make_match("NA1_2", 200, False, match_two_participants)

    conn.execute(
        """
        INSERT INTO matches (
            match_id, match_data, region, raw_match_json, ordered_match_json,
            data_source, collector_id, game_version, queue_id, game_creation, game_end_timestamp, last_updated_ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "NA1_1",
            json.dumps(_ordered_record(True, champs_one)),
            "na1",
            json.dumps(match_one),
            json.dumps(_ordered_record(True, champs_one)),
            "riot_api_match_v5",
            "tester",
            "15.5.1",
            420,
            100,
            190,
            1,
        ),
    )
    conn.execute(
        """
        INSERT INTO matches (
            match_id, match_data, region, raw_match_json, ordered_match_json,
            data_source, collector_id, game_version, queue_id, game_creation, game_end_timestamp, last_updated_ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "NA1_2",
            json.dumps(_ordered_record(False, champs_two)),
            "na1",
            json.dumps(match_two),
            json.dumps(_ordered_record(False, champs_two)),
            "riot_api_match_v5",
            "tester",
            "15.5.1",
            420,
            200,
            290,
            2,
        ),
    )
    conn.executemany(
        """
        INSERT INTO participant_history (
            match_id, puuid, queue_id, game_creation, champion_name, role, team_id, win,
            kills, deaths, assists, vision_score, damage_to_champions, healing, game_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                match_id,
                p["puuid"],
                420,
                game_creation,
                p["championName"],
                p["teamPosition"],
                p["teamId"],
                1 if p["teamId"] == 100 else 0,
                p["kills"],
                p["deaths"],
                p["assists"],
                p["visionScore"],
                p["totalDamageDealtToChampions"],
                p["totalHeal"] + p["totalHealsOnTeammates"],
                "15.5.1",
            )
            for match_id, game_creation, participants in (
                ("NA1_1", 100, match_one_participants),
                ("NA1_2", 200, match_two_participants),
            )
            for p in participants
        ],
    )
    conn.commit()

    df, _ = build_rich_feature_dataframe(
        db_path=str(db_path),
        champion_path=str(champion_path),
    )
    prediction_features = build_dense_features_for_prediction(
        conn,
        champs_two,
        players=blue_puuids + red_puuids,
        current_game_creation=200,
    )
    conn.close()

    dense_columns = list(df.columns[12:])
    richer_row = max((df.iloc[i][dense_columns].tolist() for i in range(len(df))), key=sum)
    assert richer_row == prediction_features


def test_build_rich_feature_dataframe_falls_back_to_raw_json_when_participant_history_missing(tmp_path, capsys):
    db_path = tmp_path / "fallback.db"
    champion_path = tmp_path / "champion_list.json"
    champion_path.write_text(json.dumps({}, indent=4), encoding="utf-8")

    conn = sqlite3.connect(db_path)
    ensure_match_schema(conn)

    roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    blue_puuids = [f"blue_{i}" for i in range(5)]
    red_puuids = [f"red_{i}" for i in range(5)]
    champs = ["Aatrox", "Amumu", "Ahri", "Ashe", "Braum", "Renekton", "LeeSin", "Lux", "Jinx", "Nami"]
    participants = [
        _participant(100, blue_puuids[i], champs[i], roles[i], i) for i in range(5)
    ] + [
        _participant(200, red_puuids[i], champs[i + 5], roles[i], i) for i in range(5)
    ]
    match = _make_match("NA1_1", 100, True, participants)

    conn.execute(
        """
        INSERT INTO matches (
            match_id, match_data, region, raw_match_json, ordered_match_json,
            data_source, collector_id, game_version, queue_id, game_creation, game_end_timestamp, last_updated_ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "NA1_1",
            json.dumps(_ordered_record(True, champs)),
            "na1",
            json.dumps(match),
            json.dumps(_ordered_record(True, champs)),
            "riot_api_match_v5",
            "tester",
            "15.5.1",
            420,
            100,
            190,
            1,
        ),
    )
    conn.commit()
    conn.close()

    df, champion_list = build_rich_feature_dataframe(
        db_path=str(db_path),
        champion_path=str(champion_path),
    )
    captured = capsys.readouterr()

    assert len(df) == 1
    assert len(champion_list) == 10
    assert "fell back to raw_match_json" in captured.out
