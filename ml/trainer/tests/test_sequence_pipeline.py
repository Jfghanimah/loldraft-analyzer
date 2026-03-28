import json
import sqlite3

from ml.data.match_storage import ensure_match_schema
from ml.trainer.sequence_pipeline import build_sequence_features_for_prediction, build_sequence_training_tensors


def _participant(team_id, puuid, champion, role, seed):
    return {
        "teamId": team_id,
        "puuid": puuid,
        "championName": champion,
        "teamPosition": role,
        "individualPosition": role,
        "goldEarned": 10000 + (seed * 500),
        "kills": 2 + seed,
        "deaths": 1,
        "assists": 3,
        "visionScore": 10 + seed,
        "totalDamageDealtToChampions": 1000 + (100 * seed),
        "totalHeal": 50 + seed,
        "totalHealsOnTeammates": 5,
    }


def _make_match(match_id, game_creation, blue_win, participants):
    return {
        "metadata": {"matchId": match_id},
        "info": {
            "queueId": 420,
            "gameVersion": "15.5.1",
            "gameCreation": game_creation,
            "gameEndTimestamp": game_creation + 1800000,
            "teams": [
                {
                    "teamId": 100,
                    "win": blue_win,
                    "objectives": {
                        "champion": {"first": blue_win},
                        "tower": {"first": blue_win},
                        "dragon": {"kills": 3 if blue_win else 1},
                    },
                },
                {
                    "teamId": 200,
                    "win": not blue_win,
                    "objectives": {
                        "champion": {"first": not blue_win},
                        "tower": {"first": not blue_win},
                        "dragon": {"kills": 1 if blue_win else 3},
                    },
                },
            ],
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


def test_build_sequence_training_tensors_preserves_recent_match_order_and_region(tmp_path):
    db_path = tmp_path / "sequence.db"
    champion_path = tmp_path / "champion_list.json"
    region_path = tmp_path / "region_list.json"
    cache_path = tmp_path / "sequence_cache.pt"
    champion_path.write_text(json.dumps({}, indent=4), encoding="utf-8")
    region_path.write_text(json.dumps({}, indent=4), encoding="utf-8")

    conn = sqlite3.connect(db_path)
    ensure_match_schema(conn)

    roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    blue_puuids = [f"blue_{i}" for i in range(5)]
    red_puuids = [f"red_{i}" for i in range(5)]
    champs_one = ["Aatrox", "Amumu", "Ahri", "Ashe", "Braum", "Renekton", "LeeSin", "Lux", "Jinx", "Nami"]
    champs_two = ["Garen", "Vi", "Orianna", "Caitlyn", "Soraka", "Darius", "Viego", "Syndra", "Ezreal", "Thresh"]

    participants_one = [
        _participant(100, blue_puuids[i], champs_one[i], roles[i], i) for i in range(5)
    ] + [
        _participant(200, red_puuids[i], champs_one[i + 5], roles[i], i) for i in range(5)
    ]
    participants_two = [
        _participant(100, blue_puuids[i], champs_two[i], roles[i], i + 1) for i in range(5)
    ] + [
        _participant(200, red_puuids[i], champs_two[i + 5], roles[i], i + 1) for i in range(5)
    ]

    match_one = _make_match("NA1_1", 100, True, participants_one)
    match_two = _make_match("NA1_2", 200, False, participants_two)

    for match_id, raw_match, ordered, game_creation in (
        ("NA1_1", match_one, _ordered_record(True, champs_one), 100),
        ("NA1_2", match_two, _ordered_record(False, champs_two), 200),
    ):
        blue_gold = sum(
            participant["goldEarned"]
            for participant in raw_match["info"]["participants"]
            if participant["teamId"] == 100
        )
        red_gold = sum(
            participant["goldEarned"]
            for participant in raw_match["info"]["participants"]
            if participant["teamId"] == 200
        )
        blue_dragons = raw_match["info"]["teams"][0]["objectives"]["dragon"]["kills"]
        red_dragons = raw_match["info"]["teams"][1]["objectives"]["dragon"]["kills"]
        conn.execute(
            """
            INSERT INTO matches (
                match_id, match_data, region, raw_match_json, ordered_match_json,
                data_source, collector_id, game_version, queue_id, game_creation, game_end_timestamp,
                blue_first_blood, blue_first_tower, blue_dragon_share, blue_gold_share, last_updated_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                match_id,
                json.dumps(ordered),
                "na1",
                None,
                json.dumps(ordered),
                "riot_api_match_v5",
                "tester",
                "15.5.1",
                420,
                game_creation,
                game_creation + 90,
                1 if ordered["blue_win"] else 0,
                1 if ordered["blue_win"] else 0,
                blue_dragons / (blue_dragons + red_dragons),
                blue_gold / (blue_gold + red_gold),
                game_creation,
            ),
        )

    conn.executemany(
        """
        INSERT INTO participant_history (
            match_id, puuid, queue_id, game_creation, champion_name, role, team_id, win,
            kills, deaths, assists, vision_score, damage_to_champions, healing, gold_earned, cs, game_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                match_id,
                participant["puuid"],
                420,
                game_creation,
                participant["championName"],
                participant["teamPosition"],
                participant["teamId"],
                1 if participant["teamId"] == 100 else 0,
                participant["kills"],
                participant["deaths"],
                participant["assists"],
                participant["visionScore"],
                participant["totalDamageDealtToChampions"],
                participant["totalHeal"] + participant["totalHealsOnTeammates"],
                participant["goldEarned"],
                participant.get("totalMinionsKilled", 0) + participant.get("neutralMinionsKilled", 0),
                "15.5.1",
            )
            for match_id, game_creation, participants in (
                ("NA1_1", 100, participants_one),
                ("NA1_2", 200, participants_two),
            )
            for participant in participants
        ],
    )
    conn.commit()
    conn.close()

    tensors, metadata = build_sequence_training_tensors(
        db_path=str(db_path),
        champion_path=str(champion_path),
        region_path=str(region_path),
        cache_path=str(cache_path),
        history_length=3,
    )

    assert tensors["labels"].shape[0] == 2
    assert metadata["region_list"]["na1"] == 0
    assert tensors["region_ids"].tolist() == [0, 0]
    assert tensors["history_mask"][0].sum().item() == 0
    assert tensors["history_mask"][1].sum().item() == 10
    assert tensors["history_champion_ids"][1, 0, 0].item() == metadata["champion_list"]["Aatrox"]
    assert tensors["history_result_ids"][1, 0, 0].item() == 1
    assert tensors["history_numeric"][1, 0, 0, 5].item() > 0.0
    assert tensors["history_numeric"][1, 0, 0, 6].item() >= 0.0
    assert tensors["outcome_targets"].shape == (2, 4)
    assert tensors["outcome_targets"][0, 0].item() == 1.0


def test_build_sequence_features_for_prediction_uses_recent_rows_for_each_player(tmp_path):
    db_path = tmp_path / "predict_sequence.db"
    champion_path = tmp_path / "champion_list.json"
    region_path = tmp_path / "region_list.json"
    champion_path.write_text(json.dumps({"Aatrox": 0, "Amumu": 1, "Ahri": 2, "Ashe": 3, "Braum": 4, "Renekton": 5, "LeeSin": 6, "Lux": 7, "Jinx": 8, "Nami": 9}, indent=4), encoding="utf-8")
    region_path.write_text(json.dumps({"na1": 0}, indent=4), encoding="utf-8")

    conn = sqlite3.connect(db_path)
    ensure_match_schema(conn)
    conn.executemany(
        """
        INSERT INTO participant_history (
            match_id, puuid, queue_id, game_creation, champion_name, role, team_id, win,
            kills, deaths, assists, vision_score, damage_to_champions, healing, gold_earned, cs, game_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("NA1_1", "p1", 420, 100, "Aatrox", "TOP", 100, 1, 5, 2, 4, 20.0, 2000.0, 80.0, 12000.0, 210.0, "15.5.1"),
            ("NA1_2", "p1", 420, 200, "Garen", "TOP", 100, 0, 1, 4, 2, 10.0, 1000.0, 40.0, 9000.0, 150.0, "15.5.1"),
        ],
    )
    conn.execute(
        """
        INSERT INTO matches (
            match_id, match_data, region, raw_match_json, ordered_match_json,
            data_source, collector_id, game_version, queue_id, game_creation, game_end_timestamp, last_updated_ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "NA1_3",
            "{}",
            "na1",
            "{}",
            "{}",
            "riot_api_match_v5",
            "tester",
            "15.5.1",
            420,
            300,
            390,
            3,
        ),
    )

    features = build_sequence_features_for_prediction(
        conn,
        champions=["Aatrox", "Amumu", "Ahri", "Ashe", "Braum", "Renekton", "LeeSin", "Lux", "Jinx", "Nami"],
        players=["p1", None, None, None, None, None, None, None, None, None],
        region="na1",
        current_game_creation=300,
        champion_path=str(champion_path),
        region_path=str(region_path),
        history_length=3,
    )
    conn.close()

    assert features["current_champion_ids"].shape == (1, 10)
    assert features["history_champion_ids"].shape == (1, 10, 3)
    assert features["history_mask"][0, 0, 0].item() is True
    assert features["history_result_ids"][0, 0, 0].item() == 0
    assert features["history_numeric"][0, 0, 0, 5].item() > 0.0
    assert features["history_numeric"][0, 0, 0, 6].item() > 0.0
    assert features["region_ids"][0].item() == 0
