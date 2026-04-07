import json
import sqlite3

from ml.data.match_storage import (
    connect_sqlite,
    ensure_match_schema,
    extract_storage_payload,
    rebuild_participant_history,
    upsert_match_record,
)
from ml.data.merge_match_dbs import merge_matches


def test_ensure_match_schema_adds_richer_columns(tmp_path):
    db_path = tmp_path / "matches.db"
    conn = sqlite3.connect(db_path)

    conn.execute("CREATE TABLE matches (match_id TEXT PRIMARY KEY, match_data TEXT, region TEXT)")
    ensure_match_schema(conn)

    columns = {row[1] for row in conn.execute("PRAGMA table_info(matches)").fetchall()}
    assert "raw_match_json" in columns
    assert "ordered_match_json" in columns
    assert "rank_snapshot_json" in columns
    assert "game_version" in columns
    assert "queue_id" in columns
    assert "blue_first_blood" in columns
    assert "blue_first_tower" in columns
    assert "blue_dragon_share" in columns
    assert "blue_gold_share" in columns
    assert "blue_dragons" in columns
    assert "red_dragons" in columns
    assert "gold_diff" in columns
    assert "game_length_minutes" in columns
    assert "last_updated_ts" in columns
    match_indexes = {row[1] for row in conn.execute("PRAGMA index_list(matches)").fetchall()}
    assert "idx_matches_queue_game_match" in match_indexes
    participant_columns = {row[1] for row in conn.execute("PRAGMA table_info(participant_history)").fetchall()}
    assert "puuid" in participant_columns
    assert "champion_name" in participant_columns
    assert "role" in participant_columns
    assert "gold_earned" in participant_columns
    assert "cs" in participant_columns
    participant_indexes = {row[1] for row in conn.execute("PRAGMA index_list(participant_history)").fetchall()}
    assert "idx_participant_history_queue_game_match_team" in participant_indexes
    conn.close()


def test_connect_sqlite_enables_wal_for_writer(tmp_path):
    db_path = tmp_path / "matches.db"
    conn = connect_sqlite(db_path)
    journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()

    assert journal_mode.lower() == "wal"


def test_upsert_match_record_persists_raw_and_ordered_payloads(tmp_path):
    db_path = tmp_path / "matches.db"
    conn = sqlite3.connect(db_path)
    ensure_match_schema(conn)

    match = {
        "metadata": {"matchId": "NA1_1"},
        "info": {
            "gameVersion": "15.1.1",
            "queueId": 420,
            "gameCreation": 123456,
            "gameEndTimestamp": 123999,
            "teams": [
                {
                    "teamId": 100,
                    "win": True,
                    "objectives": {
                        "champion": {"first": True},
                        "tower": {"first": True},
                        "dragon": {"kills": 3},
                    },
                },
                {
                    "teamId": 200,
                    "win": False,
                    "objectives": {
                        "champion": {"first": False},
                        "tower": {"first": False},
                        "dragon": {"kills": 1},
                    },
                },
            ],
            "participants": [
                {"teamId": 100, "puuid": "p1", "championName": "Aatrox", "teamPosition": "TOP", "individualPosition": "TOP", "goldEarned": 10000},
                {"teamId": 100, "puuid": "p2", "championName": "Amumu", "teamPosition": "JUNGLE", "individualPosition": "JUNGLE", "goldEarned": 11000},
                {"teamId": 100, "puuid": "p3", "championName": "Ahri", "teamPosition": "MIDDLE", "individualPosition": "MIDDLE", "goldEarned": 12000},
                {"teamId": 100, "puuid": "p4", "championName": "Ashe", "teamPosition": "BOTTOM", "individualPosition": "BOTTOM", "goldEarned": 13000},
                {"teamId": 100, "puuid": "p5", "championName": "Braum", "teamPosition": "UTILITY", "individualPosition": "UTILITY", "goldEarned": 14000},
                {"teamId": 200, "puuid": "p6", "championName": "Renekton", "teamPosition": "TOP", "individualPosition": "TOP", "goldEarned": 9000},
                {"teamId": 200, "puuid": "p7", "championName": "LeeSin", "teamPosition": "JUNGLE", "individualPosition": "JUNGLE", "goldEarned": 9500},
                {"teamId": 200, "puuid": "p8", "championName": "Lux", "teamPosition": "MIDDLE", "individualPosition": "MIDDLE", "goldEarned": 9800},
                {"teamId": 200, "puuid": "p9", "championName": "Jinx", "teamPosition": "BOTTOM", "individualPosition": "BOTTOM", "goldEarned": 10000},
                {"teamId": 200, "puuid": "p10", "championName": "Nami", "teamPosition": "UTILITY", "individualPosition": "UTILITY", "goldEarned": 10200},
            ],
        },
    }
    ordered = {
        "format": "role_order_v1",
        "blue_win": True,
        "blue_side": 1,
        "champions": ["Aatrox", "Amumu", "Ahri", "Ashe", "Braum", "Renekton", "LeeSin", "Lux", "Jinx", "Nami"],
        "role_order": ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"],
        "position_source": "teamPosition",
        "game_version": "15.1.1",
        "queue_id": 420,
    }

    rank_snapshots = [{"puuid": "p1", "queue_type": "RANKED_SOLO_5x5", "status": "ranked", "tier": "EMERALD"}]
    payload = extract_storage_payload(match, "NA1", ordered, collector_id="tester", rank_snapshots=rank_snapshots)
    upsert_match_record(conn, "NA1_1", payload)

    row = conn.execute(
        """
        SELECT region, raw_match_json, ordered_match_json, rank_snapshot_json, game_version, queue_id,
               blue_first_blood, blue_first_tower, blue_dragon_share, blue_gold_share,
               blue_dragons, red_dragons, gold_diff, game_length_minutes
        FROM matches WHERE match_id = ?
        """,
        ("NA1_1",),
    ).fetchone()
    participant_rows = conn.execute(
        "SELECT puuid, champion_name, role, team_id, win, gold_earned, cs FROM participant_history WHERE match_id = ? ORDER BY team_id, role",
        ("NA1_1",),
    ).fetchall()
    conn.close()

    assert row[0] == "NA1"
    assert json.loads(row[1])["metadata"]["matchId"] == "NA1_1"
    assert json.loads(row[2])["format"] == "role_order_v1"
    assert json.loads(row[3])[0]["tier"] == "EMERALD"
    assert row[4] == "15.1.1"
    assert row[5] == 420
    assert row[6] == 1
    assert row[7] == 1
    assert row[8] == 0.75
    assert row[9] > 0.5
    assert row[10] == 3
    assert row[11] == 1
    assert row[12] == 11500.0
    assert row[13] == 1.0
    assert len(participant_rows) == 10
    assert participant_rows[0] == ("p4", "Ashe", "BOTTOM", 100, 1, 13000.0, 0.0)


def test_merge_matches_prefers_richer_incoming_row(tmp_path):
    target = sqlite3.connect(tmp_path / "target.db")
    source = sqlite3.connect(tmp_path / "source.db")
    target.row_factory = sqlite3.Row
    source.row_factory = sqlite3.Row
    ensure_match_schema(target)
    ensure_match_schema(source)

    target.execute(
        "INSERT INTO matches (match_id, match_data, region, data_source, collector_id, last_updated_ts) VALUES (?, ?, ?, ?, ?, ?)",
        ("NA1_1", "[]", "na1", "legacy", "joseph", 1),
    )
    source.execute(
        """
        INSERT INTO matches (
            match_id, match_data, region, raw_match_json, ordered_match_json, rank_snapshot_json,
            data_source, collector_id, game_version, queue_id, last_updated_ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("NA1_1", '{"format":"role_order_v1"}', "na1", '{"metadata":{"matchId":"NA1_1"}}', '{"format":"role_order_v1"}', '[{"status":"ranked"}]',
         "riot_api_match_v5", "storm", "15.1.1", 420, 10),
    )

    merged, inserted, updated = merge_matches(target, source)
    row = target.execute(
        "SELECT raw_match_json, ordered_match_json, rank_snapshot_json, collector_id, data_source FROM matches WHERE match_id = ?",
        ("NA1_1",),
    ).fetchone()
    target.close()
    source.close()

    assert merged == 1
    assert inserted == 0
    assert updated == 1
    assert row["raw_match_json"] is not None
    assert row["ordered_match_json"] is not None
    assert row["rank_snapshot_json"] is not None
    assert row["collector_id"] == "storm"
    assert row["data_source"] == "riot_api_match_v5"


def test_rebuild_participant_history_rehydrates_rows_from_matches(tmp_path):
    db_path = tmp_path / "matches.db"
    conn = sqlite3.connect(db_path)
    ensure_match_schema(conn)

    match = {
        "metadata": {"matchId": "NA1_1"},
        "info": {
            "gameVersion": "15.1.1",
            "queueId": 420,
            "gameCreation": 123456,
            "gameEndTimestamp": 123999,
            "teams": [{"teamId": 100, "win": True}, {"teamId": 200, "win": False}],
            "participants": [
                {"teamId": 100, "championName": "Aatrox", "teamPosition": "TOP", "individualPosition": "TOP", "puuid": "p1"},
                {"teamId": 100, "championName": "Amumu", "teamPosition": "JUNGLE", "individualPosition": "JUNGLE", "puuid": "p2"},
                {"teamId": 100, "championName": "Ahri", "teamPosition": "MIDDLE", "individualPosition": "MIDDLE", "puuid": "p3"},
                {"teamId": 100, "championName": "Ashe", "teamPosition": "BOTTOM", "individualPosition": "BOTTOM", "puuid": "p4"},
                {"teamId": 100, "championName": "Braum", "teamPosition": "UTILITY", "individualPosition": "UTILITY", "puuid": "p5"},
                {"teamId": 200, "championName": "Renekton", "teamPosition": "TOP", "individualPosition": "TOP", "puuid": "p6"},
                {"teamId": 200, "championName": "LeeSin", "teamPosition": "JUNGLE", "individualPosition": "JUNGLE", "puuid": "p7"},
                {"teamId": 200, "championName": "Lux", "teamPosition": "MIDDLE", "individualPosition": "MIDDLE", "puuid": "p8"},
                {"teamId": 200, "championName": "Jinx", "teamPosition": "BOTTOM", "individualPosition": "BOTTOM", "puuid": "p9"},
                {"teamId": 200, "championName": "Nami", "teamPosition": "UTILITY", "individualPosition": "UTILITY", "puuid": "p10"},
            ],
        },
    }
    ordered = {
        "format": "role_order_v1",
        "blue_win": True,
        "blue_side": 1,
        "champions": ["Aatrox", "Amumu", "Ahri", "Ashe", "Braum", "Renekton", "LeeSin", "Lux", "Jinx", "Nami"],
        "role_order": ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"],
        "position_source": "teamPosition",
        "game_version": "15.1.1",
        "queue_id": 420,
    }

    payload = extract_storage_payload(match, "NA1", ordered, collector_id="tester")
    upsert_match_record(conn, "NA1_1", payload)
    conn.execute("DELETE FROM participant_history")

    matches_processed, participants_inserted = rebuild_participant_history(conn)
    rebuilt_rows = conn.execute("SELECT count(*) FROM participant_history").fetchone()[0]
    conn.close()

    assert matches_processed == 1
    assert participants_inserted == 10
    assert rebuilt_rows == 10


def test_rebuild_participant_history_reports_progress(tmp_path):
    db_path = tmp_path / "matches.db"
    conn = sqlite3.connect(db_path)
    ensure_match_schema(conn)

    match = {
        "metadata": {"matchId": "NA1_1"},
        "info": {
            "gameVersion": "15.1.1",
            "queueId": 420,
            "gameCreation": 123456,
            "gameEndTimestamp": 123999,
            "teams": [{"teamId": 100, "win": True}, {"teamId": 200, "win": False}],
            "participants": [
                {"teamId": 100, "championName": "Aatrox", "teamPosition": "TOP", "individualPosition": "TOP", "puuid": "p1"},
                {"teamId": 100, "championName": "Amumu", "teamPosition": "JUNGLE", "individualPosition": "JUNGLE", "puuid": "p2"},
                {"teamId": 100, "championName": "Ahri", "teamPosition": "MIDDLE", "individualPosition": "MIDDLE", "puuid": "p3"},
                {"teamId": 100, "championName": "Ashe", "teamPosition": "BOTTOM", "individualPosition": "BOTTOM", "puuid": "p4"},
                {"teamId": 100, "championName": "Braum", "teamPosition": "UTILITY", "individualPosition": "UTILITY", "puuid": "p5"},
                {"teamId": 200, "championName": "Renekton", "teamPosition": "TOP", "individualPosition": "TOP", "puuid": "p6"},
                {"teamId": 200, "championName": "LeeSin", "teamPosition": "JUNGLE", "individualPosition": "JUNGLE", "puuid": "p7"},
                {"teamId": 200, "championName": "Lux", "teamPosition": "MIDDLE", "individualPosition": "MIDDLE", "puuid": "p8"},
                {"teamId": 200, "championName": "Jinx", "teamPosition": "BOTTOM", "individualPosition": "BOTTOM", "puuid": "p9"},
                {"teamId": 200, "championName": "Nami", "teamPosition": "UTILITY", "individualPosition": "UTILITY", "puuid": "p10"},
            ],
        },
    }
    ordered = {
        "format": "role_order_v1",
        "blue_win": True,
        "blue_side": 1,
        "champions": ["Aatrox", "Amumu", "Ahri", "Ashe", "Braum", "Renekton", "LeeSin", "Lux", "Jinx", "Nami"],
        "role_order": ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"],
        "position_source": "teamPosition",
        "game_version": "15.1.1",
        "queue_id": 420,
    }

    payload = extract_storage_payload(match, "NA1", ordered, collector_id="tester")
    upsert_match_record(conn, "NA1_1", payload)
    conn.execute("DELETE FROM participant_history")

    updates = []

    def on_progress(**kwargs):
        updates.append(kwargs)

    rebuild_participant_history(conn, progress_callback=on_progress, progress_update_every=1)
    conn.close()

    assert len(updates) == 2
    assert updates[0]["processed"] == 0
    assert updates[0]["total"] == 1
    assert updates[0]["matches_processed"] == 0
    assert updates[0]["participants_inserted"] == 0
    assert updates[0]["force"] is False
    assert updates[-1]["processed"] == 1
    assert updates[-1]["total"] == 1
    assert updates[-1]["matches_processed"] == 1
    assert updates[-1]["participants_inserted"] == 10
    assert updates[-1]["force"] is True
