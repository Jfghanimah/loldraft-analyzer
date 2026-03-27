import json
import sqlite3

import pytest

from ml.data.match_storage import ensure_match_schema
from ml.data.pytorch_data import get_data_frames


def _ordered_record(blue_win, champions, position_source="teamPosition"):
    return {
        "format": "role_order_v1",
        "blue_win": blue_win,
        "blue_side": 1,
        "champions": champions,
        "role_order": ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"],
        "position_source": position_source,
        "game_version": "15.1.1",
        "queue_id": 420,
    }


def test_get_data_frames_syncs_missing_champions_and_avoids_negative_ids(tmp_path):
    champion_path = tmp_path / "champion_list.json"
    db_path = tmp_path / "matches.db"

    champion_path.write_text(
        json.dumps(
            {
                "Aatrox": 0,
                "Ahri": 1,
                "Akali": 2,
                "Alistar": 3,
                "Amumu": 4,
                "Anivia": 5,
                "Annie": 6,
                "Ashe": 7,
                "Azir": 8,
                "Bard": 9,
                "Blitzcrank": 10,
                "Brand": 11,
                "Braum": 12,
                "Caitlyn": 13,
            }
        ),
        encoding="utf-8",
    )

    conn = sqlite3.connect(db_path)
    ensure_match_schema(conn)
    conn.execute(
        "INSERT INTO matches (match_id, match_data, region, ordered_match_json) VALUES (?, ?, ?, ?)",
        (
            "m1",
            json.dumps(_ordered_record(
                1,
                ["Aatrox", "Ahri", "Akali", "Alistar", "Amumu", "Anivia", "Annie", "Ashe", "Azir", "Bard"],
            )),
            "na1",
            json.dumps(_ordered_record(
                1,
                ["Aatrox", "Ahri", "Akali", "Alistar", "Amumu", "Anivia", "Annie", "Ashe", "Azir", "Bard"],
            )),
        ),
    )
    conn.execute(
        "INSERT INTO matches (match_id, match_data, region, ordered_match_json) VALUES (?, ?, ?, ?)",
        (
            "m2",
            json.dumps(_ordered_record(
                0,
                ["Briar", "Blitzcrank", "Brand", "Braum", "Caitlyn", "Aatrox", "Ahri", "Akali", "Alistar", "Amumu"],
            )),
            "na1",
            json.dumps(_ordered_record(
                0,
                ["Briar", "Blitzcrank", "Brand", "Braum", "Caitlyn", "Aatrox", "Ahri", "Akali", "Alistar", "Amumu"],
            )),
        ),
    )
    conn.commit()
    conn.close()

    df, champion_list = get_data_frames(
        db_path=str(db_path),
        champion_path=str(champion_path),
    )

    assert "Briar" in champion_list
    assert champion_list["Briar"] == 14
    assert len(df) == 2
    assert df.iloc[:, 1:11].min().min() >= 0
    assert set(df.iloc[:, 11].tolist()) == {1}


def test_get_data_frames_requires_current_training_db(tmp_path):
    with pytest.raises(ValueError, match="Training database not found"):
        get_data_frames(
            db_path=str(tmp_path / "missing.db"),
            champion_path=str(tmp_path / "champion_list.json"),
        )


def test_get_data_frames_requires_ordered_match_rows(tmp_path):
    db_path = tmp_path / "matches.db"
    conn = sqlite3.connect(db_path)
    ensure_match_schema(conn)
    conn.execute(
        "INSERT INTO matches (match_id, match_data, region) VALUES (?, ?, ?)",
        ("m1", "{}", "na1"),
    )
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="No role-ordered matches are available"):
        get_data_frames(
            db_path=str(db_path),
            champion_path=str(tmp_path / "champion_list.json"),
        )
