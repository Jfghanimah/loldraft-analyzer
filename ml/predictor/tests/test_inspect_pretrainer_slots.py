import json
import sqlite3

import torch

from ml.data.match_storage import ensure_match_schema
from ml.predictor.inspect_pretrainer_slots import (
    _aggregate_role_substitutes,
    _predict_for_masked_draft,
)
from ml.predictor.models_pytorch import MLMPretrainModel


def _seed_model_with_known_preferences():
    model = MLMPretrainModel(
        num_champions=7,
        embedding_dim=8,
        nhead=2,
        dim_feedforward=16,
        num_layers=1,
    )
    model.eval()
    with torch.no_grad():
        model.head.bias.zero_()
        model.head.bias[5] = 8.0
        model.head.bias[6] = 7.0
        model.head.bias[7] = 6.0  # MASK token should be filtered out
    return model


def test_predict_for_masked_draft_filters_special_tokens_and_existing_champs():
    model = _seed_model_with_known_preferences()
    id_to_name = {0: "Aatrox", 1: "Amumu", 2: "Ahri", 3: "Ashe", 4: "Braum", 5: "Soraka", 6: "Sona"}
    champion_ids = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

    results = _predict_for_masked_draft(
        model,
        num_champions=7,
        champion_ids=champion_ids,
        mask_index=4,
        top_k=2,
        id_to_name=id_to_name,
    )

    assert results[0][0] == "Soraka"
    assert results[1][0] == "Sona"


def test_aggregate_role_substitutes_averages_masked_slot_predictions(tmp_path):
    db_path = tmp_path / "matches.db"
    conn = sqlite3.connect(db_path)
    ensure_match_schema(conn)

    ordered_record = {
        "format": "role_order_v1",
        "blue_win": True,
        "blue_side": 1,
        "champions": ["Aatrox", "Amumu", "Ahri", "Ashe", "Sona", "Renekton", "LeeSin", "Lux", "Jinx", "Nami"],
        "role_order": ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"],
        "position_source": "teamPosition",
        "game_version": "15.1.1",
        "queue_id": 420,
    }
    conn.execute(
        "INSERT INTO matches (match_id, match_data, region, ordered_match_json, queue_id) VALUES (?, ?, ?, ?, ?)",
        ("m1", json.dumps(ordered_record), "na1", json.dumps(ordered_record), 420),
    )
    conn.commit()
    conn.close()

    model = _seed_model_with_known_preferences()
    name_to_id = {"Aatrox": 0, "Amumu": 1, "Ahri": 2, "Ashe": 3, "Braum": 4, "Soraka": 5, "Sona": 6, "Renekton": 0, "LeeSin": 1, "Lux": 2, "Jinx": 3, "Nami": 4}
    id_to_name = {0: "Aatrox", 1: "Amumu", 2: "Ahri", 3: "Ashe", 4: "Braum", 5: "Soraka", 6: "Sona"}

    conn = sqlite3.connect(db_path)
    ordered_records = [json.loads(row[0]) for row in conn.execute("SELECT ordered_match_json FROM matches").fetchall()]
    contexts, results = _aggregate_role_substitutes(
        model,
        num_champions=7,
        ordered_records=ordered_records,
        name_to_id=name_to_id,
        id_to_name=id_to_name,
        champion="Sona",
        role="support",
        top_k=2,
        team="blue",
    )
    conn.close()

    assert contexts == 1
    assert results[0][0] == "Soraka"
