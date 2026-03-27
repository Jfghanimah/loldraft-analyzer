import sqlite3

import pytest
from fastapi import HTTPException

import website.server as server
from ml.predictor.models_pytorch import WinPredictorModel


def _setup_model(monkeypatch, extra_feature_dim=152):
    champion_list = {
        "Aatrox": 0,
        "Amumu": 1,
        "Ahri": 2,
        "Ashe": 3,
        "Braum": 4,
        "Renekton": 5,
        "LeeSin": 6,
        "Lux": 7,
        "Jinx": 8,
        "Nami": 9,
    }
    model = WinPredictorModel(
        num_champions=len(champion_list),
        embedding_dim=8,
        nhead=2,
        dim_feedforward=16,
        dropout=0.0,
        num_layers=1,
        extra_feature_dim=extra_feature_dim,
    )
    model.eval()
    monkeypatch.setattr(server, "_model_loaded", True)
    monkeypatch.setattr(server, "_model", model)
    monkeypatch.setattr(server, "_champion_list", champion_list)
    monkeypatch.setattr(server, "_load_resources", lambda: None)
    monkeypatch.setattr(server, "build_dense_features_for_prediction", lambda conn, champions, players=None: [0.1] * extra_feature_dim)
    monkeypatch.setattr(server, "connect_sqlite", lambda *args, **kwargs: sqlite3.connect(":memory:"))


def test_predict_accepts_optional_players(monkeypatch):
    _setup_model(monkeypatch)

    response = server.predict(
        server.PredictRequest(
            champions=["Aatrox", "Amumu", "Ahri", "Ashe", "Braum", "Renekton", "LeeSin", "Lux", "Jinx", "Nami"],
            players=["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"],
            blue_side=1.0,
        )
    )

    assert "blue_win_probability" in response
    assert "lane_scores" in response


def test_predict_rejects_wrong_player_count(monkeypatch):
    _setup_model(monkeypatch)

    with pytest.raises(HTTPException, match="aligned puuids"):
        server.predict(
            server.PredictRequest(
                champions=["Aatrox", "Amumu", "Ahri", "Ashe", "Braum", "Renekton", "LeeSin", "Lux", "Jinx", "Nami"],
                players=["p1"],
                blue_side=1.0,
            )
        )
