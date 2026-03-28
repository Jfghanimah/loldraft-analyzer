import torch

from ml.predictor.models_pytorch import MLMPretrainModel, WinPredictorModel
from ml.predictor.unified_model import UnifiedWinPredictorModel


def test_mlm_pretrain_model_output_shape():
    model = MLMPretrainModel(
        num_champions=7,
        embedding_dim=8,
        nhead=2,
        dim_feedforward=16,
        num_layers=1,
    )
    champion_ids = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 0, 1, 7],
            [6, 5, 4, 3, 2, 1, 0, 6, 5, 7],
        ],
        dtype=torch.long,
    )
    role_ids = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]] * 2, dtype=torch.long)
    team_ids = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]] * 2, dtype=torch.long)

    logits = model(champion_ids, role_ids, team_ids)

    assert logits.shape == (2, 10, 9)


def test_win_predictor_model_output_shape():
    model = WinPredictorModel(
        num_champions=7,
        embedding_dim=8,
        nhead=2,
        dim_feedforward=16,
        dropout=0.0,
        num_layers=1,
    )
    champion_ids = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 0, 1, 2],
            [6, 5, 4, 3, 2, 1, 0, 6, 5, 4],
        ],
        dtype=torch.long,
    )
    role_ids = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]] * 2, dtype=torch.long)
    team_ids = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]] * 2, dtype=torch.long)
    blue_side = torch.ones((2, 1), dtype=torch.float32)

    logits = model(champion_ids, role_ids, team_ids, blue_side)

    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()


def test_win_predictor_model_defaults_blue_side_to_ones():
    model = WinPredictorModel(
        num_champions=7,
        embedding_dim=8,
        nhead=2,
        dim_feedforward=16,
        dropout=0.0,
        num_layers=1,
    )
    champion_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 0, 1, 2]], dtype=torch.long)
    role_ids = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]], dtype=torch.long)
    team_ids = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.long)

    logits = model(champion_ids, role_ids, team_ids)

    assert logits.shape == (1, 1)


def test_win_predictor_model_accepts_extra_dense_features():
    model = WinPredictorModel(
        num_champions=7,
        embedding_dim=8,
        nhead=2,
        dim_feedforward=16,
        dropout=0.0,
        num_layers=1,
        extra_feature_dim=4,
    )
    champion_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 0, 1, 2]], dtype=torch.long)
    role_ids = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]], dtype=torch.long)
    team_ids = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.long)
    blue_side = torch.ones((1, 1), dtype=torch.float32)
    dense_features = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)

    logits = model(champion_ids, role_ids, team_ids, blue_side, dense_features)

    assert logits.shape == (1, 1)


def test_unified_win_predictor_model_output_shape():
    model = UnifiedWinPredictorModel(
        num_champions=7,
        dense_feature_dim=6,
        embedding_dim=16,
        nhead=2,
        dim_feedforward=32,
        num_layers=1,
        dropout=0.0,
        dense_hidden_dim=8,
    )
    champion_ids = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 0, 1, 2],
            [6, 5, 4, 3, 2, 1, 0, 6, 5, 4],
        ],
        dtype=torch.long,
    )
    role_ids = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]] * 2, dtype=torch.long)
    team_ids = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]] * 2, dtype=torch.long)
    blue_side = torch.ones((2, 1), dtype=torch.float32)
    dense_features = torch.rand((2, 6), dtype=torch.float32)

    logits = model(champion_ids, role_ids, team_ids, dense_features, blue_side)

    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()
