import torch

from ml.predictor.unified_model import UnifiedWinPredictorModel


def test_unified_win_predictor_model_output_shape():
    model = UnifiedWinPredictorModel(
        num_champions=7,
        dense_feature_dim=32,
        num_player_features=3,
        num_global_features=2,
        num_regions=3,
        embedding_dim=16,
        nhead=2,
        dim_feedforward=32,
        num_layers=1,
        dropout=0.0,
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
    dense_features = torch.rand((2, 32), dtype=torch.float32)
    region_ids = torch.tensor([0, 2], dtype=torch.long)

    logits = model(
        champion_ids,
        role_ids,
        team_ids,
        dense_features=dense_features,
        region_ids=region_ids,
    )

    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()


def test_unified_win_predictor_model_can_return_aux_predictions():
    model = UnifiedWinPredictorModel(
        num_champions=7,
        dense_feature_dim=22,
        num_player_features=2,
        num_global_features=2,
        num_regions=2,
        embedding_dim=16,
        nhead=2,
        dim_feedforward=32,
        num_layers=1,
        dropout=0.0,
    )
    champion_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 0, 1, 2]], dtype=torch.long)
    role_ids = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]], dtype=torch.long)
    team_ids = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.long)
    dense_features = torch.rand((1, 22), dtype=torch.float32)
    region_ids = torch.tensor([1], dtype=torch.long)

    logits, aux = model(
        champion_ids,
        role_ids,
        team_ids,
        dense_features=dense_features,
        region_ids=region_ids,
        return_aux=True,
    )

    assert logits.shape == (1, 1)
    assert aux.shape == (1, 4)
    assert torch.isfinite(aux).all()
