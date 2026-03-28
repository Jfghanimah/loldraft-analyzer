import torch

from ml.predictor.models_pytorch import SequenceWinPredictorModel


def test_sequence_win_predictor_model_output_shape():
    model = SequenceWinPredictorModel(
        num_champions=7,
        num_regions=3,
        history_length=2,
        embedding_dim=32,
        nhead=4,
        dim_feedforward=64,
        num_layers=2,
        dropout=0.0,
    )

    batch_size = 2
    logits = model(
        current_champion_ids=torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 0, 1, 2], [6, 5, 4, 3, 2, 1, 0, 6, 5, 4]],
            dtype=torch.long,
        ),
        current_role_ids=torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]] * batch_size, dtype=torch.long),
        current_team_ids=torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]] * batch_size, dtype=torch.long),
        region_ids=torch.tensor([0, 2], dtype=torch.long),
        patch_features=torch.tensor([[0.75, 0.2], [0.75, 0.2]], dtype=torch.float32),
        history_champion_ids=torch.tensor(
            [
                [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 0], [0, 1], [1, 2], [2, 3]],
                [[6, 5], [5, 4], [4, 3], [3, 2], [2, 1], [1, 0], [0, 6], [6, 5], [5, 4], [4, 3]],
            ],
            dtype=torch.long,
        ),
        history_role_ids=torch.tensor([[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]] * batch_size, dtype=torch.long),
        history_slot_ids=torch.tensor([[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]]] * batch_size, dtype=torch.long),
        history_result_ids=torch.tensor([[[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]] * batch_size, dtype=torch.long),
        history_numeric=torch.rand((batch_size, 10, 2, 9), dtype=torch.float32),
        history_mask=torch.tensor([[[True, False]] * 10] * batch_size, dtype=torch.bool),
    )

    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()


def test_sequence_win_predictor_model_can_return_joint_win_and_outcome_logits():
    model = SequenceWinPredictorModel(
        num_champions=7,
        num_regions=3,
        history_length=2,
        embedding_dim=32,
        nhead=4,
        dim_feedforward=64,
        num_layers=2,
        dropout=0.0,
    )

    outputs = model(
        current_champion_ids=torch.tensor([[0, 1, 2, 3, 4, 5, 6, 0, 1, 2]], dtype=torch.long),
        current_role_ids=torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]], dtype=torch.long),
        current_team_ids=torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.long),
        region_ids=torch.tensor([0], dtype=torch.long),
        patch_features=torch.tensor([[0.75, 0.2]], dtype=torch.float32),
        history_champion_ids=torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 0], [0, 1], [1, 2], [2, 3]]], dtype=torch.long),
        history_role_ids=torch.tensor([[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]], dtype=torch.long),
        history_slot_ids=torch.tensor([[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]]], dtype=torch.long),
        history_result_ids=torch.tensor([[[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]], dtype=torch.long),
        history_numeric=torch.rand((1, 10, 2, 9), dtype=torch.float32),
        history_mask=torch.tensor([[[True, False]] * 10], dtype=torch.bool),
        return_aux_outputs=True,
    )

    assert outputs["win_logit"].shape == (1, 1)
    assert outputs["outcome_logits"].shape == (1, 4)


def test_sequence_win_predictor_attention_mask_blocks_cross_player_history():
    model = SequenceWinPredictorModel(
        num_champions=7,
        num_regions=3,
        history_length=2,
        embedding_dim=32,
        nhead=4,
        dim_feedforward=64,
        num_layers=2,
        dropout=0.0,
    )

    mask = model._attention_mask

    # Draft tokens can still see everything.
    assert mask[3, 16].item() is False

    # Blue top history token can see its own draft token and own history block.
    assert mask[13, 3].item() is False
    assert mask[13, 14].item() is False

    # Blue top history token cannot see blue jungle draft/history.
    assert mask[13, 4].item() is True
    assert mask[13, 15].item() is True
