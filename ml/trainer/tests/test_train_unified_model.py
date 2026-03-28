import pandas as pd

from ml.trainer.train_unified_model import run_unified_training


def test_run_unified_training_zero_epochs_builds_model(monkeypatch, tmp_path):
    rows = [
        [1] + list(range(10)) + [1] + [0.1] * 172,
        [0] + list(range(1, 11)) + [1] + [0.2] * 172,
    ]
    dataframe = pd.DataFrame(rows)
    champion_list = {f"Champ_{index}": index for index in range(11)}

    monkeypatch.setattr(
        "ml.trainer.train_unified_model.build_rich_feature_dataframe",
        lambda: (dataframe, champion_list),
    )

    model = run_unified_training(
        epochs=0,
        batch_size=1,
        num_champions=len(champion_list),
        save_path=str(tmp_path / "best_unified_win_predictor.pth"),
    )

    assert model is not None
