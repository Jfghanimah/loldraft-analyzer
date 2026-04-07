import pandas as pd

from ml.data.match_format import ROLE_ORDER
from ml.features.recent_history import dense_feature_columns
from ml.trainer.feature_pipeline import AUX_TARGET_COLUMNS, CHAMPION_COLUMNS
from ml.trainer.train_unified_model import run_unified_training


def test_run_unified_training_zero_epochs_builds_model(monkeypatch, tmp_path):
    columns = ["label", *CHAMPION_COLUMNS, "region_id", *AUX_TARGET_COLUMNS, *dense_feature_columns(ROLE_ORDER)]
    rows = [
        [1] + list(range(10)) + [0] + [1000.0, 2.0, 1.0, 32.0] + [0.1] * 142,
        [0] + list(range(1, 11)) + [0] + [-800.0, 1.0, 3.0, 28.0] + [0.2] * 142,
    ]
    dataframe = pd.DataFrame(rows, columns=columns)
    champion_list = {f"Champ_{index}": index for index in range(11)}

    monkeypatch.setattr(
        "ml.trainer.train_unified_model.build_rich_feature_dataframe",
        lambda **kwargs: (dataframe, champion_list),
    )

    model = run_unified_training(
        epochs=0,
        batch_size=1,
        num_champions=len(champion_list),
        save_path=str(tmp_path / "best_unified_win_predictor.pth"),
        feature_cache_path=str(tmp_path / "feature_cache.pkl"),
    )

    assert model is not None


def test_run_unified_training_reuses_cached_feature_dataframe(monkeypatch, tmp_path):
    columns = ["label", *CHAMPION_COLUMNS, "region_id", *AUX_TARGET_COLUMNS, *dense_feature_columns(ROLE_ORDER)]
    rows = [
        [1] + list(range(10)) + [0] + [1000.0, 2.0, 1.0, 32.0] + [0.1] * 142,
        [0] + list(range(1, 11)) + [0] + [-800.0, 1.0, 3.0, 28.0] + [0.2] * 142,
    ]
    dataframe = pd.DataFrame(rows, columns=columns)
    champion_list = {f"Champ_{index}": index for index in range(11)}
    cache_path = tmp_path / "feature_cache.pkl"
    db_path = tmp_path / "training.db"
    db_path.write_text("cache signature source", encoding="utf-8")

    call_count = {"count": 0}

    def fake_builder(**kwargs):
        call_count["count"] += 1
        return dataframe, champion_list

    monkeypatch.setattr(
        "ml.trainer.train_unified_model.build_rich_feature_dataframe",
        fake_builder,
    )

    run_unified_training(
        epochs=0,
        batch_size=1,
        num_champions=len(champion_list),
        save_path=str(tmp_path / "best_unified_win_predictor_1.pth"),
        db_path=str(db_path),
        feature_cache_path=str(cache_path),
    )
    run_unified_training(
        epochs=0,
        batch_size=1,
        num_champions=len(champion_list),
        save_path=str(tmp_path / "best_unified_win_predictor_2.pth"),
        db_path=str(db_path),
        feature_cache_path=str(cache_path),
    )

    assert call_count["count"] == 1
