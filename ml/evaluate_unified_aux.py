import argparse
import math
import os
import re

import pandas as pd
import torch

from ml.data.prepare_training_examples import load_training_examples_dataframe
from ml.data.pytorch_data import LeagueDataset
from ml.predictor.unified_model import UnifiedWinPredictorModel
from ml.trainer.train_unified_model import (
    DEFAULT_FEATURE_CACHE_PATH,
    _drop_population_prior_columns,
    _infer_dense_feature_layout,
)

DEFAULT_CHECKPOINT_PATH = "ml/save_data/best_unified_win_predictor.pth"
DEFAULT_LOG_PATH = "ml/save_data/latest_train_log.txt"
TARGET_NAMES = [
    "gold_diff",
    "blue_dragons",
    "red_dragons",
    "game_length_minutes",
]
TRAIN_FRAC = 0.9
SPLIT_SEED = 42
BATCH_SIZE = 4096


def _load_cached_dataframe(cache_path):
    bundle = pd.read_pickle(cache_path)
    if not isinstance(bundle, dict) or "dataframe" not in bundle:
        raise ValueError(f"Unexpected cache bundle format at {cache_path}")
    return bundle["dataframe"]


def _load_evaluation_dataframe(cache_path, training_data_dir=None, queue_id=420, drop_population_priors=False):
    if training_data_dir:
        dataframe, _ = load_training_examples_dataframe(training_data_dir, queue_id=queue_id)
    else:
        dataframe = _load_cached_dataframe(cache_path)
    if drop_population_priors:
        dataframe, _ = _drop_population_prior_columns(dataframe)
    return dataframe


def _parse_model_config(log_path):
    if not os.path.exists(log_path):
        raise FileNotFoundError(
            f"Could not find training log at {log_path}. "
            "Run the trainer once so the evaluator can infer the saved checkpoint shape."
        )

    pattern = re.compile(
        r"Model: (?:architecture=(?P<architecture>\w+), )?dim=(?P<dim>\d+), heads=(?P<heads>\d+), layers=(?P<layers>\d+), "
        r"ff=(?P<ff>\d+), dropout=(?P<dropout>\d+(?:\.\d+)?)"
    )
    with open(log_path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    drop_population_priors = any("Drop population priors: True" in line for line in lines)
    for line in reversed(lines):
        match = pattern.search(line)
        if match:
            return {
                "architecture": match.group("architecture") or "flat",
                "embedding_dim": int(match.group("dim")),
                "nhead": int(match.group("heads")),
                "num_layers": int(match.group("layers")),
                "dim_feedforward": int(match.group("ff")),
                "dropout": float(match.group("dropout")),
                "drop_population_priors": drop_population_priors,
            }

    raise ValueError(f"Could not infer model config from {log_path}")


def _build_val_tensors(dataset):
    n = len(dataset)
    train_size = int(TRAIN_FRAC * n)
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(n, generator=generator)
    val_idx = perm[train_size:]

    champion_ids = dataset.matches[val_idx]
    labels = dataset.labels[val_idx].unsqueeze(1)
    dense_features = dataset.dense_features[val_idx] if dataset.dense_features is not None else None
    region_ids = dataset.region_ids[val_idx] if dataset.region_ids is not None else None
    aux_targets = dataset.aux_targets[val_idx] if dataset.aux_targets is not None else None
    role_ids = dataset.role_ids.unsqueeze(0).expand(len(val_idx), -1)
    team_ids = dataset.team_ids.unsqueeze(0).expand(len(val_idx), -1)

    return {
        "champion_ids": champion_ids,
        "labels": labels,
        "dense_features": dense_features,
        "region_ids": region_ids,
        "aux_targets": aux_targets,
        "role_ids": role_ids,
        "team_ids": team_ids,
    }


def _corrcoef(x, y):
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.std(unbiased=False) * y.std(unbiased=False)
    if denom.item() == 0:
        return float("nan")
    return float((x * y).mean() / denom)


def _target_summary(name, predictions, targets):
    predictions = predictions.float().cpu()
    targets = targets.float().cpu()
    baseline = torch.full_like(targets, float(targets.mean()))

    mae_model = float((predictions - targets).abs().mean())
    mae_baseline = float((baseline - targets).abs().mean())
    rmse_model = float(torch.sqrt(((predictions - targets) ** 2).mean()))
    rmse_baseline = float(torch.sqrt(((baseline - targets) ** 2).mean()))

    result = {
        "name": name,
        "target_mean": float(targets.mean()),
        "target_std": float(targets.std(unbiased=False)),
        "pred_mean": float(predictions.mean()),
        "pred_std": float(predictions.std(unbiased=False)),
        "mae_model": mae_model,
        "mae_mean_baseline": mae_baseline,
        "rmse_model": rmse_model,
        "rmse_mean_baseline": rmse_baseline,
        "corr": _corrcoef(predictions, targets),
    }

    if "dragons" in name:
        rounded = predictions.round().clamp(min=0)
        baseline_rounded = baseline.round().clamp(min=0)
        result["rounded_exact_model"] = float((rounded == targets).float().mean())
        result["rounded_exact_baseline"] = float((baseline_rounded == targets).float().mean())

    if name == "gold_diff":
        threshold = torch.quantile(targets.abs(), 0.9)
        mask = targets.abs() >= threshold
        result["outlier_mae_model"] = float((predictions[mask] - targets[mask]).abs().mean())
        result["outlier_mae_baseline"] = float((baseline[mask] - targets[mask]).abs().mean())
    elif name == "game_length_minutes":
        threshold = torch.quantile(targets, 0.9)
        mask = targets >= threshold
        result["long_game_mae_model"] = float((predictions[mask] - targets[mask]).abs().mean())
        result["long_game_mae_baseline"] = float((baseline[mask] - targets[mask]).abs().mean())

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate auxiliary targets for a unified checkpoint")
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH)
    parser.add_argument("--feature-cache-path", default=DEFAULT_FEATURE_CACHE_PATH)
    parser.add_argument("--training-data-dir", default=None)
    parser.add_argument("--queue-id", type=int, default=420)
    args = parser.parse_args()

    cache_path = args.feature_cache_path
    checkpoint_path = args.checkpoint_path
    log_path = args.log_path

    if not args.training_data_dir and not os.path.exists(cache_path):
        raise FileNotFoundError(f"Feature cache not found at {cache_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    config = _parse_model_config(log_path)
    dataframe = _load_evaluation_dataframe(
        cache_path,
        args.training_data_dir,
        queue_id=args.queue_id,
        drop_population_priors=config["drop_population_priors"],
    )
    dataset = LeagueDataset(dataframe, mode="finetune")
    val = _build_val_tensors(dataset)

    dense_feature_dim = dataset.dense_features.shape[1] if dataset.dense_features is not None else 0
    num_player_features, num_global_features = _infer_dense_feature_layout(dense_feature_dim)
    num_regions = int(dataset.region_ids.max().item()) + 1 if dataset.region_ids is not None and len(dataset.region_ids) else 1
    num_champions = int(dataset.matches.max().item()) + 1

    model = UnifiedWinPredictorModel(
        num_champions=num_champions,
        dense_feature_dim=dense_feature_dim,
        num_player_features=num_player_features,
        num_global_features=num_global_features,
        num_regions=num_regions,
        embedding_dim=config["embedding_dim"],
        nhead=config["nhead"],
        dim_feedforward=config["dim_feedforward"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        architecture=config["architecture"],
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    aux_predictions = []
    win_logits = []
    with torch.no_grad():
        for start in range(0, len(val["champion_ids"]), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(val["champion_ids"]))
            logits, aux = model(
                val["champion_ids"][start:stop],
                val["role_ids"][start:stop],
                val["team_ids"][start:stop],
                dense_features=val["dense_features"][start:stop] if val["dense_features"] is not None else None,
                region_ids=val["region_ids"][start:stop] if val["region_ids"] is not None else None,
                return_aux=True,
            )
            win_logits.append(logits.cpu())
            aux_predictions.append(aux.cpu())

    aux_predictions = torch.cat(aux_predictions, dim=0)
    win_logits = torch.cat(win_logits, dim=0)
    aux_targets = val["aux_targets"].cpu()
    labels = val["labels"].cpu()

    preds = (torch.sigmoid(win_logits) > 0.5).float()
    win_acc = float((preds == labels).float().mean())

    print(f"checkpoint={checkpoint_path}")
    print(f"log_config={config}")
    print(f"val_examples={len(labels):,}")
    print(f"win_val_acc={win_acc:.4f}")
    print()

    for idx, name in enumerate(TARGET_NAMES):
        summary = _target_summary(name, aux_predictions[:, idx], aux_targets[:, idx])
        print(f"[{name}]")
        for key, value in summary.items():
            if key == "name":
                continue
            if isinstance(value, float) and not math.isnan(value):
                print(f"  {key}={value:.4f}")
            else:
                print(f"  {key}={value}")
        print()


if __name__ == "__main__":
    main()
