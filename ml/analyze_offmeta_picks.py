import json
import math
import os

import pandas as pd
import torch

from ml.data.pytorch_data import CHAMPION_COLUMNS, LeagueDataset
from ml.evaluate_unified_aux import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_FEATURE_CACHE_PATH,
    DEFAULT_LOG_PATH,
    SPLIT_SEED,
    TRAIN_FRAC,
    _load_cached_dataframe,
    _parse_model_config,
)
from ml.features.recent_history import GLOBAL_FEATURES, PARTICIPANT_FEATURES
from ml.predictor.unified_model import UnifiedWinPredictorModel

ANALYSIS_SPECS = [
    ("Azir", "mid"),
    ("Shyvana", "top"),
    ("Lee Sin", "top"),
    ("Twitch", "jungle"),
    ("Corki", "mid"),
    ("Twitch", "support"),
]
ROLE_TO_SLOTS = {
    "top": (0, 5),
    "jungle": (1, 6),
    "mid": (2, 7),
    "bot": (3, 8),
    "support": (4, 9),
}
BATCH_SIZE = 4096
SAMPLE_COUNT = 2


def _load_champion_maps():
    with open("ml/save_data/champion_list.json", "r", encoding="utf-8") as handle:
        champion_list = json.load(handle)
    id_to_name = {idx: name for name, idx in champion_list.items()}
    return champion_list, id_to_name


def _resolve_champion_id(champion_list, champion_name):
    champion_id = champion_list.get(champion_name)
    if champion_id is not None:
        return champion_id

    normalized_target = champion_name.replace(" ", "").replace("'", "").replace(".", "").lower()
    for stored_name, stored_id in champion_list.items():
        normalized_stored = stored_name.replace(" ", "").replace("'", "").replace(".", "").lower()
        if normalized_stored == normalized_target:
            return stored_id
    return None


def _build_val_indices(n):
    train_size = int(TRAIN_FRAC * n)
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(n, generator=generator)
    return perm[train_size:]


def _load_model(dataset):
    config = _parse_model_config(DEFAULT_LOG_PATH)
    dense_feature_dim = dataset.dense_features.shape[1] if dataset.dense_features is not None else 0
    num_regions = int(dataset.region_ids.max().item()) + 1 if dataset.region_ids is not None and len(dataset.region_ids) else 1
    num_champions = int(dataset.matches.max().item()) + 1

    model = UnifiedWinPredictorModel(
        num_champions=num_champions,
        dense_feature_dim=dense_feature_dim,
        num_player_features=len(PARTICIPANT_FEATURES),
        num_global_features=len(GLOBAL_FEATURES),
        num_regions=num_regions,
        embedding_dim=config["embedding_dim"],
        nhead=config["nhead"],
        dim_feedforward=config["dim_feedforward"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    state_dict = torch.load(DEFAULT_CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def _predict_val_win_probs(model, dataset, val_idx):
    role_ids = dataset.role_ids.unsqueeze(0)
    team_ids = dataset.team_ids.unsqueeze(0)
    probs = torch.empty(len(val_idx), dtype=torch.float32)

    with torch.no_grad():
        for start in range(0, len(val_idx), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(val_idx))
            batch_idx = val_idx[start:stop]
            logits = model(
                dataset.matches[batch_idx],
                role_ids.expand(len(batch_idx), -1),
                team_ids.expand(len(batch_idx), -1),
                dense_features=dataset.dense_features[batch_idx] if dataset.dense_features is not None else None,
                region_ids=dataset.region_ids[batch_idx] if dataset.region_ids is not None else None,
            )
            probs[start:stop] = torch.sigmoid(logits.squeeze(1)).cpu()

    return probs


def _decode_team(df_row, slots, id_to_name):
    return [id_to_name.get(int(df_row[f"champion_{slot}"]), f"id:{int(df_row[f'champion_{slot}'])}") for slot in slots]


def _format_float(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{value:.4f}"


def main():
    if not os.path.exists(DEFAULT_FEATURE_CACHE_PATH):
        raise FileNotFoundError(f"Feature cache not found at {DEFAULT_FEATURE_CACHE_PATH}")
    if not os.path.exists(DEFAULT_CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {DEFAULT_CHECKPOINT_PATH}")

    champion_list, id_to_name = _load_champion_maps()
    dataframe = _load_cached_dataframe(DEFAULT_FEATURE_CACHE_PATH)
    dataset = LeagueDataset(dataframe, mode="finetune")
    val_idx = _build_val_indices(len(dataset))
    model, config = _load_model(dataset)
    val_probs = _predict_val_win_probs(model, dataset=dataset, val_idx=val_idx)

    val_index_map = {int(row_idx): pos for pos, row_idx in enumerate(val_idx.tolist())}

    print(f"checkpoint={DEFAULT_CHECKPOINT_PATH}")
    print(f"log_config={config}")
    print(f"dataset_matches={len(dataframe):,}")
    print(f"val_matches={len(val_idx):,}")
    print()

    for champion_name, role_name in ANALYSIS_SPECS:
        champion_id = _resolve_champion_id(champion_list, champion_name)
        if champion_id is None:
            print(f"[{champion_name} {role_name}]")
            print("  missing from champion_list.json")
            print()
            continue

        blue_slot, red_slot = ROLE_TO_SLOTS[role_name]
        blue_mask = dataframe[f"champion_{blue_slot}"] == champion_id
        red_mask = dataframe[f"champion_{red_slot}"] == champion_id

        blue_count = int(blue_mask.sum())
        red_count = int(red_mask.sum())
        total_count = blue_count + red_count

        blue_wins = float(dataframe.loc[blue_mask, "label"].sum())
        red_wins = float((1.0 - dataframe.loc[red_mask, "label"]).sum())
        total_win_rate = (blue_wins + red_wins) / total_count if total_count else float("nan")

        val_occurrences = []
        for row_idx, is_blue in [(idx, True) for idx in dataframe.index[blue_mask]] + [(idx, False) for idx in dataframe.index[red_mask]]:
            val_pos = val_index_map.get(int(row_idx))
            if val_pos is None:
                continue
            blue_win_prob = float(val_probs[val_pos].item())
            side_win_prob = blue_win_prob if is_blue else (1.0 - blue_win_prob)
            actual_side_win = float(dataframe.iloc[row_idx]["label"]) if is_blue else float(1.0 - dataframe.iloc[row_idx]["label"])
            val_occurrences.append((row_idx, is_blue, side_win_prob, actual_side_win))

        val_count = len(val_occurrences)
        val_actual_win_rate = (
            sum(item[3] for item in val_occurrences) / val_count if val_count else float("nan")
        )
        val_pred_win_rate = (
            sum(item[2] for item in val_occurrences) / val_count if val_count else float("nan")
        )

        print(f"[{champion_name} {role_name}]")
        print(f"  dataset_occurrences={total_count:,} (blue={blue_count:,}, red={red_count:,})")
        print(f"  dataset_side_win_rate={_format_float(total_win_rate)}")
        print(f"  val_occurrences={val_count:,}")
        print(f"  val_actual_side_win_rate={_format_float(val_actual_win_rate)}")
        print(f"  val_mean_predicted_side_win={_format_float(val_pred_win_rate)}")

        samples = sorted(val_occurrences, key=lambda item: item[2])[:SAMPLE_COUNT]
        for sample_idx, (row_idx, is_blue, side_win_prob, actual_side_win) in enumerate(samples, start=1):
            row = dataframe.iloc[row_idx]
            side_slots = range(0, 5) if is_blue else range(5, 10)
            opp_slots = range(5, 10) if is_blue else range(0, 5)
            side_team = _decode_team(row, side_slots, id_to_name)
            opp_team = _decode_team(row, opp_slots, id_to_name)
            side_label = "blue" if is_blue else "red"
            outcome = "win" if actual_side_win >= 0.5 else "loss"
            print(
                f"  sample_{sample_idx}: side={side_label} pred_side_win={side_win_prob:.4f} "
                f"actual={outcome} team={side_team} opp={opp_team}"
            )
        print()


if __name__ == "__main__":
    main()
