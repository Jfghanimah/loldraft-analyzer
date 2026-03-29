import json
import os
import sqlite3

import pandas as pd
import torch
from torch.utils.data import Dataset

from ml.data.match_format import get_record_champions, get_record_training_row
from ml.data.match_storage import get_match_columns
from ml.runtime_config import get_db_path, load_runtime_env

CHAMPION_LIST_PATH = 'ml/save_data/champion_list.json'
load_runtime_env()
DEFAULT_DB_PATH = get_db_path()
CHAMPION_COLUMNS = [f"champion_{slot}" for slot in range(10)]
AUX_TARGET_COLUMNS = [
    "target_gold_diff",
    "target_blue_dragons",
    "target_red_dragons",
    "target_game_length_minutes",
]


def _load_raw_matches(db_path):
    if not os.path.exists(db_path):
        raise ValueError(
            f"Training database not found at '{db_path}'. "
            "Set LOL_DRAFT_DB_PATH to a SQLite DB produced by the current scraper."
        )

    conn = sqlite3.connect(db_path)
    columns = get_match_columns(conn)
    if "ordered_match_json" not in columns:
        conn.close()
        raise ValueError("Database does not contain ordered_match_json. Training requires the current role-ordered format.")

    rows = conn.execute(
        """
        SELECT ordered_match_json
        FROM matches
        WHERE ordered_match_json IS NOT NULL
        """
    ).fetchall()
    conn.close()
    return [json.loads(row[0]) for row in rows]


def _load_champion_list(champion_path):
    if not os.path.exists(champion_path):
        return {}

    with open(champion_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _sync_champion_list(raw_matches, champion_path):
    champion_list = _load_champion_list(champion_path)
    next_id = max(champion_list.values(), default=-1) + 1

    observed_champions = sorted({champ for match in raw_matches for champ in get_record_champions(match)})
    missing = [champ for champ in observed_champions if champ not in champion_list]

    if missing:
        for champ in missing:
            champion_list[champ] = next_id
            next_id += 1

        os.makedirs(os.path.dirname(champion_path), exist_ok=True)
        ordered = dict(sorted(champion_list.items(), key=lambda item: item[1]))
        with open(champion_path, 'w', encoding='utf-8') as f:
            json.dump(ordered, f, indent=4)

        print(
            f"Updated champion list with {len(missing)} new champions: "
            f"{', '.join(missing)}"
        )

    return champion_list


def get_data_frames(
    db_path=DEFAULT_DB_PATH,
    champion_path=CHAMPION_LIST_PATH,
):
    raw_matches = _load_raw_matches(db_path)
    champion_list = _sync_champion_list(raw_matches, champion_path)

    converted = []
    for match in raw_matches:
        row = get_record_training_row(match)
        if row is None:
            continue
        converted.append([row[0]] + [champion_list.get(champ, -1) for champ in row[1:11]] + [row[11]])

    if not converted:
        raise ValueError(
            "No role-ordered matches are available for training in ordered_match_json. "
            "Run the current scraper and train from the resulting SQLite database."
        )

    df = pd.DataFrame(converted)

    invalid = int((df.iloc[:, 1:11] < 0).sum().sum())
    if invalid:
        raise ValueError(
            f"Found {invalid} invalid champion IDs after syncing champion list. "
            "Check the stored match data for unexpected champion names."
        )

    return df.sample(frac=1, random_state=42).reset_index(drop=True), champion_list


class LeagueDataset(Dataset):
    def __init__(self, dataframe, mode='finetune', mask_token_id=None):
        if mode not in ('pretrain', 'finetune'):
            raise ValueError("Mode must be 'pretrain' or 'finetune'")

        if all(column in dataframe.columns for column in CHAMPION_COLUMNS):
            champion_values = dataframe.loc[:, CHAMPION_COLUMNS].values
            label_values = dataframe.loc[:, "label"].values if "label" in dataframe.columns else dataframe.iloc[:, 0].values
            blue_side_values = dataframe.loc[:, "blue_side"].values if "blue_side" in dataframe.columns else dataframe.iloc[:, 11].values
            region_values = dataframe.loc[:, "region_id"].values if "region_id" in dataframe.columns else None
            aux_values = dataframe.loc[:, AUX_TARGET_COLUMNS].values if all(
                column in dataframe.columns for column in AUX_TARGET_COLUMNS
            ) else None
            excluded = {"label", *CHAMPION_COLUMNS, "blue_side", "region_id", *AUX_TARGET_COLUMNS}
            dense_columns = [column for column in dataframe.columns if column not in excluded]
            dense_values = dataframe.loc[:, dense_columns].values if dense_columns else None
        else:
            champion_values = dataframe.iloc[:, 1:11].values
            label_values = dataframe.iloc[:, 0].values
            blue_side_values = dataframe.iloc[:, 11].values
            region_values = None
            aux_values = None
            dense_values = dataframe.iloc[:, 12:].values if dataframe.shape[1] > 12 else None

        self.matches = torch.tensor(champion_values, dtype=torch.long)
        self.labels = torch.tensor(label_values, dtype=torch.float32)
        self.mode = mode
        self.mask_token_id = (
            mask_token_id if mask_token_id is not None else int(self.matches.max().item()) + 1
        )
        self.blue_side = torch.tensor(blue_side_values, dtype=torch.float32)
        self.region_ids = torch.tensor(region_values, dtype=torch.long) if region_values is not None else None
        self.aux_targets = torch.tensor(aux_values, dtype=torch.float32) if aux_values is not None else None
        self.has_dense_features = dense_values is not None and dense_values.shape[1] > 0
        if self.has_dense_features:
            self.dense_features = torch.tensor(dense_values, dtype=torch.float32)
        else:
            self.dense_features = None

        self.role_ids = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=torch.long)
        self.team_ids = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        if self.mode == 'pretrain':
            # Return raw champion IDs; masking is applied in the training loop on the GPU
            return self.matches[idx]
        else:
            return {
                'champion_ids': self.matches[idx],
                'role_ids': self.role_ids,
                'team_ids': self.team_ids,
                'label': self.labels[idx],
                'blue_side': self.blue_side[idx],
                'region_id': self.region_ids[idx] if self.region_ids is not None else None,
                'aux_targets': self.aux_targets[idx] if self.aux_targets is not None else None,
                'dense_features': self.dense_features[idx] if self.has_dense_features else None,
            }


class GpuCache:
    """
    Pins the entire dataset onto the GPU and serves batches with zero CPU-GPU transfers
    per batch. Suitable when the full dataset fits in VRAM (~50MB for 660k matches).

    Replaces DataLoader for both pretrain and finetune modes. Batches are assembled
    from shuffled index slices entirely on-device.
    """

    def __init__(self, dataset: LeagueDataset, device: str, train_frac: float = 0.9, seed: int = 42):
        n = len(dataset)
        train_size = int(train_frac * n)

        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        train_idx = perm[:train_size]
        val_idx   = perm[train_size:]

        matches = dataset.matches.to(device)
        self.train_matches = matches[train_idx].contiguous()
        self.val_matches   = matches[val_idx].contiguous()

        self.mode = dataset.mode
        self.mask_token_id = dataset.mask_token_id
        self.train_size = train_size
        self.val_size = n - train_size

        if dataset.mode == 'finetune':
            labels = dataset.labels.to(device)
            self.train_labels = labels[train_idx].contiguous()
            self.val_labels   = labels[val_idx].contiguous()
            blue_side = dataset.blue_side.to(device)
            self.train_blue_side = blue_side[train_idx].contiguous()
            self.val_blue_side   = blue_side[val_idx].contiguous()
            if dataset.region_ids is not None:
                region_ids = dataset.region_ids.to(device)
                self.train_region_ids = region_ids[train_idx].contiguous()
                self.val_region_ids = region_ids[val_idx].contiguous()
            else:
                self.train_region_ids = None
                self.val_region_ids = None
            if dataset.aux_targets is not None:
                aux_targets = dataset.aux_targets.to(device)
                self.train_aux_targets = aux_targets[train_idx].contiguous()
                self.val_aux_targets = aux_targets[val_idx].contiguous()
            else:
                self.train_aux_targets = None
                self.val_aux_targets = None
            self.has_dense_features = dataset.has_dense_features
            if dataset.has_dense_features:
                dense = dataset.dense_features.to(device)
                self.train_dense_features = dense[train_idx].contiguous()
                self.val_dense_features   = dense[val_idx].contiguous()
            else:
                self.train_dense_features = None
                self.val_dense_features = None
            # role/team IDs are identical for every sample; keep one copy on GPU
            self.role_ids = dataset.role_ids.to(device)
            self.team_ids = dataset.team_ids.to(device)
        else:
            self.has_dense_features = False

    def batches(self, split: str, batch_size: int):
        """Yields batches from 'train' or 'val' split. Train split is reshuffled each call."""
        assert split in ('train', 'val'), "split must be 'train' or 'val'"
        matches = self.train_matches if split == 'train' else self.val_matches
        n = len(matches)
        idx = (torch.randperm(n, device=matches.device) if split == 'train'
               else torch.arange(n, device=matches.device))

        for i in range(0, n, batch_size):
            b = idx[i:i + batch_size]
            batch_matches = matches[b]

            if self.mode == 'finetune':
                labels = self.train_labels if split == 'train' else self.val_labels
                blue_side = self.train_blue_side if split == 'train' else self.val_blue_side
                region_ids = self.train_region_ids if split == 'train' else self.val_region_ids
                aux_targets = self.train_aux_targets if split == 'train' else self.val_aux_targets
                dense_features = self.train_dense_features if split == 'train' else self.val_dense_features
                bs = len(b)
                yield {
                    'champion_ids': batch_matches,
                    'role_ids':     self.role_ids.unsqueeze(0).expand(bs, -1),
                    'team_ids':     self.team_ids.unsqueeze(0).expand(bs, -1),
                    'label':        labels[b].unsqueeze(1),
                    'blue_side':    blue_side[b].unsqueeze(1),
                    'region_ids':   region_ids[b] if region_ids is not None else None,
                    'aux_targets':  aux_targets[b] if aux_targets is not None else None,
                    'dense_features': dense_features[b] if self.has_dense_features else None,
                }
            else:
                yield batch_matches

    def num_batches(self, split: str, batch_size: int) -> int:
        n = self.train_size if split == 'train' else self.val_size
        return (n + batch_size - 1) // batch_size
