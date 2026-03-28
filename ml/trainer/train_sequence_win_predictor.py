import os
import time

import torch
import torch.nn as nn

from ml.predictor.models_pytorch import SequenceWinPredictorModel
from ml.trainer.sequence_pipeline import build_sequence_training_tensors

try:
    from torch.amp import GradScaler, autocast
except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast  # type: ignore


class SequenceGpuCache:
    def __init__(self, tensors, device, train_frac=0.9, seed=42):
        n = len(tensors["labels"])
        train_size = int(train_frac * n)

        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=generator)
        train_idx = perm[:train_size]
        val_idx = perm[train_size:]

        self.train_size = train_size
        self.val_size = n - train_size
        self.device = device
        self.tensors = {}
        for name, tensor in tensors.items():
            on_device = tensor.to(device)
            self.tensors[f"train_{name}"] = on_device[train_idx].contiguous()
            self.tensors[f"val_{name}"] = on_device[val_idx].contiguous()

    def num_batches(self, split, batch_size):
        size = self.train_size if split == "train" else self.val_size
        return (size + batch_size - 1) // batch_size

    def batches(self, split, batch_size):
        size = self.train_size if split == "train" else self.val_size
        idx = (
            torch.randperm(size, device=self.device)
            if split == "train"
            else torch.arange(size, device=self.device)
        )
        prefix = "train" if split == "train" else "val"
        for start in range(0, size, batch_size):
            batch_idx = idx[start:start + batch_size]
            batch = {}
            for name in (
                "labels",
                "blue_side",
                "region_ids",
                "patch_features",
                "outcome_targets",
                "current_champion_ids",
                "current_role_ids",
                "current_team_ids",
                "history_champion_ids",
                "history_role_ids",
                "history_slot_ids",
                "history_result_ids",
                "history_numeric",
                "history_mask",
            ):
                batch[name] = self.tensors[f"{prefix}_{name}"][batch_idx]
            batch["labels"] = batch["labels"].unsqueeze(1)
            batch["blue_side"] = batch["blue_side"].unsqueeze(1)
            yield batch

def run_sequence_finetune(
    batch_size=1024,
    epochs=40,
    lr=3e-4,
    history_length=10,
    embedding_dim=96,
    nhead=4,
    dim_feedforward=192,
    num_layers=2,
    early_stopping_patience=12,
    early_stopping_min_delta=1e-4,
    min_epochs_before_stopping=20,
    scheduler_patience=3,  # accepted for CLI compatibility; OneCycleLR ignores it
    scheduler_factor=0.5,  # accepted for CLI compatibility; OneCycleLR ignores it
    scheduler_min_lr=1e-5,  # accepted for CLI compatibility; OneCycleLR ignores it
    pretrained_path="ml/save_data/pretrained_champ_embeddings.pth",  # unused in E2E path
    save_path="ml/save_data/best_sequence_win_predictor.pth",
    outcome_loss_weight=0.1,
):
    del scheduler_patience, scheduler_factor, scheduler_min_lr, pretrained_path

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    amp_enabled = device == "cuda"
    print(f"[Sequence] Device: {device}")
    print("[Sequence] Using single-phase end-to-end multitask training.")
    print("Loading sequence-preserving data...")
    tensors, metadata = build_sequence_training_tensors(history_length=history_length)
    cache = SequenceGpuCache(tensors, device)
    print(f"[Sequence] Champion vocab: {metadata['num_champions']}")
    print(f"[Sequence] Regions: {metadata['num_regions']}")
    print(
        f"[Sequence] Model: dim={embedding_dim}, heads={nhead}, layers={num_layers}, "
        f"ff={dim_feedforward}, history={metadata['history_length']}"
    )
    print(
        f"Train: {cache.train_size:,} | Val: {cache.val_size:,} "
        f"(history={metadata['history_length']}, batch={batch_size}, cached on {device})"
    )

    model = SequenceWinPredictorModel(
        num_champions=metadata["num_champions"],
        num_regions=metadata["num_regions"],
        history_length=metadata["history_length"],
        embedding_dim=embedding_dim,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_layers=num_layers,
    ).to(device)

    win_criterion = nn.BCEWithLogitsLoss()
    outcome_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n_train_batches = cache.num_batches("train", batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=n_train_batches,
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=100.0,
    )
    scaler = GradScaler(device="cuda", enabled=amp_enabled)

    best_val_win_loss = float("inf")
    epochs_without_improvement = 0
    n_val_batches = cache.num_batches("val", batch_size)

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_total_loss = 0.0
        train_win_loss = 0.0
        train_outcome_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in cache.batches("train", batch_size):
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=amp_enabled):
                outputs = model(
                    batch["current_champion_ids"],
                    batch["current_role_ids"],
                    batch["current_team_ids"],
                    batch["region_ids"],
                    batch["patch_features"],
                    batch["history_champion_ids"],
                    batch["history_role_ids"],
                    batch["history_slot_ids"],
                    batch["history_result_ids"],
                    batch["history_numeric"],
                    batch["history_mask"],
                    return_aux_outputs=True,
                )
                labels = batch["labels"]
                win_loss = win_criterion(outputs["win_logit"], labels)
                outcome_loss = outcome_criterion(outputs["outcome_logits"], batch["outcome_targets"])
                total_loss = win_loss + (outcome_loss_weight * outcome_loss)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_total_loss += total_loss.item()
            train_win_loss += win_loss.item()
            train_outcome_loss += outcome_loss.item()
            preds = (torch.sigmoid(outputs["win_logit"]) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_total_loss = 0.0
        val_win_loss = 0.0
        val_outcome_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in cache.batches("val", batch_size):
                with autocast(device_type="cuda", enabled=amp_enabled):
                    outputs = model(
                        batch["current_champion_ids"],
                        batch["current_role_ids"],
                        batch["current_team_ids"],
                        batch["region_ids"],
                        batch["patch_features"],
                        batch["history_champion_ids"],
                        batch["history_role_ids"],
                        batch["history_slot_ids"],
                        batch["history_result_ids"],
                        batch["history_numeric"],
                        batch["history_mask"],
                        return_aux_outputs=True,
                    )
                    labels = batch["labels"]
                    win_loss = win_criterion(outputs["win_logit"], labels)
                    outcome_loss = outcome_criterion(outputs["outcome_logits"], batch["outcome_targets"])
                    total_loss = win_loss + (outcome_loss_weight * outcome_loss)

                val_total_loss += total_loss.item()
                val_win_loss += win_loss.item()
                val_outcome_loss += outcome_loss.item()
                preds = (torch.sigmoid(outputs["win_logit"]) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_train_total_loss = train_total_loss / n_train_batches
        avg_train_win_loss = train_win_loss / n_train_batches
        avg_train_outcome_loss = train_outcome_loss / n_train_batches
        avg_val_total_loss = val_total_loss / n_val_batches
        avg_val_win_loss = val_win_loss / n_val_batches
        avg_val_outcome_loss = val_outcome_loss / n_val_batches
        train_acc = train_correct / train_total if train_total else 0.0
        val_acc = val_correct / val_total if val_total else 0.0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Total T/V: {avg_train_total_loss:.4f}/{avg_val_total_loss:.4f} | "
            f"Win T/V: {avg_train_win_loss:.4f}/{avg_val_win_loss:.4f} | "
            f"Outcome T/V: {avg_train_outcome_loss:.4f}/{avg_val_outcome_loss:.4f} | "
            f"Acc T/V: {train_acc:.4f}/{val_acc:.4f} | "
            f"LR: {current_lr:.6f} | Time: {time.time()-t0:.1f}s"
        )

        if avg_val_win_loss < (best_val_win_loss - early_stopping_min_delta):
            best_val_win_loss = avg_val_win_loss
            epochs_without_improvement = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"--> New Best Val Win Loss ({best_val_win_loss:.4f}) saved to '{save_path}'")
        else:
            epochs_without_improvement += 1

        if (epoch + 1) >= min_epochs_before_stopping and epochs_without_improvement >= early_stopping_patience:
            print(
                f"--> Early stopping after {epoch+1} epochs "
                f"(no val-win-loss improvement greater than {early_stopping_min_delta:.1e} "
                f"for {early_stopping_patience} epochs after at least {min_epochs_before_stopping} epochs)."
            )
            break

    return model


if __name__ == "__main__":
    run_sequence_finetune()
