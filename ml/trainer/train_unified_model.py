import gc
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.data.pytorch_data import GpuCache, LeagueDataset
from ml.predictor.unified_model import UnifiedWinPredictorModel
from ml.trainer.feature_pipeline import build_rich_feature_dataframe

try:
    from torch.amp import GradScaler, autocast
except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast  # type: ignore


AUX_TARGET_SCALES = (25000.0, 6.0, 6.0, 45.0)
AUX_TARGET_WEIGHTS = (0.25, 0.15, 0.15, 0.10)


def _compute_multitask_losses(win_logits, aux_predictions, labels, aux_targets, criterion):
    win_loss = criterion(win_logits, labels)
    if aux_predictions is None or aux_targets is None:
        return win_loss, win_loss.new_tensor(0.0)

    scales = aux_targets.new_tensor(AUX_TARGET_SCALES)
    weights = aux_targets.new_tensor(AUX_TARGET_WEIGHTS)
    component_losses = F.smooth_l1_loss(aux_predictions / scales, aux_targets / scales, reduction="none").mean(dim=0)
    aux_loss = torch.sum(component_losses * weights)
    return win_loss + aux_loss, aux_loss


def run_unified_training(
    num_champions=None,
    embedding_dim=96,
    nhead=4,
    dim_feedforward=256,
    num_layers=2,
    dropout=0.35,
    batch_size=1024,
    epochs=40,
    lr=5e-4,
    early_stopping_patience=12,
    early_stopping_min_delta=1e-4,
    min_epochs_before_stopping=20,
    scheduler_patience=3,
    scheduler_factor=0.5,
    scheduler_min_lr=1e-5,
    save_path="ml/save_data/best_unified_win_predictor.pth",
):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    amp_enabled = device == "cuda"
    print(f"[Unified] Device: {device}")
    print("Loading pre-match training data...")

    started_at = time.time()
    print("[Unified] Building feature dataframe from SQLite matches...", flush=True)
    df_matches, champion_list = build_rich_feature_dataframe()
    print(
        f"[Unified] Feature dataframe ready in {time.time() - started_at:.1f}s "
        f"({len(df_matches):,} rows x {len(df_matches.columns):,} columns)",
        flush=True,
    )
    actual_num_champions = len(champion_list)
    if num_champions is None:
        num_champions = actual_num_champions
    elif num_champions != actual_num_champions:
        raise ValueError(
            f"Configured num_champions={num_champions}, but dataset mapping contains "
            f"{actual_num_champions} champions."
        )

    started_at = time.time()
    print("[Unified] Converting feature dataframe to torch tensors...", flush=True)
    dataset = LeagueDataset(df_matches, mode="finetune")
    dense_feature_dim = dataset.dense_features.shape[1] if dataset.dense_features is not None else 0
    num_regions = int(dataset.region_ids.max().item()) + 1 if dataset.region_ids is not None and len(dataset.region_ids) else 1
    del df_matches
    gc.collect()
    print(f"[Unified] Torch dataset ready in {time.time() - started_at:.1f}s", flush=True)

    started_at = time.time()
    print(f"[Unified] Caching train/val tensors on {device}...", flush=True)
    cache = GpuCache(dataset, device)
    del dataset
    gc.collect()
    print(f"[Unified] Tensor cache ready in {time.time() - started_at:.1f}s", flush=True)
    print(f"[Unified] Champion vocab: {num_champions}")
    print(f"[Unified] Regions: {num_regions}")
    print(f"[Unified] Dense feature dim: {dense_feature_dim}")
    print(
        f"[Unified] Model: dim={embedding_dim}, heads={nhead}, layers={num_layers}, "
        f"ff={dim_feedforward}, dropout={dropout:.2f}"
    )
    print(f"Train: {cache.train_size:,} | Val: {cache.val_size:,} (cached on {device})")

    model = UnifiedWinPredictorModel(
        num_champions=num_champions,
        dense_feature_dim=dense_feature_dim,
        num_regions=num_regions,
        embedding_dim=embedding_dim,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr,
    )
    scaler = GradScaler(device="cuda", enabled=amp_enabled)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    n_train_batches = cache.num_batches("train", batch_size)
    n_val_batches = cache.num_batches("val", batch_size)

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_aux_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in cache.batches("train", batch_size):
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=amp_enabled):
                logits, aux_predictions = model(
                    batch["champion_ids"],
                    batch["role_ids"],
                    batch["team_ids"],
                    dense_features=batch["dense_features"],
                    blue_side=batch["blue_side"],
                    region_ids=batch["region_ids"],
                    return_aux=True,
                )
                labels = batch["label"]
                loss, aux_loss = _compute_multitask_losses(
                    logits,
                    aux_predictions,
                    labels,
                    batch["aux_targets"],
                    criterion,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_aux_loss += aux_loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_loss = 0.0
        val_aux_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in cache.batches("val", batch_size):
                with autocast(device_type="cuda", enabled=amp_enabled):
                    logits, aux_predictions = model(
                        batch["champion_ids"],
                        batch["role_ids"],
                        batch["team_ids"],
                        dense_features=batch["dense_features"],
                        blue_side=batch["blue_side"],
                        region_ids=batch["region_ids"],
                        return_aux=True,
                    )
                    labels = batch["label"]
                    loss, aux_loss = _compute_multitask_losses(
                        logits,
                        aux_predictions,
                        labels,
                        batch["aux_targets"],
                        criterion,
                    )

                val_loss += loss.item()
                val_aux_loss += aux_loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_train_loss = train_loss / n_train_batches
        avg_val_loss = val_loss / n_val_batches
        avg_train_aux_loss = train_aux_loss / n_train_batches
        avg_val_aux_loss = val_aux_loss / n_val_batches
        train_acc = train_correct / train_total if train_total else 0.0
        val_acc = val_correct / val_total if val_total else 0.0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Train Aux: {avg_train_aux_loss:.4f} | "
            f"Val Aux: {avg_val_aux_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {time.time()-t0:.1f}s"
        )

        if avg_val_loss < (best_val_loss - early_stopping_min_delta):
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"--> New Best Val Loss ({best_val_loss:.4f}) saved to '{save_path}'")
        else:
            epochs_without_improvement += 1

        previous_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val_loss)
        next_lr = optimizer.param_groups[0]["lr"]
        if next_lr < previous_lr:
            epochs_without_improvement = 0
            print(f"--> Reduced learning rate to {next_lr:.6f}")

        if (epoch + 1) >= min_epochs_before_stopping and epochs_without_improvement >= early_stopping_patience:
            print(
                f"--> Early stopping after {epoch+1} epochs "
                f"(no val-loss improvement greater than {early_stopping_min_delta:.1e} "
                f"for {early_stopping_patience} epochs after at least {min_epochs_before_stopping} epochs)."
            )
            break

    return model


if __name__ == "__main__":
    run_unified_training()
