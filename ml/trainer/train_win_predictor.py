import os
import time

import torch
import torch.nn as nn

from ml.data.pytorch_data import LeagueDataset, GpuCache
from ml.trainer.feature_pipeline import build_rich_feature_dataframe
from ml.predictor.models_pytorch import WinPredictorModel


def run_finetune(
    num_champions=None,
    embedding_dim=128,
    batch_size=1024,
    epochs=40,
    lr=5e-4,
    early_stopping_patience=8,
    early_stopping_min_delta=1e-4,
    scheduler_patience=3,
    scheduler_factor=0.5,
    scheduler_min_lr=1e-5,
    pretrained_path='ml/save_data/pretrained_champ_embeddings.pth',
    save_path='ml/save_data/best_win_predictor.pth',
):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"[Phase 2] Device: {device}")

    print("Loading data...")
    df_matches, champion_list = build_rich_feature_dataframe()
    print("[Phase 2] Using unified pre-match feature pipeline.")
    actual_num_champions = len(champion_list)
    if num_champions is None:
        num_champions = actual_num_champions
    elif num_champions != actual_num_champions:
        raise ValueError(
            f"Configured num_champions={num_champions}, but dataset mapping contains "
            f"{actual_num_champions} champions."
        )

    print(f"[Phase 2] Champion vocab: {num_champions}")
    dataset = LeagueDataset(df_matches, mode='finetune')
    cache = GpuCache(dataset, device)
    print(f"Train: {cache.train_size:,} | Val: {cache.val_size:,} (cached on {device})")

    model = WinPredictorModel(
        num_champions=num_champions,
        embedding_dim=embedding_dim,
        num_layers=2,
        dropout=0.3,
        embedding_dropout=0.1,
        dense_feature_dropout=0.1,
        extra_feature_dim=(df_matches.shape[1] - 12),
    ).to(device)

    if os.path.exists(pretrained_path):
        print(f"Loading pre-trained embeddings from '{pretrained_path}'...")
        try:
            pretrained = torch.load(pretrained_path, map_location=device)
            if isinstance(pretrained, dict) and "champ_emb" in pretrained:
                model.champ_emb.load_state_dict(pretrained["champ_emb"])
                if "role_emb" in pretrained:
                    model.role_emb.load_state_dict(pretrained["role_emb"])
                if "team_emb" in pretrained:
                    model.team_emb.load_state_dict(pretrained["team_emb"])
            else:
                model.champ_emb.load_state_dict(pretrained)
        except RuntimeError as exc:
            raise RuntimeError(
                "Pretrained embedding checkpoint does not match the current champion vocab. "
                "Re-run Phase 1 so the embedding file is rebuilt against the updated mapping."
            ) from exc
        print("Embeddings loaded (unfrozen for fine-tuning).")
    else:
        print("WARNING: No pre-trained embeddings found. Training from scratch.")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr,
    )
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    n_train_batches = cache.num_batches('train', batch_size)
    n_val_batches   = cache.num_batches('val',   batch_size)

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for batch in cache.batches('train', batch_size):
            logits = model(
                batch['champion_ids'],
                batch['role_ids'],
                batch['team_ids'],
                batch['blue_side'],
                batch['dense_features'],
            )
            labels = batch['label']
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in cache.batches('val', batch_size):
                logits = model(
                    batch['champion_ids'],
                    batch['role_ids'],
                    batch['team_ids'],
                    batch['blue_side'],
                    batch['dense_features'],
                )
                labels = batch['label']
                val_loss += criterion(logits, labels).item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_train_loss = train_loss / n_train_batches
        avg_val_loss = val_loss / n_val_batches
        val_acc = val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Train Acc: {correct/total:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {time.time()-t0:.1f}s")

        if avg_val_loss < (best_val_loss - early_stopping_min_delta):
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            os.makedirs('ml/save_data', exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"--> New Best Val Loss ({best_val_loss:.4f}) saved to '{save_path}'")
        else:
            epochs_without_improvement += 1

        previous_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        next_lr = optimizer.param_groups[0]['lr']
        if next_lr < previous_lr:
            print(f"--> Reduced learning rate to {next_lr:.6f}")

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"--> Early stopping after {epoch+1} epochs "
                f"(no val-loss improvement greater than {early_stopping_min_delta:.1e} "
                f"for {early_stopping_patience} epochs)."
            )
            break

    return model


if __name__ == "__main__":
    run_finetune()
