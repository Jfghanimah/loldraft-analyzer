import torch
import torch.nn as nn
import os
import time

from ml.data.pytorch_data import get_data_frames, LeagueDataset, GpuCache
from ml.predictor.models_pytorch import MLMPretrainModel

ROLE_IDS = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=torch.long)
TEAM_IDS = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)


def _apply_mask(target_ids, mask_token_id):
    """Randomly masks one champion slot per sample. Returns (input_ids, loss_labels)."""
    device = target_ids.device
    batch_size = target_ids.size(0)

    input_ids = target_ids.clone()
    mask_indices = torch.randint(0, 10, (batch_size, 1), device=device)
    input_ids.scatter_(1, mask_indices, mask_token_id)

    loss_labels = torch.full_like(target_ids, fill_value=-100)
    flat_mask = torch.arange(batch_size, device=device) * 10 + mask_indices.squeeze(1)
    loss_labels.view(-1)[flat_mask] = target_ids.view(-1)[flat_mask]
    return input_ids, loss_labels


def run_pretrain(
    num_champions=None,
    embedding_dim=128,
    batch_size=1024,
    epochs=25,
    lr=1e-3,
    save_path='ml/save_data/pretrained_champ_embeddings.pth',
    full_save_path='ml/save_data/pretrained_mlm_full.pth',
):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"[Phase 1] Device: {device}")
    print("Loading data...")

    df_matches, champion_list = get_data_frames()
    actual_num_champions = len(champion_list)
    if num_champions is None:
        num_champions = actual_num_champions
    elif num_champions != actual_num_champions:
        raise ValueError(
            f"Configured num_champions={num_champions}, but dataset mapping contains "
            f"{actual_num_champions} champions."
        )

    mask_token_id = num_champions
    print(f"[Phase 1] Champion vocab: {num_champions}")

    dataset = LeagueDataset(df_matches, mode='pretrain', mask_token_id=mask_token_id)
    cache = GpuCache(dataset, device)
    print(f"Train: {cache.train_size:,} | Val: {cache.val_size:,} (cached on {device})")

    model = MLMPretrainModel(
        num_champions=num_champions,
        embedding_dim=embedding_dim,
        nhead=4,
        dim_feedforward=256,
        num_layers=3,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    role_ids = ROLE_IDS.to(device)
    team_ids = TEAM_IDS.to(device)
    best_val_loss = float("inf")

    n_train_batches = cache.num_batches('train', batch_size)
    n_val_batches   = cache.num_batches('val',   batch_size)

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for target_ids in cache.batches('train', batch_size):
            input_ids, loss_labels = _apply_mask(target_ids, mask_token_id)
            batch_role_ids = role_ids.unsqueeze(0).expand(target_ids.size(0), -1)
            batch_team_ids = team_ids.unsqueeze(0).expand(target_ids.size(0), -1)
            logits = model(input_ids, batch_role_ids, batch_team_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), loss_labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for target_ids in cache.batches('val', batch_size):
                input_ids, loss_labels = _apply_mask(target_ids, mask_token_id)
                batch_role_ids = role_ids.unsqueeze(0).expand(target_ids.size(0), -1)
                batch_team_ids = team_ids.unsqueeze(0).expand(target_ids.size(0), -1)
                logits = model(input_ids, batch_role_ids, batch_team_ids)
                val_loss += criterion(logits.view(-1, logits.size(-1)), loss_labels.view(-1)).item()

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss/n_train_batches:.4f} | "
              f"Val Loss: {val_loss/n_val_batches:.4f} | "
              f"Time: {time.time()-t0:.1f}s")

        avg_val_loss = val_loss / n_val_batches
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'num_champions': num_champions,
                    'embedding_dim': embedding_dim,
                    'nhead': 4,
                    'dim_feedforward': 256,
                    'num_layers': 3,
                    'best_val_loss': best_val_loss,
                },
                full_save_path,
            )
            print(f"--> New Best MLM Val Loss ({best_val_loss:.4f}) saved to '{full_save_path}'")

    os.makedirs('ml/save_data', exist_ok=True)
    torch.save(
        {
            'champ_emb': model.champ_emb.state_dict(),
            'role_emb': model.role_emb.state_dict(),
            'team_emb': model.team_emb.state_dict(),
        },
        save_path,
    )
    print(f"Embeddings saved to '{save_path}'")
    return model


if __name__ == "__main__":
    run_pretrain()
