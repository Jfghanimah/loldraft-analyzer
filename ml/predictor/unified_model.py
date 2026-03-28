import torch
import torch.nn as nn


class UnifiedWinPredictorModel(nn.Module):
    """
    Clean single-phase pre-match model.

    Inputs:
    - 10 ordered champion IDs
    - fixed role/team IDs
    - dense recent-history features
    - optional blue-side scalar
    """

    def __init__(
        self,
        num_champions,
        dense_feature_dim,
        embedding_dim=96,
        nhead=4,
        dim_feedforward=256,
        num_layers=2,
        dropout=0.35,
        dense_hidden_dim=128,
    ):
        super().__init__()
        self.champ_emb = nn.Embedding(num_champions + 2, embedding_dim)
        self.role_emb = nn.Embedding(5, embedding_dim)
        self.team_emb = nn.Embedding(2, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.slot_norm = nn.LayerNorm(embedding_dim)

        self.dense_feature_dim = dense_feature_dim
        if dense_feature_dim > 0:
            self.dense_encoder = nn.Sequential(
                nn.LayerNorm(dense_feature_dim),
                nn.Linear(dense_feature_dim, dense_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dense_hidden_dim, dense_hidden_dim),
                nn.GELU(),
            )
        else:
            self.dense_encoder = None

        summary_dim = (4 * embedding_dim) + (5 * embedding_dim) + 1 + (
            dense_hidden_dim if dense_feature_dim > 0 else 0
        )
        self.head = nn.Sequential(
            nn.Linear(summary_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, champion_ids, role_ids, team_ids, dense_features=None, blue_side=None):
        x = self.champ_emb(champion_ids) + self.role_emb(role_ids) + self.team_emb(team_ids)
        x = self.slot_norm(self.transformer(x))

        blue_slots = x[:, :5, :]
        red_slots = x[:, 5:, :]
        global_pool = x.mean(dim=1)
        blue_pool = blue_slots.mean(dim=1)
        red_pool = red_slots.mean(dim=1)
        team_gap = blue_pool - red_pool
        lane_diffs = (blue_slots - red_slots).reshape(x.size(0), -1)

        if blue_side is None:
            blue_side = torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype)
        else:
            blue_side = blue_side.to(dtype=x.dtype)
            if blue_side.dim() == 1:
                blue_side = blue_side.unsqueeze(1)

        features = [global_pool, blue_pool, red_pool, team_gap, lane_diffs, blue_side]
        if self.dense_encoder is not None:
            if dense_features is None:
                dense_features = torch.zeros((x.size(0), self.dense_feature_dim), device=x.device, dtype=x.dtype)
            else:
                dense_features = dense_features.to(dtype=x.dtype)
            features.append(self.dense_encoder(dense_features))

        return self.head(torch.cat(features, dim=1))
