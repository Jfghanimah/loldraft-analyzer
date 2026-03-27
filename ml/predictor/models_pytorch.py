import torch
import torch.nn as nn


class MLMPretrainModel(nn.Module):
    """
    Transformer encoder trained with a BERT-style masked language modeling task.
    Learns champion embeddings with explicit role/team context so slot-specific
    picks like jungle-vs-support are distinguishable during pretraining.
    """
    def __init__(self, num_champions, embedding_dim=128, nhead=4, dim_feedforward=256, num_layers=3):
        super().__init__()
        vocab_size = num_champions + 2  # champions + MASK + PAD

        self.champ_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.role_emb = nn.Embedding(num_embeddings=5, embedding_dim=embedding_dim)
        self.team_emb = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, champion_ids, role_ids, team_ids):
        x = self.champ_emb(champion_ids) + self.role_emb(role_ids) + self.team_emb(team_ids)
        x = self.transformer_encoder(x)
        return self.head(x)


class WinPredictorModel(nn.Module):
    """
    Transformer model for win prediction using pre-trained champion embeddings.

    Each of the 10 champion slots receives summed champion + role + team embeddings.
    The classifier input combines:
      - Flattened per-slot features  (10 * embedding_dim)
      - Per-lane blue-minus-red diff  (5  * embedding_dim)
      - Explicit blue-side indicator (1)
    Optional dense features can be concatenated before classification.
    """
    def __init__(
        self,
        num_champions,
        embedding_dim=128,
        nhead=4,
        dim_feedforward=256,
        dropout=0.3,
        num_layers=2,
        extra_feature_dim=0,
        embedding_dropout=0.1,
        dense_feature_dropout=0.1,
    ):
        super().__init__()

        self.champ_emb = nn.Embedding(num_champions + 2, embedding_dim)
        self.role_emb  = nn.Embedding(5, embedding_dim)
        self.team_emb  = nn.Embedding(2, embedding_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.dense_feature_dropout = nn.Dropout(dense_feature_dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.extra_feature_dim = extra_feature_dim
        input_dim = 15 * embedding_dim + 1 + extra_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, champion_ids, role_ids, team_ids, blue_side=None, dense_features=None):
        x = self.champ_emb(champion_ids) + self.role_emb(role_ids) + self.team_emb(team_ids)
        x = self.embedding_dropout(x)
        x = self.transformer(x)  # [B, 10, dim]

        flattened = x.view(x.size(0), -1)                          # [B, 10*dim]
        lane_diff = (x[:, :5, :] - x[:, 5:, :]).view(x.size(0), -1)  # [B, 5*dim]
        if blue_side is None:
            blue_side = torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype)
        else:
            blue_side = blue_side.to(dtype=x.dtype)
            if blue_side.dim() == 1:
                blue_side = blue_side.unsqueeze(1)

        features = [flattened, lane_diff, blue_side]
        if self.extra_feature_dim:
            if dense_features is None:
                dense_features = torch.zeros(
                    (x.size(0), self.extra_feature_dim),
                    device=x.device,
                    dtype=x.dtype,
                )
            else:
                dense_features = dense_features.to(dtype=x.dtype)
            dense_features = self.dense_feature_dropout(dense_features)
            features.append(dense_features)

        return self.classifier(torch.cat(features, dim=1))
