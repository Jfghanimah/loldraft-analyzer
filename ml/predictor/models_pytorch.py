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
    ):
        super().__init__()

        self.champ_emb = nn.Embedding(num_champions + 2, embedding_dim)
        self.role_emb  = nn.Embedding(5, embedding_dim)
        self.team_emb  = nn.Embedding(2, embedding_dim)

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
            features.append(dense_features)

        return self.classifier(torch.cat(features, dim=1))


class SequenceWinPredictorModel(nn.Module):
    """
    Sequence-preserving pre-match transformer.

    Tokens:
      - 1 CLS token
      - 1 region token
      - 1 patch token
      - 10 current draft tokens
      - 10 * history_length prior-match history tokens
    """

    def __init__(
        self,
        num_champions,
        num_regions,
        history_length,
        embedding_dim=96,
        nhead=4,
        dim_feedforward=192,
        dropout=0.2,
        num_layers=2,
    ):
        super().__init__()
        self.num_champions = num_champions
        self.pad_token_id = num_champions + 1
        self.history_length = history_length
        self.history_numeric_dim = 9

        self.champ_emb = nn.Embedding(num_champions + 2, embedding_dim)
        self.role_emb = nn.Embedding(5, embedding_dim)
        self.team_emb = nn.Embedding(2, embedding_dim)
        self.slot_emb = nn.Embedding(10, embedding_dim)
        self.result_emb = nn.Embedding(2, embedding_dim)
        self.history_position_emb = nn.Embedding(history_length, embedding_dim)
        self.region_emb = nn.Embedding(max(num_regions, 1), embedding_dim)
        self.token_type_emb = nn.Embedding(4, embedding_dim)
        self.patch_proj = nn.Linear(2, embedding_dim)
        self.history_numeric_proj = nn.Linear(self.history_numeric_dim, embedding_dim)
        self.patch_norm = nn.LayerNorm(embedding_dim)
        self.history_numeric_norm = nn.LayerNorm(embedding_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.register_buffer("_attention_mask", self._build_attention_mask(history_length), persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )
        self.outcome_head = nn.Linear(embedding_dim, 4)

    @staticmethod
    def _build_attention_mask(history_length):
        draft_prefix_tokens = 13
        total_tokens = draft_prefix_tokens + (10 * history_length)
        mask = torch.ones((total_tokens, total_tokens), dtype=torch.bool)

        # CLS, region, patch, and all 10 draft tokens can attend to everything.
        mask[:draft_prefix_tokens, :] = False

        # History tokens can see global context, their own draft token, and their own history block.
        for slot_index in range(10):
            draft_token_index = 3 + slot_index
            history_start = draft_prefix_tokens + (slot_index * history_length)
            history_end = history_start + history_length
            mask[history_start:history_end, 0:3] = False
            mask[history_start:history_end, draft_token_index] = False
            mask[history_start:history_end, history_start:history_end] = False

        return mask

    def encode(
        self,
        current_champion_ids,
        current_role_ids,
        current_team_ids,
        region_ids,
        patch_features,
        history_champion_ids,
        history_role_ids,
        history_slot_ids,
        history_result_ids,
        history_numeric,
        history_mask,
    ):
        batch_size = current_champion_ids.size(0)
        device = current_champion_ids.device

        cls = self.cls_token.expand(batch_size, -1, -1) + self.token_type_emb(
            torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        )
        region_token = self.region_emb(region_ids).unsqueeze(1) + self.token_type_emb(
            torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
        )
        patch_token = self.patch_norm(self.patch_proj(patch_features)).unsqueeze(1) + self.token_type_emb(
            torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
        )

        slot_ids = torch.arange(10, device=device).unsqueeze(0).expand(batch_size, -1)
        draft_tokens = (
            self.champ_emb(current_champion_ids)
            + self.role_emb(current_role_ids)
            + self.team_emb(current_team_ids)
            + self.slot_emb(slot_ids)
            + self.token_type_emb(torch.full((batch_size, 10), 2, dtype=torch.long, device=device))
        )

        hist_champs = history_champion_ids.view(batch_size, -1)
        hist_roles = history_role_ids.view(batch_size, -1)
        hist_slots = history_slot_ids.view(batch_size, -1)
        hist_results = history_result_ids.view(batch_size, -1)
        hist_numeric = history_numeric.view(batch_size, -1, self.history_numeric_dim)
        hist_mask_flat = history_mask.view(batch_size, -1)
        hist_positions = (
            torch.arange(self.history_length, device=device)
            .view(1, 1, self.history_length)
            .expand(batch_size, 10, self.history_length)
            .reshape(batch_size, -1)
        )

        history_tokens = (
            self.champ_emb(hist_champs)
            + self.role_emb(hist_roles)
            + self.slot_emb(hist_slots)
            + self.result_emb(hist_results)
            + self.history_position_emb(hist_positions)
            + self.history_numeric_norm(self.history_numeric_proj(hist_numeric))
            + self.token_type_emb(
                torch.full((batch_size, hist_champs.size(1)), 3, dtype=torch.long, device=device)
            )
        )

        sequence = torch.cat([cls, region_token, patch_token, draft_tokens, history_tokens], dim=1)
        padding_mask = torch.cat(
            [
                torch.zeros((batch_size, 13), dtype=torch.bool, device=device),
                ~hist_mask_flat,
            ],
            dim=1,
        )
        encoded = self.transformer(
            sequence,
            mask=self._attention_mask.to(device=device),
            src_key_padding_mask=padding_mask,
        )
        return encoded

    def forward(
        self,
        current_champion_ids,
        current_role_ids,
        current_team_ids,
        region_ids,
        patch_features,
        history_champion_ids,
        history_role_ids,
        history_slot_ids,
        history_result_ids,
        history_numeric,
        history_mask,
        *,
        return_aux_outputs=False,
    ):
        encoded = self.encode(
            current_champion_ids,
            current_role_ids,
            current_team_ids,
            region_ids,
            patch_features,
            history_champion_ids,
            history_role_ids,
            history_slot_ids,
            history_result_ids,
            history_numeric,
            history_mask,
        )
        win_logit = self.classifier(encoded[:, 0, :])
        if not return_aux_outputs:
            return win_logit
        return {
            "win_logit": win_logit,
            "outcome_logits": self.outcome_head(encoded[:, 0, :]),
        }
