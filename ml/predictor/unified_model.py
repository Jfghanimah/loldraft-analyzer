import torch
import torch.nn as nn


class UnifiedWinPredictorModel(nn.Module):
    """
    Pre-match model built around 10 enriched player-slot tokens.
    """

    def __init__(
        self,
        num_champions,
        dense_feature_dim,
        num_player_features=0,
        num_global_features=0,
        num_regions=1,
        embedding_dim=128,
        nhead=8,
        dim_feedforward=512,
        num_layers=3,
        dropout=0.40,
        trunk_hidden_dim=None,
        head_hidden_dim=None,
        architecture="flat",
    ):
        super().__init__()
        self.num_slots = 10
        self.dense_feature_dim = dense_feature_dim
        self.num_player_features = num_player_features
        self.num_global_features = num_global_features
        self.architecture = architecture
        self.ally_pairs = tuple((i, j) for i in range(5) for j in range(i + 1, 5))
        self.enemy_pairs = tuple((i, j) for i in range(5) for j in range(5))
        if architecture not in {"flat", "team_compare", "pairwise", "cls_global"}:
            raise ValueError("architecture must be one of: flat, team_compare, pairwise, cls_global")
        expected_dense_dim = (self.num_slots * num_player_features) + num_global_features
        if dense_feature_dim not in (0, expected_dense_dim):
            raise ValueError(
                f"dense_feature_dim={dense_feature_dim} does not match "
                f"{self.num_slots}*{num_player_features}+{num_global_features}={expected_dense_dim}"
            )

        self.champ_emb = nn.Embedding(num_champions + 2, embedding_dim)
        self.role_emb = nn.Embedding(5, embedding_dim)
        self.team_emb = nn.Embedding(2, embedding_dim)
        self.region_emb = nn.Embedding(max(num_regions, 1), embedding_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        if trunk_hidden_dim is None:
            trunk_hidden_dim = max(embedding_dim * 4, 512)
        if head_hidden_dim is None:
            head_hidden_dim = max(embedding_dim * 2, 256)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.slot_norm = nn.LayerNorm(embedding_dim)

        if num_player_features > 0:
            self.player_feature_encoder = nn.Sequential(
                nn.LayerNorm(num_player_features),
                nn.Linear(num_player_features, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, embedding_dim),
            )
        else:
            self.player_feature_encoder = None

        global_input_dim = embedding_dim + num_global_features
        self.global_context_encoder = nn.Sequential(
            nn.LayerNorm(global_input_dim),
            nn.Linear(global_input_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

        if architecture == "cls_global":
            sequence_dim = embedding_dim
        elif architecture == "pairwise":
            # team_compare features + allied products + enemy products/differences.
            sequence_dim = 89 * embedding_dim
        elif architecture == "team_compare":
            # Preserve slot identity while adding an explicit two-team comparison path.
            sequence_dim = (self.num_slots + 9) * embedding_dim
        else:
            sequence_dim = self.num_slots * embedding_dim
        self.shared_trunk = nn.Sequential(
            nn.LayerNorm(sequence_dim),
            nn.Linear(sequence_dim, trunk_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_hidden_dim, head_hidden_dim),
            nn.ReLU(),
        )
        self.win_head = nn.Linear(head_hidden_dim, 1)
        self.aux_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 4),
        )

    def _split_dense_features(self, dense_features, batch_size, device, dtype):
        if self.dense_feature_dim == 0:
            return None, None

        if dense_features is None:
            dense_features = torch.zeros(
                (batch_size, self.dense_feature_dim),
                device=device,
                dtype=dtype,
            )
        else:
            dense_features = dense_features.to(device=device, dtype=dtype)

        player_dense = None
        if self.num_player_features > 0:
            player_dim = self.num_slots * self.num_player_features
            player_dense = dense_features[:, :player_dim].reshape(
                batch_size,
                self.num_slots,
                self.num_player_features,
            )
        global_start = self.num_slots * self.num_player_features
        global_dense = dense_features[:, global_start:global_start + self.num_global_features]
        return player_dense, global_dense

    def _team_compare_features(self, x):
        blue = x[:, :5, :]
        red = x[:, 5:, :]
        blue_pool = blue.mean(dim=1)
        red_pool = red.mean(dim=1)
        lane_diff = blue - red
        comparison = torch.cat(
            [
                x.reshape(x.size(0), -1),
                blue_pool,
                red_pool,
                blue_pool - red_pool,
                (blue_pool - red_pool).abs(),
                lane_diff.reshape(x.size(0), -1),
            ],
            dim=1,
        )
        return comparison

    def _flatten_pair_products(self, left, right, pairs):
        products = [left[:, left_idx, :] * right[:, right_idx, :] for left_idx, right_idx in pairs]
        return torch.stack(products, dim=1).reshape(left.size(0), -1)

    def _flatten_pair_abs_diffs(self, left, right, pairs):
        diffs = [(left[:, left_idx, :] - right[:, right_idx, :]).abs() for left_idx, right_idx in pairs]
        return torch.stack(diffs, dim=1).reshape(left.size(0), -1)

    def _pairwise_features(self, x):
        blue = x[:, :5, :]
        red = x[:, 5:, :]
        return torch.cat(
            [
                self._team_compare_features(x),
                self._flatten_pair_products(blue, blue, self.ally_pairs),
                self._flatten_pair_products(red, red, self.ally_pairs),
                self._flatten_pair_products(blue, red, self.enemy_pairs),
                self._flatten_pair_abs_diffs(blue, red, self.enemy_pairs),
            ],
            dim=1,
        )

    def _sequence_features(self, x):
        if self.architecture == "cls_global":
            return x[:, 0, :]
        if self.architecture == "flat":
            return x.reshape(x.size(0), -1)
        if self.architecture == "team_compare":
            return self._team_compare_features(x)
        return self._pairwise_features(x)

    def forward(self, champion_ids, role_ids, team_ids, dense_features=None, region_ids=None, return_aux=False):
        x = self.champ_emb(champion_ids) + self.role_emb(role_ids) + self.team_emb(team_ids)
        batch_size = x.size(0)

        player_dense, global_dense = self._split_dense_features(
            dense_features,
            batch_size,
            x.device,
            x.dtype,
        )
        if player_dense is not None:
            x = x + self.player_feature_encoder(player_dense)

        if region_ids is None:
            region_features = self.region_emb(
                torch.zeros((batch_size,), device=x.device, dtype=torch.long)
            )
        else:
            region_features = self.region_emb(region_ids.to(device=x.device, dtype=torch.long))

        if global_dense is None:
            global_dense = x.new_zeros((batch_size, self.num_global_features))
        global_context = self.global_context_encoder(torch.cat([region_features, global_dense], dim=1))
        if self.architecture == "cls_global":
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x, global_context.unsqueeze(1)], dim=1)
        else:
            x = x + global_context.unsqueeze(1)

        x = self.slot_norm(self.transformer(x))
        shared = self.shared_trunk(self._sequence_features(x))
        win_logit = self.win_head(shared)
        if return_aux:
            return win_logit, self.aux_head(shared)
        return win_logit
