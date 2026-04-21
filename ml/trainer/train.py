"""
Unified training entry point.

One active model.
One active trainer.
One active command.
"""

import argparse
import os
import sys

from ml.trainer.train_unified_model import AUX_TARGET_WEIGHTS, run_unified_training

DIVIDER = "=" * 50
DEFAULT_TRAIN_LOG_PATH = "ml/save_data/latest_train_log.txt"


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description="Train the unified LoL draft win predictor")
    parser.add_argument("--finetune-epochs", type=int, default=20, help="Training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size (default: 512)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Optimizer learning rate (default: 3e-4)")
    parser.add_argument("--weight-decay", type=float, default=0.02, help="AdamW weight decay (default: 0.02)")
    parser.add_argument("--model-dim", type=int, default=96, help="Model embedding dim (default: 96)")
    parser.add_argument("--model-heads", type=int, default=4, help="Attention heads (default: 4)")
    parser.add_argument("--model-layers", type=int, default=2, help="Transformer layers (default: 2)")
    parser.add_argument("--model-ff", type=int, default=256, help="Feedforward width (default: 256)")
    parser.add_argument(
        "--architecture",
        choices=("flat", "team_compare", "pairwise", "cls_global"),
        default="flat",
        help="Prediction head architecture (default: flat)",
    )
    parser.add_argument("--dropout", type=float, default=0.45, help="Model dropout (default: 0.45)")
    parser.add_argument("--finetune-patience", type=int, default=20, help="Early stopping patience (default: 20)")
    parser.add_argument("--finetune-min-delta", type=float, default=1e-4, help="Minimum val-loss improvement (default: 1e-4)")
    parser.add_argument("--finetune-min-epochs", type=int, default=20, help="Minimum epochs before stopping (default: 20)")
    parser.add_argument("--finetune-lr-patience", type=int, default=6, help="LR scheduler patience (default: 6)")
    parser.add_argument("--finetune-lr-factor", type=float, default=0.5, help="LR scheduler factor (default: 0.5)")
    parser.add_argument("--finetune-min-lr", type=float, default=2e-5, help="Minimum LR floor (default: 2e-5)")
    parser.add_argument("--refresh-feature-cache", action="store_true", help="Rebuild the cached feature dataframe before training")
    parser.add_argument("--feature-cache-path", default="ml/save_data/unified_feature_cache.pkl", help="Path to the cached feature dataframe bundle")
    parser.add_argument("--training-data-dir", default=None, help="Prepared compact Parquet dataset root containing training_examples")
    parser.add_argument("--save-path", default="ml/save_data/best_unified_win_predictor.pth", help="Path for the best checkpoint")
    parser.add_argument("--train-log-path", default=DEFAULT_TRAIN_LOG_PATH, help="Path for the training log")
    parser.add_argument(
        "--selection-metric",
        choices=("val_loss", "val_acc"),
        default="val_loss",
        help="Metric used to select the saved checkpoint (default: val_loss)",
    )
    parser.add_argument(
        "--aux-target-weights",
        type=float,
        nargs=4,
        metavar=("GOLD", "BLUE_DRAGONS", "RED_DRAGONS", "GAME_LENGTH"),
        default=None,
        help="Auxiliary loss weights for gold, blue dragons, red dragons, and game length",
    )
    parser.add_argument("--seed", type=int, default=42, help="Torch RNG seed for model init and train shuffling")
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Binary label smoothing floor; 0.15 maps targets to 0.15/0.85",
    )
    parser.add_argument(
        "--drop-population-priors",
        action="store_true",
        help="Drop champ-role population win-rate/frequency priors from dense inputs",
    )
    args = parser.parse_args()

    train_log_dir = os.path.dirname(args.train_log_path)
    if train_log_dir:
        os.makedirs(train_log_dir, exist_ok=True)
    with open(args.train_log_path, "w", encoding="utf-8") as log_file:
        tee = _TeeStream(sys.stdout, log_file)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = tee
        sys.stderr = tee
        try:
            print(DIVIDER)
            print("Training Unified Win Predictor")
            print(DIVIDER)
            print(f"[Unified] Writing training log to {args.train_log_path}")
            run_unified_training(
                epochs=args.finetune_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                embedding_dim=args.model_dim,
                nhead=args.model_heads,
                num_layers=args.model_layers,
                dim_feedforward=args.model_ff,
                architecture=args.architecture,
                dropout=args.dropout,
                early_stopping_patience=args.finetune_patience,
                early_stopping_min_delta=args.finetune_min_delta,
                min_epochs_before_stopping=args.finetune_min_epochs,
                scheduler_patience=args.finetune_lr_patience,
                scheduler_factor=args.finetune_lr_factor,
                scheduler_min_lr=args.finetune_min_lr,
                training_data_dir=args.training_data_dir,
                feature_cache_path=args.feature_cache_path,
                save_path=args.save_path,
                selection_metric=args.selection_metric,
                aux_target_weights=args.aux_target_weights if args.aux_target_weights is not None else AUX_TARGET_WEIGHTS,
                seed=args.seed,
                label_smoothing=args.label_smoothing,
                drop_population_priors=args.drop_population_priors,
                refresh_feature_cache=args.refresh_feature_cache,
            )
            print("\nDone.")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
