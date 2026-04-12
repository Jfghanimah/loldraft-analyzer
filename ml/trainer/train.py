"""
Unified training entry point.

One active model.
One active trainer.
One active command.
"""

import argparse
import os
import sys

from ml.trainer.train_unified_model import run_unified_training

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
    args = parser.parse_args()

    os.makedirs(os.path.dirname(DEFAULT_TRAIN_LOG_PATH), exist_ok=True)
    with open(DEFAULT_TRAIN_LOG_PATH, "w", encoding="utf-8") as log_file:
        tee = _TeeStream(sys.stdout, log_file)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = tee
        sys.stderr = tee
        try:
            print(DIVIDER)
            print("Training Unified Win Predictor")
            print(DIVIDER)
            print(f"[Unified] Writing training log to {DEFAULT_TRAIN_LOG_PATH}")
            run_unified_training(
                epochs=args.finetune_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                embedding_dim=args.model_dim,
                nhead=args.model_heads,
                num_layers=args.model_layers,
                dim_feedforward=args.model_ff,
                dropout=args.dropout,
                early_stopping_patience=args.finetune_patience,
                early_stopping_min_delta=args.finetune_min_delta,
                min_epochs_before_stopping=args.finetune_min_epochs,
                scheduler_patience=args.finetune_lr_patience,
                scheduler_factor=args.finetune_lr_factor,
                scheduler_min_lr=args.finetune_min_lr,
                training_data_dir=args.training_data_dir,
                feature_cache_path=args.feature_cache_path,
                refresh_feature_cache=args.refresh_feature_cache,
            )
            print("\nDone.")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
