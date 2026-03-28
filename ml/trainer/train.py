"""
Unified training entry point.

One active model.
One active trainer.
One active command.
"""

import argparse

from ml.trainer.train_unified_model import run_unified_training

DIVIDER = "=" * 50


def main():
    parser = argparse.ArgumentParser(description="Train the unified LoL draft win predictor")
    parser.add_argument("--finetune-epochs", type=int, default=40, help="Training epochs (default: 40)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size (default: 1024)")
    parser.add_argument("--model-dim", type=int, default=96, help="Model embedding dim (default: 96)")
    parser.add_argument("--model-heads", type=int, default=4, help="Attention heads (default: 4)")
    parser.add_argument("--model-layers", type=int, default=2, help="Transformer layers (default: 2)")
    parser.add_argument("--model-ff", type=int, default=256, help="Feedforward width (default: 256)")
    parser.add_argument("--dropout", type=float, default=0.35, help="Model dropout (default: 0.35)")
    parser.add_argument("--finetune-patience", type=int, default=12, help="Early stopping patience (default: 12)")
    parser.add_argument("--finetune-min-delta", type=float, default=1e-4, help="Minimum val-loss improvement (default: 1e-4)")
    parser.add_argument("--finetune-min-epochs", type=int, default=20, help="Minimum epochs before stopping (default: 20)")
    parser.add_argument("--finetune-lr-patience", type=int, default=3, help="LR scheduler patience (default: 3)")
    parser.add_argument("--finetune-lr-factor", type=float, default=0.5, help="LR scheduler factor (default: 0.5)")
    args = parser.parse_args()

    print(DIVIDER)
    print("Training Unified Win Predictor")
    print(DIVIDER)
    run_unified_training(
        epochs=args.finetune_epochs,
        batch_size=args.batch_size,
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
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
