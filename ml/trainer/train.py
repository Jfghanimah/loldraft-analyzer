"""
Unified training script. Runs Phase 1 (MLM pre-training) then Phase 2 (win prediction).
Phase 2 uses the unified richer pre-match feature pipeline by default.

Usage:
    python -m ml.trainer.train                  # Full pipeline (pretrain + finetune)
    python -m ml.trainer.train --skip-pretrain  # Fine-tune only (requires existing embeddings)
    python -m ml.trainer.train --pretrain-only  # Phase 1 only
    python -m ml.trainer.train --pretrain-epochs 35 --finetune-epochs 80
    python -m ml.trainer.train --skip-pretrain --finetune-patience 10
"""
import argparse

from ml.trainer.pretrain_embeddings import run_pretrain
from ml.trainer.train_win_predictor import run_finetune

DIVIDER = "=" * 50


def main():
    parser = argparse.ArgumentParser(description="Train the LoL draft win predictor")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--skip-pretrain', action='store_true',
                       help='Skip Phase 1 and fine-tune using existing embeddings')
    group.add_argument('--pretrain-only', action='store_true',
                       help='Run Phase 1 only')
    parser.add_argument(
        '--pretrain-epochs',
        type=int,
        default=25,
        help='Number of epochs for Phase 1 embedding pretraining (default: 25)',
    )
    parser.add_argument(
        '--finetune-epochs',
        type=int,
        default=40,
        help='Number of epochs for Phase 2 win-predictor training (default: 40)',
    )
    parser.add_argument(
        '--finetune-patience',
        type=int,
        default=12,
        help='Early stopping patience for Phase 2 based on validation loss (default: 12)',
    )
    parser.add_argument(
        '--finetune-min-delta',
        type=float,
        default=1e-4,
        help='Minimum validation-loss improvement to reset Phase 2 early stopping (default: 1e-4)',
    )
    parser.add_argument(
        '--finetune-min-epochs',
        type=int,
        default=20,
        help='Minimum number of Phase 2 epochs before early stopping can trigger (default: 20)',
    )
    parser.add_argument(
        '--finetune-lr-patience',
        type=int,
        default=3,
        help='ReduceLROnPlateau patience for Phase 2 (default: 3)',
    )
    parser.add_argument(
        '--finetune-lr-factor',
        type=float,
        default=0.5,
        help='ReduceLROnPlateau factor for Phase 2 (default: 0.5)',
    )
    args = parser.parse_args()

    if not args.skip_pretrain:
        print(DIVIDER)
        print("Phase 1: Pre-training Champion Embeddings")
        print(DIVIDER)
        run_pretrain(epochs=args.pretrain_epochs)

    if not args.pretrain_only:
        print(DIVIDER)
        print("Phase 2: Training Win Predictor")
        print(DIVIDER)
        run_finetune(
            epochs=args.finetune_epochs,
            early_stopping_patience=args.finetune_patience,
            early_stopping_min_delta=args.finetune_min_delta,
            min_epochs_before_stopping=args.finetune_min_epochs,
            scheduler_patience=args.finetune_lr_patience,
            scheduler_factor=args.finetune_lr_factor,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
