# ML

This folder contains the data pipeline, training code, model definitions, and saved ML artifacts for LoL Draft Analyzer.

Important context:

- The current goal is to improve model quality on existing data before expanding collection again.
- The active scraper currently collects match payloads and ordered draft projections only.
- The active ML path is now one unified single-phase model.
- It trains from scratch on ordered drafts plus recent player-history features.
- Humans should launch meaningful training runs for now.
- The repo-root `ROADMAP.MD` is the big-picture plan. `ml/todo.txt` is the short local working list for ML-specific tasks.

Project structure:

- `data/`: SQLite collection, storage helpers, merge utilities, and dataset loading
- `trainer/`: active unified training entry point and feature pipeline
- `predictor/`: the active unified PyTorch model
- `z_leagacy_save_data/`: saved checkpoints and champion mappings
- `runtime_config.py`: shared scraper/runtime config loading

Useful entry points:

- `py -m ml.data.data_api_sqlite`
- `py -m ml.trainer.train`
- `py -m ml.trainer.train --finetune-epochs 80`
- `py -m ml.trainer.train --batch-size 2048`
- `py -m ml.trainer.train --dropout 0.40`

If you are new here, start with:

1. repo-root `ROADMAP.MD`
2. `ml/todo.txt`
3. `ml/trainer/train.py`
4. `ml/predictor/unified_model.py`
