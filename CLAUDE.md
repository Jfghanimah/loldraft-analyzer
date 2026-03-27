# CLAUDE.md

This file gives repository-specific guidance to coding agents working in this project.

## Source Of Truth

- `ROADMAP.MD` is the main planning document.
- `todo.txt` is the short execution queue and should stay aligned with the roadmap.

## Training Ownership

Humans should run meaningful training jobs for now.

Agents are useful for:

- code changes
- feature engineering
- experiment setup
- profiling hooks
- documentation
- updating `ROADMAP.MD` and other documentation to reflect completed work
- result interpretation after a human-run experiment

Agents should not casually kick off long training runs unless the user explicitly asks for it.

## Setup
A virtual environment already exists in the `venv` directory. Always activate it before installing dependencies or running scripts.

```bash
pip install -r requirements.txt
```

Requires a `.env` file with:

Optional local overrides can live in `.env`. This is the preferred way to set per-operator values such as:

```bash
RIOT_API_KEY=<local-key>
LOL_DRAFT_DB_PATH=league_data_v2_<operator>.db
LOL_DRAFT_COLLECTOR=<operator>
```

## Common Commands

Training (run from project root):

```bash
python -m ml.trainer.train
python -m ml.trainer.train --skip-pretrain
python -m ml.trainer.train --pretrain-only

python -m ml.trainer.pretrain_embeddings
python -m ml.trainer.train_win_predictor
```

Data work:

```bash
python -m ml.data.data_api_sqlite
python -m ml.data.migrate_to_sqlite
python -m ml.data.backfill_ordered_matches --limit 1000
python -m ml.data.merge_match_dbs --target league_data_v2_merged.db league_data_v2_a.db league_data_v2_b.db
```

Server:

```bash
uvicorn website.server:app --reload
```

Analysis scripts:

```bash
python ml/data/tests/api_test.py
python ml/data/tests/check_duplicates.py
python ml/predictor/tests/check_embeddings.py
python ml/predictor/tests/benchmark_model.py
```

Tests:

```bash
pytest
```

## Project Layout

```
ml/
  data/           # data ingestion, storage, format utilities
    tests/        # co-located tests + analysis scripts
  predictor/      # model definitions
    tests/        # co-located tests + benchmark scripts
  trainer/        # training scripts (pretrain + finetune)
    tests/        # co-located tests
  runtime_config.py
  save_data/      # champion_list.json, model checkpoints
website/
  server.py       # FastAPI BFF server (run with uvicorn website.server:app)
  tests/
```

## Current Architecture

### Data

- Current operational dataset: per-operator V2 SQLite DBs (e.g. `league_data_v2_joseph.db`), set via `LOL_DRAFT_DB_PATH` in `.env`
- Champion mapping: `ml/save_data/champion_list.json`
