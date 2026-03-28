# LoL Draft Analyzer

LoL Draft Analyzer is a PyTorch project for predicting League of Legends match outcomes from champion draft compositions. The codebase includes a SQLite-backed collection pipeline and one active unified ML training path built around ordered drafts plus recent player-history features.

## Setup

A virtual environment already exists in the `venv` directory. Activate it before running anything:

```bash
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with your credentials:

```env
RIOT_API_KEY=your_key_here
LOL_DRAFT_DB_PATH=league_data_v2.db
LOL_DRAFT_COLLECTOR=joseph
```

Scraper behavior is configured in `config/scraper.json`:

```json
{
  "queue_id": 420,
  "start_time": 1772323200,
  "rank_snapshots": {
    "enabled": false,
    "ttl_seconds": 21600
  }
}
```

## Training

```bash
python -m ml.trainer.train
python -m ml.trainer.train --finetune-epochs 80
python -m ml.trainer.train --batch-size 2048
python -m ml.trainer.train --dropout 0.40
```

The default `python -m ml.trainer.train` path now uses the unified single-phase model.
That model trains from scratch on ordered drafts plus recent player-history features.
Older pretraining and sequence experiments still exist in the repo for reference, but they are no longer the main path.

To train from a specific DB:

```bash
LOL_DRAFT_DB_PATH=league_data_v2_merged.db python -m ml.trainer.train
```

On PowerShell:

```powershell
$env:LOL_DRAFT_DB_PATH="league_data_v2_merged.db"
python -m ml.trainer.train
```

## Data Collection

```bash
python -m ml.data.data_api_sqlite
python -m ml.data.merge_match_dbs --target league_data_v2_merged.db league_data_v2_a.db league_data_v2_b.db
```

## Server

```bash
uvicorn website.server:app --reload
```

## Analysis Scripts

```bash
python ml/data/tests/api_test.py
python ml/data/tests/check_duplicates.py
python ml/predictor/tests/check_embeddings.py
python ml/predictor/tests/benchmark_model.py
```

## Tests

```bash
pytest
```

## Multi-Operator Collection

Each operator runs the scraper on their own machine with their own API key and writes to a separate local DB:

```env
# Operator 1
RIOT_API_KEY=key_for_operator_1
LOL_DRAFT_DB_PATH=league_data_v2_operator1.db
LOL_DRAFT_COLLECTOR=operator1
```

Merge operator DBs into a single training DB periodically:

```bash
python -m ml.data.merge_match_dbs --target league_data_v2_merged.db league_data_v2_operator1.db league_data_v2_operator2.db
```

The merge logic prefers richer rows — rows with ordered and raw payloads beat thinner ones.

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
  server.py       # FastAPI BFF server
  app.js          # browser-side interactions and API calls
  index.html      # page structure
  style.css       # layout and visual styling
  tests/
```

## Documentation

- `ROADMAP.MD`: repo-wide planning source of truth
- `website/todo.txt`: short execution queue for the website component
- `ml/`: see `CLAUDE.md` for training ownership policy

If there is a conflict between older assumptions and current docs, prefer the roadmap and the current code.
