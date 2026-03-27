# LoL Draft Analyzer

LoL Draft Analyzer is a PyTorch project for predicting League of Legends match outcomes from champion draft compositions. The codebase includes a SQLite-backed collection pipeline and a two-stage training setup built around champion-embedding pretraining plus win prediction.

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
python -m ml.trainer.train --skip-pretrain
python -m ml.trainer.train --pretrain-only

python -m ml.trainer.pretrain_embeddings
python -m ml.trainer.train_win_predictor
```

Phase 1 still learns champion embeddings from ordered drafts.
Phase 2 now uses the single unified richer pre-match feature pipeline by default instead of maintaining a separate "basic vs rich" finetuning split.

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
