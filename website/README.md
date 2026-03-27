# Website

This folder contains the web UI and API server for the LoL Draft Analyzer product surface.

- The website is a thin product layer on top of the trained ML model.
- Live game lookup is still stubbed — see `todo.txt`.
- `website/todo.txt` is the execution queue for web-specific work.
- Repo-root `ROADMAP.MD` is the shared project roadmap.

## Project structure

- `index.html`: page structure
- `style.css`: layout and visual styling
- `app.js`: browser-side interactions and API calls
- `server.py`: FastAPI server — predictions, demo game, live game stub
- `tests/`: web test package placeholder

## Running

```bash
uvicorn website.server:app --reload
```

LAN access:

```bash
uvicorn website.server:app --reload --host 0.0.0.0
```

## API Endpoints

| Endpoint | Status | Description |
|---|---|---|
| `GET /api/champions` | Working | Sorted champion name list for autocomplete |
| `POST /api/predict` | Working | Draft win probability + lane scores |
| `GET /api/demo-game` | Working | Random historical match from DB (bans, KDA, PUUIDs) |
| `GET /api/live-game` | Stub (501) | Live champion select lookup via Spectator API |

### POST /api/predict

```json
{
  "champions": ["Jinx", "Vi", "Ahri", "Thresh", "Garen", "Caitlyn", "Jarvan IV", "Zed", "Lulu", "Darius"],
  "blue_side": 1.0,
  "players": ["puuid1", ..., "puuid10"]
}
```

- `champions`: 10 names in role order — blue Top/Jgl/Mid/Bot/Sup then red Top/Jgl/Mid/Bot/Sup
- `blue_side`: `1.0` = predicting from blue perspective, `0.0` = red
- `players`: optional 10 PUUIDs — if provided, server fetches recent match history from DB and builds 150 dense features (15 stats × 10 players) plus 2 patch features for the model

Response:

```json
{
  "blue_win_probability": 0.6234,
  "red_win_probability": 0.3766,
  "confidence": "high",
  "lane_scores": [0.15, -0.08, 0.22, -0.05, 0.12]
}
```

`lane_scores` is `[top, jgl, mid, bot, sup]` from blue's perspective (positive = blue advantage).

## ML Integration

`server.py` pulls from the ML layer at startup and on first request:

```
ml/save_data/champion_list.json         → champion name → ID mapping (172 champs)
ml/save_data/best_win_predictor.pth     → model weights (auto-detects architecture)
```

Runtime imports:

```
ml.predictor.models_pytorch             → WinPredictorModel
ml.trainer.feature_pipeline             → build_dense_features_for_prediction
ml.data.match_storage                   → connect_sqlite (player history lookup)
ml.data.match_format                    → try_build_ordered_participant_record
ml.runtime_config                       → load_runtime_env, get_db_path
```

Model loading is lazy (first `/api/predict` call) and auto-detects `embedding_dim`, `num_layers`, and `extra_feature_dim` from checkpoint weight shapes — no manual config needed after retraining.

Dense features require `participant_history` to be populated in the DB. Run:

```bash
python -m ml.data.backfill_participant_history --limit 1000
```

## New here?

1. Repo-root `ROADMAP.MD` — project-wide goals
2. `website/todo.txt` — current web execution queue
3. `website/server.py` — API endpoints and model loading
4. `website/app.js` — frontend logic
