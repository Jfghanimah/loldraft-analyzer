# Website Architecture

## Overview

The LoL Draft Analyzer website is a FastAPI + vanilla JavaScript frontend for predicting League of Legends draft outcomes using a trained transformer model. It consists of:

1. **FastAPI Server** (`website/server.py`) — API endpoints + static file serving
2. **Analyzer Page** (`website/index.html` + `website/app.js`) — live game lookup & manual draft building
3. **Guess the Winner Page** (`website/guess.html`) — minigame where users predict historical match outcomes
4. **Styling** (`website/style.css`) — unified design system for both pages

## Running the Server

```bash
# Activate venv first
source venv/Scripts/activate  # or venv\Scripts\activate on Windows

# Start server on localhost:8000
uvicorn website.server:app --reload
```

Visit `http://localhost:8000` in a browser. The server:
- Serves static files from `website/` directory
- Exposes `/api/*` endpoints for champion lookups and win prediction
- Returns HTTP 503 with helpful message if the trained model is missing

## API Endpoints

### GET /api/champions

Returns sorted list of champion names for autocomplete.

**Response:**
```json
{
  "champions": ["Aatrox", "Ahri", "Akali", ...]
}
```

### GET /api/live-game?name=<name>&tag=<tag>&region=<region>

**Not yet implemented** (returns 501). Intended to fetch a player's current live game draft via Riot API. When implemented, will return:

```json
{
  "in_game": true,
  "blue_team": ["Jinx", "Thresh", "Ahri", "Vi", "Garen"],
  "red_team": ["Caitlyn", "Lulu", "Zed", "Jarvan IV", "Darius"],
  "blue_puuids": ["..."],
  "red_puuids": ["..."],
  "player_team": "blue"
}
```

### GET /api/demo-game

Returns a random historical match from the parquet dataset.

**Response:**
```json
{
  "in_game": true,
  "blue_team": ["Ahri", "Lee Sin", "Orianna", "Jinx", "Thresh"],
  "red_team": ["Zed", "Jarvan IV", "Viktor", "Caitlyn", "Lulu"],
  "blue_bans": [],
  "red_bans": [],
  "blue_win": true,
  "match_id": "NA1_abc123def456"
}
```

**Performance:** ~260ms per call (was 3+ seconds before parquet optimization)

**Implementation Details:**
- Randomly selects from 50,731 parquet files in `ml/data/compact_dataset/matches/`
- Reads only 12 columns: `match_id`, `blue_win`, `champion_0` through `champion_9`
- Retries up to 10 times if a file doesn't have valid data
- Falls back to 503 if no valid match found after retries

### POST /api/predict

Runs win prediction on a fully specified 10-champion draft.

**Request:**
```json
{
  "champions": [
    "Ahri", "Lee Sin", "Orianna", "Jinx", "Thresh",
    "Zed", "Jarvan IV", "Viktor", "Caitlyn", "Lulu"
  ],
  "blue_side": 1.0,
  "players": [null, null, null, null, null, null, null, null, null, null]
}
```

**Response:**
```json
{
  "blue_win_probability": 0.6234,
  "red_win_probability": 0.3766,
  "confidence": "high",
  "lane_scores": [0.124, -0.087, 0.256, 0.089, -0.045]
}
```

**Key Details:**
- `champions`: Strict role order — `[blue_top, blue_jgl, blue_mid, blue_bot, blue_sup, red_top, red_jgl, red_mid, red_bot, red_sup]`
- `blue_side`: 1.0 = prediction from blue perspective, 0.0 = from red perspective
- `players`: Optional puuids for fetching historical win rates; can be null
- `lane_scores`: Per-lane advantage estimates [Top, Jgl, Mid, Bot, Sup] (only present for cls_global architecture)
- `confidence`: "high" (>10% diff), "medium" (>5% diff), or "low"

## Model Loading

The server lazily loads the trained model on first request. It:

1. **Parses training hyperparameters** from `ml/save_data/latest_train_log_cls_global_l4_acc.txt` using regex patterns:
   - Model architecture, embedding dim, attention heads, layers, feedforward dim, dropout
   - Dense feature dim, player/global feature counts, region count

2. **Reconstructs the model** using `UnifiedWinPredictorModel` with parsed config

3. **Loads checkpoint** from `ml/save_data/best_unified_win_predictor_cls_global_l4_acc.pth`

4. **Sets to eval mode** and pins to CPU

If model or log files are missing, the server continues running but returns 503 on `/api/predict` with a helpful message.

## Pages

### Analyzer (`/` → `website/index.html`)

Main page for testing draft predictions. Features:

- **Live Game Lookup** — search for a player by Riot ID (name#tag) and region (not yet implemented)
- **Demo Button** — "Feeling Lucky?" loads a random historical match
- **Draft Board** — 5×5 grid of empty slots (blue left, red right)
- **Champion Picker Modal** — searchable champion select with filters (role tags when implemented)
- **Prediction Panel** — shows blue/red win probabilities when draft is complete
- **Lane Matchups** — per-role comparison of strengths/weaknesses (appears after analysis)
- **Draft Strengths** — composition breakdown (appears after analysis)

**Key JavaScript Objects** (`website/app.js`):
- `draft` — `{blue: [5 champs], red: [5 champs], bans: {blue: [...], red: [...]}}`
- `champions` — map of champion name → metadata (role tags, id, etc.)
- UI state — which slot is being edited, modal visibility, etc.

### Guess the Winner (`/guess.html`)

Interactive minigame. Features:

- **Session Score Bar** — shows correct/total guesses and percentage (green ≥50%, red <50%)
- **Draft Card** — displays blue/red teams with 48×48px champion icons, roles, and names
- **Guess Buttons** — "Blue Wins" / "Red Wins" (disabled during result reveal)
- **Result Banner** — reveals actual outcome ("✓ Correct!" or "✗ Wrong!") with winning side
- **Next Game Button** — loads next match and re-enables guess buttons

**Session State** (persists during page session, resets on refresh):
```javascript
let session = { correct: 0, total: 0 };
let currentGame = null;  // Current match data from /api/demo-game
```

**Game Loop:**
1. `loadGame()` fetches `/api/demo-game`, renders draft in `guess-draft` div
2. User clicks "Blue Wins" or "Red Wins" → `makeGuess(guessedBlue)`
3. `makeGuess()` compares user guess to `currentGame.blue_win`, updates score, calls `showResult()`
4. `showResult()` hides draft, shows result banner and final teams (colored blue/red)
5. User clicks "Next Game →" → calls `loadGame()` again, buttons re-enabled

**Performance Optimization:**
- Parquet demo-game endpoint loads in ~260ms (no visible delay)
- No loading spinner displayed (data arrives too fast)
- If `/api/demo-game` fails, error message appears in draft panel

## CSS Architecture

Shared design system in `website/style.css`:

**Key Classes:**
- `.card` — white box with shadow, padding, rounded corners
- `.btn-primary` / `.btn-secondary` / `.btn-ghost` — button styles with hover/active states
- `.site-nav` — top navigation with active underline
- `.guess-main` — centered container, max-width 860px
- `.guess-team-card` — blue/red side-by-side team display
  - `.winner` — gold background glow
  - `.loser` — dimmed (opacity ~38%)
- `.guess-champ-row` — flex row: [48px icon] [role label] [champion name]
- `.result-banner` — "✓ Correct!" / "✗ Wrong!" reveal with side indicator

**Important Fix:**
```css
[hidden] { display: none !important; }
```
Ensures the HTML `hidden` attribute (set by JavaScript) properly hides elements. Without `!important`, inline display styles could override it.

## Data Pipeline

### Champion List (`ml/save_data/champion_list.json`)

JSON map: `{ "Champion Name": champion_id, ... }`

Used by:
- Server to validate incoming champion names
- Server to build champion_id tensors for model
- Frontend to map champion names to DDragon icon URLs (via `DDRAGON_SPECIAL` mapping)

### Parquet Dataset

Location: `ml/data/compact_dataset/matches/*.parquet`

50,731 files, each containing historical matches with:
- `match_id` — unique identifier
- `blue_win` — boolean outcome
- `champion_0` through `champion_9` — champion names in role order

Used by `/api/demo-game` for fast random match sampling.

### Model Checkpoint

**File:** `ml/save_data/best_unified_win_predictor_cls_global_l4_acc.pth`

**Accuracy:** 57.84% (on holdout test set)

**Architecture:** `cls_global`
- Transformer-based encoder
- Champion embeddings (learned)
- Role & team embeddings (learned)
- Optional dense player features (historical win rates, etc.)
- Global context token
- Output head predicting blue win probability via sigmoid

**Config Parsing:** Server extracts hyperparams from `latest_train_log_cls_global_l4_acc.txt`:
```
Model: architecture=cls_global, dim=128, heads=4, layers=2, ff=256, dropout=0.3
Dense feature dim: 14
Dense layout: player_features=14, global_features=2
Regions: 1
```

## Common Issues & Fixes

### Model fails to load
- **Symptom:** Server starts but `/api/predict` returns 503
- **Cause:** Missing `best_unified_win_predictor_cls_global_l4_acc.pth` or `latest_train_log_cls_global_l4_acc.txt`
- **Fix:** Run training: `python -m ml.trainer.train`

### Demo game takes 3+ seconds
- **Symptom:** "Feeling Lucky?" button is slow
- **Cause:** Using SQLite with `ORDER BY RANDOM()` on huge table (full scan)
- **Fix:** Already applied—uses parquet file sampling instead

### Feature lookup takes 800+ seconds
- **Symptom:** `/api/predict` hangs when building dense features
- **Cause:** SQL query `WHERE (? IS NULL OR queue_id = ?)` prevents index usage
- **Fix:** Already applied—split into two separate queries in `ml.trainer.feature_pipeline`

### Guess buttons stay disabled after first guess
- **Symptom:** User can make one guess, then buttons are grayed out
- **Cause:** `makeGuess()` disables buttons, but `renderDraft()` never re-enables them
- **Fix:** Already applied—`renderDraft()` now resets `disabled = false` on both buttons

### Champion icons not loading or blurry
- **Symptom:** DDragon images fail to load or appear pixelated
- **Cause:** Stale browser cache or wrong DDragon version
- **Fix:** Hard refresh (Shift+F5 or Ctrl+Shift+R) to clear cache; server fetches latest DDragon version on startup

## Testing

**Manual Testing:**
1. Start server: `uvicorn website.server:app --reload`
2. Open `http://localhost:8000` in browser
3. On Analyzer page:
   - Click "Feeling Lucky?" to load a random match
   - Select champions from the grid
   - Click "Analyze" to get prediction
   - Check lane scores appear below
4. On Guess the Winner page:
   - Click "Next Game" to load match
   - Make a guess
   - Verify result appears and buttons reset for next game
   - Refresh page and verify score resets

**API Testing:**
```bash
# Get champion list
curl http://localhost:8000/api/champions

# Get demo game
curl http://localhost:8000/api/demo-game

# Predict on a draft
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "champions": ["Ahri", "Lee Sin", "Orianna", "Jinx", "Thresh", "Zed", "Jarvan IV", "Viktor", "Caitlyn", "Lulu"],
    "blue_side": 1.0
  }'
```

## Future Work

- **Live Game Lookup:** Implement `/api/live-game` using `riotwatcher` library
- **Per-Lane Scores:** Frontend visualization for lane_scores array (currently only returned by API)
- **Share Links:** URL encoding of draft state for sharing predictions
- **Draft Strengths Panel:** Composition analysis (scaling, CC, etc.)
- **Bans Display:** Include ban data in parquet and show in guess page
