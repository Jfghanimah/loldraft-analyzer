"""
LoL Draft Analyzer — API server

Serves the frontend (website/) and exposes:
  GET  /api/champions                                      -> champion name list for autocomplete
  GET  /api/live-game?name=<name>&tag=<tag>&region=<na1>  -> current game draft (STUB)
  POST /api/predict   body: {"champions": [10 names], "blue_side": 1, "players": [10 puuids?]}  -> win probability

Run with:
  uvicorn website.server:app --reload
"""

import json
import math
import os
import random
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ml.data.match_storage import connect_sqlite
from ml.trainer.feature_pipeline import build_dense_features_for_prediction
from ml.runtime_config import load_runtime_env

load_runtime_env()

app = FastAPI(title="LoL Draft Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHAMP_LIST_PATH = Path("ml/save_data/champion_list.json")
MODEL_PATH      = Path("ml/save_data/best_unified_win_predictor_cls_global_l4_acc.pth")
LOG_PATH        = Path("ml/save_data/latest_train_log_cls_global_l4_acc.txt")

# Population prior defaults: (champ_role_win_rate, champ_role_frequency)
# Used to zero-fill slots when population priors are absent at inference time.
_PRIOR_DEFAULTS = (0.5, 0.0)

# ---------------------------------------------------------------------------
# Lazy model + champion list loading
# ---------------------------------------------------------------------------

_model         = None
_champion_list = None
_model_loaded  = False

# ---------------------------------------------------------------------------
# Parquet match file cache (for fast demo-game random selection)
# ---------------------------------------------------------------------------

_parquet_match_files: list = []
_parquet_files_loaded = False

_PARQUET_COLS = ["match_id", "blue_win"] + [f"champion_{i}" for i in range(10)]

def _load_parquet_files():
    global _parquet_match_files, _parquet_files_loaded
    if _parquet_files_loaded:
        return
    from ml.runtime_config import get_compact_dataset_dir
    matches_dir = Path(get_compact_dataset_dir()) / "matches"
    if matches_dir.exists():
        _parquet_match_files = list(matches_dir.glob("**/*.parquet"))
    print(f"[server] Parquet match files indexed: {len(_parquet_match_files):,}")
    _parquet_files_loaded = True


def _parse_model_config_from_log(log_path):
    """Parse model hyperparameters and data layout from a training log file."""
    patterns = {
        "model":   re.compile(
            r"Model: (?:architecture=(?P<architecture>\w+), )?dim=(?P<dim>\d+), heads=(?P<heads>\d+), "
            r"layers=(?P<layers>\d+), ff=(?P<ff>\d+), dropout=(?P<dropout>\d+(?:\.\d+)?)"
        ),
        "dense":   re.compile(r"Dense feature dim: (?P<dim>\d+)"),
        "layout":  re.compile(r"Dense layout: player_features=(?P<player>\d+), global_features=(?P<global>\d+)"),
        "regions": re.compile(r"Regions: (?P<n>\d+)"),
    }
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()

    config = {}
    remaining = dict(patterns)
    for line in reversed(lines):
        for key, pattern in list(remaining.items()):
            m = pattern.search(line)
            if not m:
                continue
            if key == "model":
                config["architecture"]    = m.group("architecture") or "flat"
                config["embedding_dim"]   = int(m.group("dim"))
                config["nhead"]           = int(m.group("heads"))
                config["num_layers"]      = int(m.group("layers"))
                config["dim_feedforward"] = int(m.group("ff"))
                config["dropout"]         = float(m.group("dropout"))
            elif key == "dense":
                config["dense_feature_dim"] = int(m.group("dim"))
            elif key == "layout":
                config["num_player_features"] = int(m.group("player"))
                config["num_global_features"]  = int(m.group("global"))
            elif key == "regions":
                config["num_regions"] = int(m.group("n"))
            del remaining[key]
        if not remaining:
            break
    return config


def _load_resources():
    global _model, _champion_list, _model_loaded
    if _model_loaded:
        return

    if not CHAMP_LIST_PATH.exists():
        _model_loaded = True
        return

    with open(CHAMP_LIST_PATH, encoding="utf-8") as f:
        _champion_list = json.load(f)

    from ml.predictor.unified_model import UnifiedWinPredictorModel

    num_champions = len(_champion_list)

    if MODEL_PATH.exists():
        try:
            config = _parse_model_config_from_log(LOG_PATH)
        except Exception as exc:
            print(f"[server] Could not parse model config from {LOG_PATH}: {exc}")
            _model = None
            _model_loaded = True
            return

        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        _model = UnifiedWinPredictorModel(
            num_champions=num_champions,
            dense_feature_dim=config["dense_feature_dim"],
            num_player_features=config["num_player_features"],
            num_global_features=config["num_global_features"],
            num_regions=config.get("num_regions", 1),
            embedding_dim=config["embedding_dim"],
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            num_layers=config["num_layers"],
            dropout=0.0,
            architecture=config["architecture"],
        )
        _model.load_state_dict(checkpoint)
        _model.eval()
        print(
            f"[server] Unified model loaded from {MODEL_PATH} "
            f"({num_champions} champions, arch={config['architecture']}, "
            f"emb={config['embedding_dim']}, layers={config['num_layers']}, "
            f"dense_dim={config['dense_feature_dim']})"
        )
    else:
        print(f"[server] WARNING: {MODEL_PATH} not found — /api/predict will return 503.")
        _model = None

    _model_loaded = True


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup():
    _load_resources()
    _load_parquet_files()


@app.get("/api/champions")
def get_champions():
    """Returns sorted champion name list for frontend autocomplete."""
    _load_resources()
    if not _champion_list:
        return {"champions": []}
    return {"champions": sorted(_champion_list.keys())}


@app.get("/api/live-game")
def get_live_game(name: str, tag: str, region: str = "na1"):
    """
    Looks up a player by Riot ID and returns their current live game draft.

    TODO — implement with riotwatcher:

        REGION_MAP = {
            "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
            "euw1": "europe", "eun1": "europe", "tr1": "europe", "ru": "europe",
            "kr": "asia", "jp1": "asia",
        }
        routing = REGION_MAP.get(region, "americas")

        riot_watcher = RiotWatcher(os.getenv("RIOT_API_KEY"))
        lol_watcher  = LolWatcher(os.getenv("RIOT_API_KEY"))

        account   = riot_watcher.account.by_riot_id(routing, name, tag)
        summoner  = lol_watcher.summoner.by_puuid(region, account["puuid"])
        live_game = lol_watcher.spectator.by_summoner(region, summoner["id"])

        # live_game["participants"] has 10 entries sorted by teamId then position
        # Each has: championName, teamId (100=blue, 200=red), summonerName, etc.

    Expected response shape (already matches what app.js expects):
    {
        "in_game": true,
        "blue_team": ["Jinx", "Thresh", "Ahri", "Vi", "Garen"],   // strict Top→Sup order
        "red_team":  ["Caitlyn", "Lulu", "Zed", "Jarvan IV", "Darius"],
        "blue_puuids": ["..."],
        "red_puuids": ["..."],
        "player_team": "blue"   // which team the searched summoner is on
    }

    NOTE: Spectator API does not guarantee role order — you may need to infer or sort by
    timeline position. A reasonable heuristic is position field: TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY.
    """
    raise HTTPException(
        status_code=501,
        detail="Live game lookup not yet implemented. See TODO in server.py:get_live_game().",
    )


@app.get("/api/demo-game")
def get_demo_game():
    """
    Returns a random historical match from the parquet dataset.
    Fast: picks a random file, reads only the 12 needed columns, returns one row.
    """
    import random
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise HTTPException(status_code=503, detail="pyarrow not installed.")

    _load_parquet_files()
    if not _parquet_match_files:
        raise HTTPException(status_code=503, detail="No parquet match files found.")

    for _ in range(10):
        path = random.choice(_parquet_match_files)
        try:
            df = pq.ParquetFile(path).read(columns=_PARQUET_COLS).to_pandas()
        except Exception:
            continue
        df = df[df["blue_win"].notna() & df["champion_0"].notna()]
        if df.empty:
            continue
        row = df.sample(1).iloc[0]
        champs = [str(row[f"champion_{i}"]) for i in range(10)]
        if all(champs):
            return {
                "in_game":  True,
                "blue_team": champs[:5],
                "red_team":  champs[5:],
                "blue_bans": [],
                "red_bans":  [],
                "blue_win":  bool(row["blue_win"]),
                "match_id":  str(row["match_id"]),
            }

    raise HTTPException(status_code=404, detail="Could not sample a valid match.")


class PredictRequest(BaseModel):
    # 10 champion names in strict role order:
    # [blue_top, blue_jgl, blue_mid, blue_bot, blue_sup,
    #   red_top,  red_jgl,  red_mid,  red_bot,  red_sup]
    champions: list[str]
    blue_side: Optional[float] = 1.0  # 1.0 = perspective is blue side, 0.0 = red side
    players: Optional[list[Optional[str]]] = None


@app.post("/api/predict")
def predict(req: PredictRequest):
    """Runs win prediction on a fully specified 10-champion draft."""
    _load_resources()

    if len(req.champions) != 10:
        raise HTTPException(status_code=400, detail="Exactly 10 champion names required in role order.")
    if req.players is not None and len(req.players) != 10:
        raise HTTPException(status_code=400, detail="If provided, players must contain exactly 10 aligned puuids.")

    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model available. Run 'python -m ml.trainer.train' first.",
        )

    unknown = [c for c in req.champions if c not in _champion_list]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown champions: {unknown}")

    champ_ids = [_champion_list[c] for c in req.champions]

    champion_ids = torch.tensor([champ_ids],               dtype=torch.long)
    role_ids     = torch.tensor([[0,1,2,3,4, 0,1,2,3,4]], dtype=torch.long)
    team_ids     = torch.tensor([[0,0,0,0,0, 1,1,1,1,1]], dtype=torch.long)

    dense_features = None
    has_players = req.players is not None and any(p is not None for p in req.players)
    if _model.dense_feature_dim > 0 and has_players:
        from ml.runtime_config import get_db_path

        db_path = Path(get_db_path())
        try:
            conn = connect_sqlite(db_path, read_only=True) if db_path.exists() else sqlite3.connect(":memory:")
            try:
                dense_values = build_dense_features_for_prediction(
                    conn,
                    req.champions,
                    players=req.players,
                )
            finally:
                conn.close()

            # The model may have been trained with population prior features
            # (champ_role_win_rate, champ_role_frequency) interleaved after each
            # player's base feature block. Pad with defaults when they're absent.
            base_player_dim = len(dense_values) - _model.num_global_features
            pipeline_per_slot = base_player_dim // 10
            model_per_slot = _model.num_player_features
            if pipeline_per_slot < model_per_slot:
                prior_dim = model_per_slot - pipeline_per_slot
                padded = []
                for i in range(10):
                    padded.extend(dense_values[i * pipeline_per_slot : (i + 1) * pipeline_per_slot])
                    padded.extend(_PRIOR_DEFAULTS[:prior_dim])
                padded.extend(dense_values[base_player_dim:])  # global features
                dense_values = padded

            dense_features = torch.tensor([dense_values[:_model.dense_feature_dim]], dtype=torch.float32)
        except Exception as exc:
            print(f"[server] WARNING: could not build player features ({exc}); using zero-padded features.")

    # Capture transformer output to compute per-lane scores
    _captured: dict = {}
    def _hook(module, inp, out):
        _captured['x'] = out
    handle = _model.transformer.register_forward_hook(_hook)

    try:
        with torch.no_grad():
            logit = _model(champion_ids, role_ids, team_ids, dense_features=dense_features)
            blue_prob = float(torch.sigmoid(logit).item())
    finally:
        handle.remove()

    # Per-lane scores: for cls_global the transformer processes
    # [CLS, slot_0..slot_9, global_ctx] so blue slots are indices 1..5,
    # red slots are 6..10. Project each lane diff through trunk+head
    # as a proxy for per-lane advantage.
    lane_scores = None
    if 'x' in _captured and getattr(_model, 'architecture', None) == 'cls_global':
        x = _captured['x']  # [1, seq_len, d]
        if x.shape[1] >= 11:
            blue_slots = x[0, 1:6, :]   # [5, d]
            red_slots  = x[0, 6:11, :]  # [5, d]
            lane_diffs = blue_slots - red_slots  # [5, d]
            with torch.no_grad():
                raw = [
                    float(_model.win_head(_model.shared_trunk(lane_diffs[i:i+1])).item())
                    for i in range(5)
                ]
            max_abs = max(abs(s) for s in raw) or 1.0
            lane_scores = [round(math.tanh(s / max_abs), 3) for s in raw]

    red_prob = 1.0 - blue_prob
    diff = abs(blue_prob - 0.5)
    confidence = "high" if diff > 0.1 else "medium" if diff > 0.05 else "low"

    result = {
        "blue_win_probability": round(blue_prob, 4),
        "red_win_probability":  round(red_prob,  4),
        "confidence":           confidence,
    }
    if lane_scores is not None:
        result["lane_scores"] = lane_scores
    return result


# ---------------------------------------------------------------------------
# Serve frontend — must be last so API routes take priority
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory="website", html=True), name="frontend")
