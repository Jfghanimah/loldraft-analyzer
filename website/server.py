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
MODEL_PATH      = Path("ml/save_data/best_win_predictor.pth")

# ---------------------------------------------------------------------------
# Lazy model + champion list loading
# ---------------------------------------------------------------------------

_model         = None
_champion_list = None
_model_loaded  = False

# ---------------------------------------------------------------------------
# DDragon champion ID → name map (used for ban resolution in demo endpoint)
# ---------------------------------------------------------------------------

_ddragon_id_to_name: dict[int, str] = {}

def _get_ddragon_id_map() -> dict[int, str]:
    """Fetch Riot champion-ID → display-name from DDragon (lazy, cached)."""
    global _ddragon_id_to_name
    if _ddragon_id_to_name:
        return _ddragon_id_to_name
    try:
        import urllib.request
        with urllib.request.urlopen(
            "https://ddragon.leagueoflegends.com/api/versions.json", timeout=5
        ) as r:
            version = json.loads(r.read())[0]
        url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        for entry in data["data"].values():
            _ddragon_id_to_name[int(entry["key"])] = entry["name"]
        print(f"[server] DDragon ID map loaded: {len(_ddragon_id_to_name)} champions (patch {version})")
    except Exception as exc:
        print(f"[server] Could not fetch DDragon champion map: {exc}")
    return _ddragon_id_to_name


def _load_resources():
    global _model, _champion_list, _model_loaded
    if _model_loaded:
        return

    if not CHAMP_LIST_PATH.exists():
        _model_loaded = True
        return

    with open(CHAMP_LIST_PATH, encoding="utf-8") as f:
        _champion_list = json.load(f)

    from ml.predictor.models_pytorch import WinPredictorModel

    num_champions = len(_champion_list)

    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        # Infer architecture from saved weights so we match whatever was trained
        embedding_dim      = checkpoint["champ_emb.weight"].shape[1]
        actual_input_dim   = checkpoint["classifier.0.weight"].shape[1]
        extra_feature_dim  = max(0, actual_input_dim - (15 * embedding_dim + 1))
        # Count transformer layers from checkpoint keys
        num_layers = sum(1 for k in checkpoint if k.startswith("transformer.layers.") and k.endswith(".norm1.weight"))

        _model = WinPredictorModel(
            num_champions=num_champions,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=0.0,
            extra_feature_dim=extra_feature_dim,
        )
        _model.load_state_dict(checkpoint)
        _model.eval()
        print(
            f"[server] Model loaded from {MODEL_PATH} "
            f"({num_champions} champions, emb={embedding_dim}, "
            f"layers={num_layers}, extra={extra_feature_dim})"
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
    Returns a random historical match from the local SQLite database, formatted
    identically to the live-game response so the frontend can exercise all the
    player-name / stat-row / lane-matchup UI without a real Riot API key.
    """
    from ml.runtime_config import get_db_path
    from ml.data.match_format import try_build_ordered_participant_record

    db_path = Path(get_db_path())
    if not db_path.exists():
        raise HTTPException(status_code=503, detail=f"Database not found at {db_path}.")

    try:
        conn = connect_sqlite(db_path, read_only=True)
        row = conn.execute(
            "SELECT match_id, ordered_match_json, raw_match_json FROM matches "
            "WHERE ordered_match_json IS NOT NULL AND raw_match_json IS NOT NULL "
            "ORDER BY RANDOM() LIMIT 1"
        ).fetchone()
        conn.close()
    except sqlite3.Error as exc:
        raise HTTPException(status_code=503, detail=f"Could not read database at {db_path}: {exc}") from exc

    if not row:
        raise HTTPException(status_code=404, detail="No fully-stored matches found in database.")

    match_id, ordered_json, raw_json = row
    ordered = json.loads(ordered_json)
    raw     = json.loads(raw_json)

    champions = ordered["champions"]          # 10 names: blue[0:5] + red[5:10]
    blue_team = champions[:5]
    red_team  = champions[5:]

    # Pull participants in the same role order using match_format logic
    ordered_participants, err = try_build_ordered_participant_record(raw["info"])

    # Resolve ban champion names (requires DDragon ID map)
    id_to_name  = _get_ddragon_id_map()
    blue_bans, red_bans = [], []
    for team in raw["info"].get("teams", []):
        resolved = [id_to_name.get(b.get("championId", -1), "") for b in team.get("bans", [])]
        if team.get("teamId") == 100:
            blue_bans = resolved
        else:
            red_bans = resolved

    if err or not ordered_participants:
        return {
            "in_game":   True,
            "blue_team": blue_team,
            "red_team":  red_team,
            "blue_bans": blue_bans,
            "red_bans":  red_bans,
            "match_id":  match_id,
        }

    blue_players, red_players = [], []
    blue_stats,   red_stats   = [], []
    blue_puuids,  red_puuids  = [], []

    for idx, p in enumerate(ordered_participants):
        game_name = p.get("riotIdGameName") or p.get("summonerName", "Unknown")
        tag       = p.get("riotIdTagline", "")
        display   = f"{game_name}#{tag}" if tag else game_name
        if idx < 5:
            blue_players.append(display)
            blue_stats.append({})
            blue_puuids.append(p.get("puuid"))
        else:
            red_players.append(display)
            red_stats.append({})
            red_puuids.append(p.get("puuid"))

    return {
        "in_game":      True,
        "blue_team":    blue_team,
        "red_team":     red_team,
        "blue_players": blue_players,
        "red_players":  red_players,
        "blue_stats":   blue_stats,
        "red_stats":    red_stats,
        "blue_puuids":  blue_puuids,
        "red_puuids":   red_puuids,
        "blue_bans":    blue_bans,
        "red_bans":     red_bans,
        "blue_win":     ordered.get("blue_win"),
        "match_id":     match_id,
    }


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

    champion_ids = torch.tensor([champ_ids],              dtype=torch.long)
    role_ids     = torch.tensor([[0,1,2,3,4, 0,1,2,3,4]], dtype=torch.long)
    team_ids     = torch.tensor([[0,0,0,0,0, 1,1,1,1,1]], dtype=torch.long)
    blue_side    = torch.tensor([[req.blue_side]],         dtype=torch.float32)
    dense_values = None
    if _model.extra_feature_dim:
        from ml.runtime_config import get_db_path

        db_path = Path(get_db_path())
        conn = connect_sqlite(db_path, read_only=True) if db_path.exists() else sqlite3.connect(":memory:")
        try:
            dense_values = build_dense_features_for_prediction(
                conn,
                req.champions,
                players=req.players,
            )
        finally:
            conn.close()
        if len(dense_values) != _model.extra_feature_dim:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model dense feature shape does not match the current feature pipeline. "
                    "Re-run 'python -m ml.trainer.train'."
                ),
            )
    dense_features = (
        torch.tensor([dense_values], dtype=torch.float32)
        if dense_values is not None else None
    )

    # Capture transformer output to compute per-lane scores
    _captured: dict = {}
    def _hook(module, inp, out):
        _captured['x'] = out  # [1, 10, embedding_dim]
    handle = _model.transformer.register_forward_hook(_hook)

    try:
        with torch.no_grad():
            logit = _model(champion_ids, role_ids, team_ids, blue_side, dense_features)
            blue_prob = float(torch.sigmoid(logit).item())
    finally:
        handle.remove()

    # Per-lane advantage: project each lane's blue-minus-red diff through the
    # classifier's first linear layer weights.
    # Feature layout: [flattened(10*d), lane_diff(5*d), blue_side(1), dense(extra)]
    lane_scores = None
    if 'x' in _captured:
        x = _captured['x']                          # [1, 10, d]
        lane_diffs = x[0, :5, :] - x[0, 5:, :]     # [5, d]
        d = lane_diffs.shape[1]
        lane_diff_start = 10 * d
        w = _model.classifier[0].weight.detach()     # [hidden, input_dim]
        raw = []
        for i in range(5):
            w_lane = w[:, lane_diff_start + i * d : lane_diff_start + (i + 1) * d]
            raw.append(float((w_lane @ lane_diffs[i]).sum().item()))
        max_abs = max(abs(s) for s in raw) or 1.0
        # tanh(s/max_abs): max lane → ≈0.76, not ±1.0, so uniform advantages
        # don't all collapse to "100%" on the frontend
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
