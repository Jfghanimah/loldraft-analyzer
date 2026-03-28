import json
import math
import os
import sqlite3
import time
from collections import deque

import torch

from ml.data.match_format import ROLE_ORDER
from ml.data.match_storage import extract_participant_history_rows, get_match_columns
from ml.features.recent_history import QUEUE_ID_SOLO, parse_patch
from ml.runtime_config import get_db_path, load_runtime_env

CHAMPION_LIST_PATH = "ml/save_data/champion_list.json"
REGION_LIST_PATH = "ml/save_data/region_list.json"
SEQUENCE_CACHE_PATH = "ml/save_data/sequence_cache.pt"
DEFAULT_DB_PATH = get_db_path()
HISTORY_LENGTH = 10
PROGRESS_UPDATE_EVERY = 1000
ROLE_TO_ID = {role: index for index, role in enumerate(ROLE_ORDER)}
TEAM_TO_ID = {100: 0, 200: 1}

load_runtime_env()


def _load_json_mapping(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _sync_mapping(path, observed_values):
    mapping = _load_json_mapping(path)
    next_id = max(mapping.values(), default=-1) + 1
    missing = [value for value in observed_values if value not in mapping]
    if missing:
        for value in missing:
            mapping[value] = next_id
            next_id += 1
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ordered = dict(sorted(mapping.items(), key=lambda item: item[1]))
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(ordered, handle, indent=4)
        return ordered
    return mapping


def _sync_champion_list(ordered_records, champion_path):
    observed = sorted({champ for record in ordered_records for champ in record["champions"]})
    return _sync_mapping(champion_path, observed)


def _sync_region_list(regions, region_path):
    observed = sorted({region for region in regions if region})
    return _sync_mapping(region_path, observed)


def _ensure_mapping_values(mapping, path, values):
    missing = sorted({value for value in values if value not in mapping})
    if not missing:
        return mapping
    next_id = max(mapping.values(), default=-1) + 1
    for value in missing:
        mapping[value] = next_id
        next_id += 1
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ordered = dict(sorted(mapping.items(), key=lambda item: item[1]))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(ordered, handle, indent=4)
    return ordered


def _build_sequence_cache_fingerprint(db_path, queue_id, history_length):
    stat = os.stat(db_path)
    return {
        "db_path": os.path.abspath(db_path),
        "db_size": stat.st_size,
        "db_mtime_ns": stat.st_mtime_ns,
        "queue_id": queue_id,
        "history_length": history_length,
    }


def _load_sequence_cache(cache_path, fingerprint):
    if not os.path.exists(cache_path):
        return None
    cached = torch.load(cache_path, map_location="cpu")
    if cached.get("fingerprint") != fingerprint:
        return None
    return cached


def _save_sequence_cache(cache_path, fingerprint, tensors, metadata):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(
        {
            "fingerprint": fingerprint,
            "tensors": tensors,
            "metadata": metadata,
        },
        cache_path,
    )


def _load_training_matches(conn, queue_id):
    columns = get_match_columns(conn)
    if "ordered_match_json" not in columns:
        raise ValueError("Database does not contain ordered_match_json. Sequence training requires ordered matches.")

    blue_first_blood = "blue_first_blood" if "blue_first_blood" in columns else "NULL"
    blue_first_tower = "blue_first_tower" if "blue_first_tower" in columns else "NULL"
    blue_dragon_share = "blue_dragon_share" if "blue_dragon_share" in columns else "NULL"
    blue_gold_share = "blue_gold_share" if "blue_gold_share" in columns else "NULL"

    return conn.execute(
        f"""
        SELECT match_id, region, game_creation,
               {blue_first_blood} AS blue_first_blood,
               {blue_first_tower} AS blue_first_tower,
               {blue_dragon_share} AS blue_dragon_share,
               {blue_gold_share} AS blue_gold_share
        FROM matches
        WHERE ordered_match_json IS NOT NULL
          AND (? IS NULL OR queue_id = ?)
        ORDER BY COALESCE(game_creation, 0), match_id
        """,
        (queue_id, queue_id),
    ).fetchall()


def _fetch_raw_match(conn, match_id):
    try:
        row = conn.execute(
            "SELECT raw_match_json FROM matches WHERE match_id = ?",
            (match_id,),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if not row or not row[0]:
        return None
    return json.loads(row[0])


def _fetch_raw_participant_rows(conn, match_id):
    raw_match = _fetch_raw_match(conn, match_id)
    return extract_participant_history_rows(raw_match) if raw_match else []


def _load_participant_rows_by_match(conn, queue_id):
    try:
        rows = conn.execute(
            """
            SELECT match_id, puuid, champion_name, role, win, kills, deaths, assists,
                   vision_score, damage_to_champions, healing, gold_earned, cs, game_creation, game_version, team_id
            FROM participant_history
            WHERE (? IS NULL OR queue_id = ?)
            ORDER BY COALESCE(game_creation, 0), match_id, team_id,
                     CASE role
                         WHEN 'TOP' THEN 0
                         WHEN 'JUNGLE' THEN 1
                         WHEN 'MIDDLE' THEN 2
                         WHEN 'BOTTOM' THEN 3
                         WHEN 'UTILITY' THEN 4
                         ELSE 99
                     END
            """,
            (queue_id, queue_id),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}

    rows_by_match = {}
    for row in rows:
        rows_by_match.setdefault(row[0], []).append(
            {
                "puuid": row[1],
                "champion_name": row[2],
                "role": row[3],
                "win": int(row[4]),
                "kills": int(row[5]),
                "deaths": int(row[6]),
                "assists": int(row[7]),
                "vision_score": float(row[8]),
                "damage_to_champions": float(row[9]),
                "healing": float(row[10]),
                "gold_earned": float(row[11] or 0.0),
                "cs": float(row[12] or 0.0),
                "game_creation": int(row[13] or 0),
                "game_version": row[14] or "",
                "team_id": int(row[15]),
            }
        )
    return rows_by_match


def _fetch_recent_sequence_rows(conn, puuid, current_game_creation, queue_id, limit):
    try:
        rows = conn.execute(
            """
            SELECT champion_name, role, win, kills, deaths, assists, vision_score,
                   damage_to_champions, healing, gold_earned, cs, game_creation, game_version
            FROM participant_history
            WHERE puuid = ?
              AND queue_id = ?
              AND game_creation < ?
            ORDER BY game_creation DESC, match_id DESC
            LIMIT ?
            """,
            (puuid, queue_id, current_game_creation, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {
            "champion_name": row[0],
            "role": row[1],
            "win": int(row[2]),
            "kills": int(row[3]),
            "deaths": int(row[4]),
            "assists": int(row[5]),
            "vision_score": float(row[6]),
            "damage_to_champions": float(row[7]),
            "healing": float(row[8]),
            "gold_earned": float(row[9] or 0.0),
            "cs": float(row[10] or 0.0),
            "game_creation": int(row[11] or 0),
            "game_version": row[12] or "",
        }
        for row in rows
    ]


def _normalize_kda(row):
    value = (float(row["kills"]) + float(row["assists"])) / max(1.0, float(row["deaths"]))
    return min(value, 10.0) / 10.0


def _normalize_avg(value, scale):
    return min(float(value), scale) / scale if scale else 0.0


def _normalize_age(current_game_creation, prior_game_creation):
    if not current_game_creation or not prior_game_creation:
        return 1.0
    delta_hours = max(0.0, float(current_game_creation - prior_game_creation) / 3_600_000.0)
    return min(math.log1p(delta_hours) / 6.0, 1.0)


def _extract_auxiliary_targets(match):
    info = match.get("info", {})
    teams = {team.get("teamId"): team for team in info.get("teams", [])}
    blue_team = teams.get(100, {})
    red_team = teams.get(200, {})
    blue_objectives = blue_team.get("objectives", {})
    red_objectives = red_team.get("objectives", {})

    def _objective(team_objectives, objective_name, key, default):
        return team_objectives.get(objective_name, {}).get(key, default)

    blue_dragons = float(_objective(blue_objectives, "dragon", "kills", 0) or 0)
    red_dragons = float(_objective(red_objectives, "dragon", "kills", 0) or 0)
    total_dragons = blue_dragons + red_dragons
    dragon_share = blue_dragons / total_dragons if total_dragons > 0 else 0.5

    blue_gold = 0.0
    red_gold = 0.0
    for participant in info.get("participants", []):
        team_id = int(participant.get("teamId", 0) or 0)
        gold = float(participant.get("goldEarned", 0.0) or 0.0)
        if team_id == 100:
            blue_gold += gold
        elif team_id == 200:
            red_gold += gold
    total_gold = blue_gold + red_gold
    gold_share = blue_gold / total_gold if total_gold > 0 else 0.5

    return [
        float(bool(_objective(blue_objectives, "champion", "first", False))),
        float(bool(_objective(blue_objectives, "tower", "first", False))),
        float(dragon_share),
        float(gold_share),
    ]


def _extract_auxiliary_targets_from_columns(
    blue_first_blood,
    blue_first_tower,
    blue_dragon_share,
    blue_gold_share,
):
    values = [blue_first_blood, blue_first_tower, blue_dragon_share, blue_gold_share]
    if any(value is None for value in values):
        return None
    return [
        float(blue_first_blood),
        float(blue_first_tower),
        float(blue_dragon_share),
        float(blue_gold_share),
    ]


def _print_sequence_progress(processed, total, built_rows, started_at, *, force=False):
    if total <= 0:
        return
    elapsed = time.time() - started_at
    percent = (processed / total) * 100.0
    rate = processed / elapsed if elapsed > 0 else 0.0
    eta = (total - processed) / rate if rate > 0 else 0.0
    print(
        (
            f"\r[Sequence] Building examples {processed:,}/{total:,} ({percent:5.1f}%) "
            f"| built={built_rows:,} | elapsed={elapsed:.1f}s | eta={eta:.1f}s"
        ).ljust(140),
        end="\n" if force else "",
        flush=True,
    )


def build_sequence_training_tensors(
    db_path=DEFAULT_DB_PATH,
    champion_path=CHAMPION_LIST_PATH,
    region_path=REGION_LIST_PATH,
    cache_path=SEQUENCE_CACHE_PATH,
    queue_id=QUEUE_ID_SOLO,
    history_length=HISTORY_LENGTH,
):
    if not os.path.exists(db_path):
        raise ValueError(f"Database not found: {db_path}")

    fingerprint = _build_sequence_cache_fingerprint(db_path, queue_id, history_length)
    cached = _load_sequence_cache(cache_path, fingerprint)
    if cached is not None:
        print(f"[Sequence] Loaded cached tensors from {cache_path}")
        return cached["tensors"], cached["metadata"]

    conn = sqlite3.connect(db_path)
    rows = _load_training_matches(conn, queue_id)
    participant_rows_by_match = _load_participant_rows_by_match(conn, queue_id)

    champion_list = _sync_mapping(
        champion_path,
        sorted(
            {
                row["champion_name"]
                for participant_rows in participant_rows_by_match.values()
                for row in participant_rows
                if row["champion_name"]
            }
        ),
    )
    region_list = _sync_region_list([row[1] or "" for row in rows], region_path)
    pad_champion_id = len(champion_list) + 1
    history_store = {}
    built = []
    total_rows = len(rows)
    started_at = time.time()

    for index, (
        match_id,
        region,
        game_creation,
        blue_first_blood,
        blue_first_tower,
        blue_dragon_share,
        blue_gold_share,
    ) in enumerate(rows, start=1):
        participant_rows = participant_rows_by_match.get(match_id, [])
        raw_match = None
        if len(participant_rows) != 10:
            raw_match = _fetch_raw_match(conn, match_id)
            participant_rows = extract_participant_history_rows(raw_match) if raw_match else []
        if len(participant_rows) != 10:
            continue

        champion_list = _ensure_mapping_values(
            champion_list,
            champion_path,
            [row["champion_name"] for row in participant_rows if row["champion_name"]],
        )

        current_champion_ids = [champion_list[row["champion_name"]] for row in participant_rows]
        current_role_ids = [ROLE_TO_ID[row["role"]] for row in participant_rows]
        current_team_ids = [TEAM_TO_ID[row["team_id"]] for row in participant_rows]

        history_champion_ids = []
        history_role_ids = []
        history_slot_ids = []
        history_result_ids = []
        history_numeric = []
        history_mask = []

        current_game_creation = int(game_creation or 0)
        current_patch_major, current_patch_minor = parse_patch(participant_rows[0].get("game_version", ""))

        for slot_index, participant in enumerate(participant_rows):
            prior_rows = list(history_store.get(participant["puuid"], ()))[:history_length]
            slot_champs = [pad_champion_id] * history_length
            slot_roles = [0] * history_length
            slot_slots = [slot_index] * history_length
            slot_results = [0] * history_length
            slot_numeric = [[0.0] * 9 for _ in range(history_length)]
            slot_mask = [0] * history_length

            for history_index, prior in enumerate(prior_rows):
                patch_major, patch_minor = parse_patch(prior.get("game_version", ""))
                slot_champs[history_index] = champion_list.get(prior["champion_name"], pad_champion_id)
                slot_roles[history_index] = ROLE_TO_ID.get(prior["role"], 0)
                slot_results[history_index] = int(prior["win"])
                slot_numeric[history_index] = [
                    _normalize_age(current_game_creation, prior.get("game_creation", 0)),
                    _normalize_kda(prior),
                    _normalize_avg(prior["vision_score"], 100.0),
                    _normalize_avg(prior["damage_to_champions"], 50000.0),
                    _normalize_avg(prior["healing"], 20000.0),
                    _normalize_avg(prior["gold_earned"], 25000.0),
                    _normalize_avg(prior["cs"], 400.0),
                    patch_major,
                    patch_minor,
                ]
                slot_mask[history_index] = 1

            history_champion_ids.append(slot_champs)
            history_role_ids.append(slot_roles)
            history_slot_ids.append(slot_slots)
            history_result_ids.append(slot_results)
            history_numeric.append(slot_numeric)
            history_mask.append(slot_mask)

        outcome_targets = _extract_auxiliary_targets_from_columns(
            blue_first_blood,
            blue_first_tower,
            blue_dragon_share,
            blue_gold_share,
        )
        if outcome_targets is None:
            if raw_match is None:
                raw_match = _fetch_raw_match(conn, match_id)
            outcome_targets = _extract_auxiliary_targets(raw_match) if raw_match else [0.5, 0.5, 0.5, 0.5]

        built.append(
            {
                "label": int(bool(participant_rows[0]["win"])),
                "blue_side": 1,
                "region_id": region_list.get(region or "", 0),
                "patch_features": [current_patch_major, current_patch_minor],
                "outcome_targets": outcome_targets,
                "current_champion_ids": current_champion_ids,
                "current_role_ids": current_role_ids,
                "current_team_ids": current_team_ids,
                "history_champion_ids": history_champion_ids,
                "history_role_ids": history_role_ids,
                "history_slot_ids": history_slot_ids,
                "history_result_ids": history_result_ids,
                "history_numeric": history_numeric,
                "history_mask": history_mask,
            }
        )

        for row in participant_rows:
            history = history_store.setdefault(row["puuid"], deque(maxlen=history_length))
            history.appendleft(row)

        if index % PROGRESS_UPDATE_EVERY == 0 or index == total_rows:
            _print_sequence_progress(index, total_rows, len(built), started_at, force=index == total_rows)

    conn.close()

    if not built:
        raise ValueError("No sequence-preserving training rows could be built from the current database.")

    tensors = {
        "labels": torch.tensor([row["label"] for row in built], dtype=torch.float32),
        "blue_side": torch.tensor([row["blue_side"] for row in built], dtype=torch.float32),
        "region_ids": torch.tensor([row["region_id"] for row in built], dtype=torch.long),
        "patch_features": torch.tensor([row["patch_features"] for row in built], dtype=torch.float32),
        "outcome_targets": torch.tensor([row["outcome_targets"] for row in built], dtype=torch.float32),
        "current_champion_ids": torch.tensor([row["current_champion_ids"] for row in built], dtype=torch.long),
        "current_role_ids": torch.tensor([row["current_role_ids"] for row in built], dtype=torch.long),
        "current_team_ids": torch.tensor([row["current_team_ids"] for row in built], dtype=torch.long),
        "history_champion_ids": torch.tensor([row["history_champion_ids"] for row in built], dtype=torch.long),
        "history_role_ids": torch.tensor([row["history_role_ids"] for row in built], dtype=torch.long),
        "history_slot_ids": torch.tensor([row["history_slot_ids"] for row in built], dtype=torch.long),
        "history_result_ids": torch.tensor([row["history_result_ids"] for row in built], dtype=torch.long),
        "history_numeric": torch.tensor([row["history_numeric"] for row in built], dtype=torch.float32),
        "history_mask": torch.tensor([row["history_mask"] for row in built], dtype=torch.bool),
    }
    metadata = {
        "champion_list": champion_list,
        "region_list": region_list,
        "num_champions": len(champion_list),
        "num_regions": max(len(region_list), 1),
        "history_length": history_length,
        "pad_champion_id": pad_champion_id,
    }
    _save_sequence_cache(cache_path, fingerprint, tensors, metadata)
    return tensors, metadata


def build_sequence_features_for_prediction(
    conn,
    champions,
    players=None,
    *,
    region=None,
    current_game_creation=None,
    game_version="",
    queue_id=QUEUE_ID_SOLO,
    champion_path=CHAMPION_LIST_PATH,
    region_path=REGION_LIST_PATH,
    history_length=HISTORY_LENGTH,
):
    if len(champions) != 10:
        raise ValueError("Exactly 10 champions are required in role order.")

    if players is None:
        players = [None] * 10
    elif len(players) != 10:
        raise ValueError("If provided, players must contain exactly 10 aligned entries.")

    champion_list = _load_json_mapping(champion_path)
    missing = [champ for champ in champions if champ not in champion_list]
    if missing:
        raise ValueError(f"Unknown champions for sequence features: {missing}")

    region_list = _load_json_mapping(region_path)
    if current_game_creation is None:
        current_game_creation = int(time.time() * 1000)

    if not game_version:
        try:
            row = conn.execute(
                """
                SELECT game_version
                FROM matches
                WHERE (? IS NULL OR queue_id = ?)
                  AND game_version IS NOT NULL
                  AND game_version != ''
                ORDER BY COALESCE(game_creation, 0) DESC, match_id DESC
                LIMIT 1
                """,
                (queue_id, queue_id),
            ).fetchone()
        except sqlite3.OperationalError:
            row = None
        game_version = row[0] if row else ""

    current_patch_major, current_patch_minor = parse_patch(game_version)
    pad_champion_id = len(champion_list) + 1
    current_role_ids = [ROLE_TO_ID[ROLE_ORDER[slot % 5]] for slot in range(10)]
    current_team_ids = [0] * 5 + [1] * 5

    history_champion_ids = []
    history_role_ids = []
    history_slot_ids = []
    history_result_ids = []
    history_numeric = []
    history_mask = []

    for slot_index, champion_name in enumerate(champions):
        prior_rows = []
        puuid = players[slot_index]
        if puuid:
            prior_rows = _fetch_recent_sequence_rows(
                conn,
                puuid,
                current_game_creation,
                queue_id,
                history_length,
            )

        slot_champs = [pad_champion_id] * history_length
        slot_roles = [0] * history_length
        slot_slots = [slot_index] * history_length
        slot_results = [0] * history_length
        slot_numeric = [[0.0] * 9 for _ in range(history_length)]
        slot_mask = [0] * history_length

        for history_index, prior in enumerate(prior_rows):
            patch_major, patch_minor = parse_patch(prior.get("game_version", ""))
            slot_champs[history_index] = champion_list.get(prior["champion_name"], pad_champion_id)
            slot_roles[history_index] = ROLE_TO_ID.get(prior["role"], 0)
            slot_results[history_index] = int(prior["win"])
            slot_numeric[history_index] = [
                _normalize_age(current_game_creation, prior.get("game_creation", 0)),
                _normalize_kda(prior),
                _normalize_avg(prior["vision_score"], 100.0),
                _normalize_avg(prior["damage_to_champions"], 50000.0),
                _normalize_avg(prior["healing"], 20000.0),
                _normalize_avg(prior["gold_earned"], 25000.0),
                _normalize_avg(prior["cs"], 400.0),
                patch_major,
                patch_minor,
            ]
            slot_mask[history_index] = 1

        history_champion_ids.append(slot_champs)
        history_role_ids.append(slot_roles)
        history_slot_ids.append(slot_slots)
        history_result_ids.append(slot_results)
        history_numeric.append(slot_numeric)
        history_mask.append(slot_mask)

    return {
        "labels": torch.tensor([[0.0]], dtype=torch.float32),
        "blue_side": torch.tensor([[1.0]], dtype=torch.float32),
        "region_ids": torch.tensor([region_list.get(region or "", 0)], dtype=torch.long),
        "patch_features": torch.tensor([[current_patch_major, current_patch_minor]], dtype=torch.float32),
        "current_champion_ids": torch.tensor([[champion_list[champ] for champ in champions]], dtype=torch.long),
        "current_role_ids": torch.tensor([current_role_ids], dtype=torch.long),
        "current_team_ids": torch.tensor([current_team_ids], dtype=torch.long),
        "history_champion_ids": torch.tensor([history_champion_ids], dtype=torch.long),
        "history_role_ids": torch.tensor([history_role_ids], dtype=torch.long),
        "history_slot_ids": torch.tensor([history_slot_ids], dtype=torch.long),
        "history_result_ids": torch.tensor([history_result_ids], dtype=torch.long),
        "history_numeric": torch.tensor([history_numeric], dtype=torch.float32),
        "history_mask": torch.tensor([history_mask], dtype=torch.bool),
    }


__all__ = [
    "HISTORY_LENGTH",
    "build_sequence_features_for_prediction",
    "build_sequence_training_tensors",
]
