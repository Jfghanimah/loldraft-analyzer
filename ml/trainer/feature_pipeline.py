import json
import os
import sqlite3
import time

import pandas as pd

from ml.data.match_format import ROLE_ORDER
from ml.data.match_storage import extract_participant_history_rows, get_match_columns
from ml.features.recent_history import (
    GLOBAL_FEATURES,
    PARTICIPANT_FEATURES,
    QUEUE_ID_SOLO,
    RecentHistoryStore,
    build_player_recent_feature_vector,
    dense_feature_columns,
    parse_patch,
)
from ml.runtime_config import get_db_path, load_runtime_env

CHAMPION_LIST_PATH = "ml/save_data/champion_list.json"
load_runtime_env()
DEFAULT_DB_PATH = get_db_path()
PROGRESS_UPDATE_EVERY = 1000
PROGRESS_BAR_WIDTH = 24
HISTORY_PROGRESS_UPDATE_EVERY = 25000


def _load_champion_list(champion_path):
    if not os.path.exists(champion_path):
        return {}
    with open(champion_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sync_champion_list(champion_list, ordered_records, champion_path):
    next_id = max(champion_list.values(), default=-1) + 1
    observed = sorted({champ for record in ordered_records for champ in record["champions"]})
    missing = [champ for champ in observed if champ not in champion_list]
    if missing:
        for champ in missing:
            champion_list[champ] = next_id
            next_id += 1
        os.makedirs(os.path.dirname(champion_path), exist_ok=True)
        ordered = dict(sorted(champion_list.items(), key=lambda item: item[1]))
        with open(champion_path, "w", encoding="utf-8") as f:
            json.dump(ordered, f, indent=4)
    return champion_list


def _load_training_matches(conn, queue_id):
    columns = get_match_columns(conn)
    if "ordered_match_json" not in columns:
        raise ValueError("Database does not contain the current ordered_match_json training column.")

    if "raw_match_json" in columns:
        return conn.execute(
            """
            SELECT match_id, raw_match_json, ordered_match_json, game_creation, game_version
            FROM matches
            WHERE ordered_match_json IS NOT NULL
              AND (? IS NULL OR queue_id = ?)
            ORDER BY COALESCE(game_creation, 0), match_id
            """,
            (queue_id, queue_id),
        ).fetchall()

    return conn.execute(
        """
        SELECT match_id, NULL as raw_match_json, ordered_match_json, game_creation, game_version
        FROM matches
        WHERE ordered_match_json IS NOT NULL
          AND (? IS NULL OR queue_id = ?)
        ORDER BY COALESCE(game_creation, 0), match_id
        """,
        (queue_id, queue_id),
    ).fetchall()


def _load_participant_history_rows_by_match(conn, queue_id):
    try:
        cursor = conn.execute(
            """
            SELECT match_id, puuid, champion_name, role, win, kills, deaths, assists,
                   vision_score, damage_to_champions, healing, gold_earned, cs, game_creation, team_id
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
        )
    except sqlite3.OperationalError:
        return {}

    rows_by_match = {}
    total_rows = 0
    started_at = time.time()
    for row in cursor:
        total_rows += 1
        rows_by_match.setdefault(row[0], []).append(
            {
                "puuid": row[1],
                "champion_name": row[2],
                "role": row[3],
                "win": row[4],
                "kills": row[5],
                "deaths": row[6],
                "assists": row[7],
                "vision_score": row[8],
                "damage_to_champions": row[9],
                "healing": row[10],
                "gold_earned": row[11],
                "cs": row[12],
                "game_creation": row[13],
                "team_id": row[14],
            }
        )
        if total_rows % HISTORY_PROGRESS_UPDATE_EVERY == 0:
            elapsed = time.time() - started_at
            _print_phase_status(
                f"Loaded {total_rows:,} participant-history rows across "
                f"{len(rows_by_match):,} matches in {_format_duration(elapsed)}..."
            )
    return rows_by_match


def _build_dense_features_from_rows(participant_rows, history_store, current_game_creation, game_version):
    feature_values = []
    for participant in participant_rows:
        feature_values.extend(
            history_store.feature_vector(
                participant["puuid"],
                participant["champion_name"],
                participant["role"],
                current_game_creation,
            )
        )
    feature_values.extend(parse_patch(game_version))
    return feature_values


def _format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(seconds), 60)
    return f"{minutes}m {remainder}s"


def _print_phase_status(message):
    print(f"[Phase 2] {message}", flush=True)


def _print_feature_progress(processed, total, built_rows, fallback_rows, started_at, *, force=False):
    if total <= 0:
        return
    percent = (processed / total) * 100.0
    elapsed = time.time() - started_at
    filled = min(PROGRESS_BAR_WIDTH, int((processed / total) * PROGRESS_BAR_WIDTH))
    bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
    rate = processed / elapsed if elapsed > 0 else 0.0
    remaining = total - processed
    eta = remaining / rate if rate > 0 else 0.0
    line = (
        f"\r[Phase 2] Building features [{bar}] {processed:,}/{total:,} "
        f"({percent:5.1f}%) | built={built_rows:,} | fallback={fallback_rows:,} | "
        f"elapsed={elapsed:.1f}s | eta={eta:.1f}s"
    )
    print(line.ljust(160), end="\n" if force else "", flush=True)


def build_rich_feature_dataframe(
    db_path=DEFAULT_DB_PATH,
    champion_path=CHAMPION_LIST_PATH,
    queue_id=QUEUE_ID_SOLO,
):
    if not os.path.exists(db_path):
        raise ValueError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    rows = _load_training_matches(conn, queue_id)
    participant_rows_by_match = _load_participant_history_rows_by_match(conn, queue_id)
    conn.close()

    ordered_records = [json.loads(row[2]) for row in rows]
    champion_list = _sync_champion_list(_load_champion_list(champion_path), ordered_records, champion_path)
    history_store = RecentHistoryStore()
    converted_rows = []
    fallback_rows = 0
    total_rows = len(rows)
    started_at = time.time()

    for index, (match_id, raw_json, ordered_json, game_creation, game_version) in enumerate(rows, start=1):
        ordered_record = json.loads(ordered_json)
        participant_rows = participant_rows_by_match.get(match_id, [])
        if len(participant_rows) != 10 and raw_json:
            participant_rows = extract_participant_history_rows(json.loads(raw_json))
            if len(participant_rows) == 10:
                fallback_rows += 1
        if len(participant_rows) != 10:
            continue

        feature_values = _build_dense_features_from_rows(
            participant_rows,
            history_store,
            game_creation or 0,
            game_version or ordered_record.get("game_version", ""),
        )

        blue_win = int(bool(ordered_record["blue_win"]))
        blue_side = int(ordered_record.get("blue_side", 1))
        champion_ids = [champion_list[champ] for champ in ordered_record["champions"]]
        converted_rows.append([blue_win] + champion_ids + [blue_side] + feature_values)
        history_store.add_match_rows(participant_rows)

        if index % PROGRESS_UPDATE_EVERY == 0 or index == total_rows:
            _print_feature_progress(index, total_rows, len(converted_rows), fallback_rows, started_at, force=index == total_rows)

    if not converted_rows:
        raise ValueError(
            "No rich feature rows could be built. Make sure the DB contains raw_match_json and ordered_match_json."
        )

    if fallback_rows:
        print(f"[Phase 2] participant_history missing for {fallback_rows:,} matches; fell back to raw_match_json.")

    column_names = [0] + list(range(1, 12)) + dense_feature_columns(ROLE_ORDER)
    df = pd.DataFrame(converted_rows, columns=column_names)
    return df.sample(frac=1, random_state=42).reset_index(drop=True), champion_list


def fetch_recent_history_rows(conn, puuid, current_game_creation, queue_id=QUEUE_ID_SOLO, limit=20):
    try:
        rows = conn.execute(
            """
            SELECT puuid, champion_name, role, win, kills, deaths, assists, vision_score,
                   damage_to_champions, healing, gold_earned, cs, game_creation
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
            "puuid": row[0],
            "champion_name": row[1],
            "role": row[2],
            "win": row[3],
            "kills": row[4],
            "deaths": row[5],
            "assists": row[6],
            "vision_score": row[7],
            "damage_to_champions": row[8],
            "healing": row[9],
            "gold_earned": row[10],
            "cs": row[11],
            "game_creation": row[12],
        }
        for row in rows
    ]


def build_dense_features_for_prediction(
    conn,
    champions,
    players=None,
    current_game_creation=None,
    queue_id=QUEUE_ID_SOLO,
):
    if len(champions) != 10:
        raise ValueError("Exactly 10 champions are required in role order.")

    if players is None:
        players = [None] * 10
    elif len(players) != 10:
        raise ValueError("If provided, players must contain exactly 10 aligned entries.")

    if current_game_creation is None:
        current_game_creation = int(time.time() * 1000)

    feature_values = []
    for slot, champion_name in enumerate(champions):
        role = ROLE_ORDER[slot % 5]
        puuid = players[slot]
        prior_rows = []
        if puuid:
            prior_rows = fetch_recent_history_rows(conn, puuid, current_game_creation, queue_id=queue_id)
        feature_values.extend(
            build_player_recent_feature_vector(
                prior_rows,
                champion_name,
                role,
                current_game_creation,
            )
        )

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
    feature_values.extend(parse_patch(row[0] if row else ""))
    return feature_values


__all__ = [
    "PARTICIPANT_FEATURES",
    "GLOBAL_FEATURES",
    "build_dense_features_for_prediction",
    "build_rich_feature_dataframe",
    "dense_feature_columns",
    "fetch_recent_history_rows",
]
