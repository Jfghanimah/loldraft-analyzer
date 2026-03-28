import json
import os
import sqlite3
import time

import numpy as np
import pandas as pd

from ml.data.match_format import ROLE_ORDER
from ml.data.match_storage import (
    connect_sqlite,
    ensure_training_read_indexes,
    extract_participant_history_rows,
    get_match_columns,
)
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
ROLE_SORT_ORDER = {role: index for index, role in enumerate(ROLE_ORDER)}
READ_PRAGMAS = (
    "PRAGMA cache_size = -65536",
    "PRAGMA mmap_size = 0",
    "PRAGMA read_uncommitted = ON",
    "PRAGMA temp_store = MEMORY",
)


def _load_champion_list(champion_path):
    if not os.path.exists(champion_path):
        return {}
    with open(champion_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sync_champion_list(champion_list, observed_champions, champion_path):
    next_id = max(champion_list.values(), default=-1) + 1
    observed = sorted(set(observed_champions))
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


def _ensure_training_schema(conn):
    columns = get_match_columns(conn)
    if "ordered_match_json" not in columns:
        raise ValueError("Database does not contain the current ordered_match_json training column.")
    return columns


def _configure_training_connection(conn):
    for pragma in READ_PRAGMAS:
        try:
            conn.execute(pragma)
        except sqlite3.OperationalError:
            continue


def _count_training_matches(conn, queue_id):
    row = conn.execute(
        """
        SELECT count(*)
        FROM matches
        WHERE ordered_match_json IS NOT NULL
          AND (? IS NULL OR queue_id = ?)
        """,
        (queue_id, queue_id),
    ).fetchone()
    return int(row[0] or 0) if row else 0


def _iter_training_ordered_match_json(conn, queue_id):
    return conn.execute(
        """
        SELECT ordered_match_json
        FROM matches
        WHERE ordered_match_json IS NOT NULL
          AND (? IS NULL OR queue_id = ?)
        """,
        (queue_id, queue_id),
    )


def _iter_training_matches(conn, queue_id):
    return conn.execute(
        """
        SELECT match_id, ordered_match_json, game_creation, game_version
        FROM matches
        WHERE ordered_match_json IS NOT NULL
          AND (? IS NULL OR queue_id = ?)
        ORDER BY COALESCE(game_creation, 0), match_id
        """,
        (queue_id, queue_id),
    )


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
    if not raw_match:
        return []
    return extract_participant_history_rows(raw_match)


def _participant_history_row_from_tuple(row):
    return {
        "puuid": row[1],
        "champion_name": row[2],
        "role": row[3],
        "win": int(row[4] or 0),
        "kills": int(row[5] or 0),
        "deaths": int(row[6] or 0),
        "assists": int(row[7] or 0),
        "vision_score": float(row[8] or 0.0),
        "damage_to_champions": float(row[9] or 0.0),
        "healing": float(row[10] or 0.0),
        "gold_earned": float(row[11] or 0.0),
        "cs": float(row[12] or 0.0),
        "game_creation": int(row[13] or 0),
        "team_id": int(row[14] or 0),
    }


def _sort_participant_rows(rows):
    rows.sort(key=lambda row: (row["team_id"], ROLE_SORT_ORDER.get(row["role"], 99)))
    return rows


def _iter_participant_rows_by_match(conn, queue_id):
    try:
        cursor = conn.execute(
            """
            SELECT match_id, puuid, champion_name, role, win, kills, deaths, assists,
                   vision_score, damage_to_champions, healing, gold_earned, cs, game_creation, team_id
            FROM participant_history
            WHERE (? IS NULL OR queue_id = ?)
            ORDER BY COALESCE(game_creation, 0), match_id, team_id
            """,
            (queue_id, queue_id),
        )
    except sqlite3.OperationalError:
        return

    current_match_id = None
    current_game_creation = 0
    current_rows = []
    total_rows = 0
    started_at = time.time()
    for row in cursor:
        total_rows += 1
        match_id = row[0]
        game_creation = int(row[13] or 0)
        if current_match_id is not None and match_id != current_match_id:
            yield (current_game_creation, current_match_id), current_match_id, _sort_participant_rows(current_rows)
            current_rows = []
        current_match_id = match_id
        current_game_creation = game_creation
        current_rows.append(_participant_history_row_from_tuple(row))
        if total_rows % HISTORY_PROGRESS_UPDATE_EVERY == 0:
            elapsed = time.time() - started_at
            _print_phase_status(
                f"Streamed {total_rows:,} participant-history rows in {_format_duration(elapsed)}..."
            )
    if current_match_id is not None:
        yield (current_game_creation, current_match_id), current_match_id, _sort_participant_rows(current_rows)


def _next_or_none(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None


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

    _print_phase_status(f"Opening training database at {db_path}")
    conn = connect_sqlite(db_path)
    _configure_training_connection(conn)
    _ensure_training_schema(conn)

    started_at = time.time()
    _print_phase_status("Ensuring training read indexes...")
    try:
        ensure_training_read_indexes(conn)
    except sqlite3.OperationalError as exc:
        _print_phase_status(f"Could not ensure training read indexes ({exc}); continuing with existing indexes.")
    else:
        _print_phase_status(f"Training read indexes ready in {_format_duration(time.time() - started_at)}")

    total_rows = _count_training_matches(conn, queue_id)
    _print_phase_status(
        f"Found {total_rows:,} ordered matches for feature generation"
    )

    started_at = time.time()
    _print_phase_status("Scanning ordered match rows to sync champion IDs...")
    observed_champions = set()
    for index, (ordered_match_json,) in enumerate(_iter_training_ordered_match_json(conn, queue_id), start=1):
        observed_champions.update(json.loads(ordered_match_json)["champions"])
        if index % PROGRESS_UPDATE_EVERY == 0 or index == total_rows:
            _print_phase_status(f"Scanned {index:,}/{total_rows:,} ordered matches for champion sync")
    champion_list = _sync_champion_list(_load_champion_list(champion_path), observed_champions, champion_path)
    _print_phase_status(
        f"Champion map ready with {len(champion_list):,} entries after "
        f"{_format_duration(time.time() - started_at)}"
    )

    history_store = RecentHistoryStore()
    column_names = [0] + list(range(1, 12)) + dense_feature_columns(ROLE_ORDER)
    feature_matrix = np.empty((max(total_rows, 1), len(column_names)), dtype=np.float64)
    participant_rows_by_match = iter(_iter_participant_rows_by_match(conn, queue_id))
    current_participant_group = _next_or_none(participant_rows_by_match)
    built_rows = 0
    fallback_rows = 0
    started_at = time.time()
    _print_phase_status(f"Building dense pre-match features for {total_rows:,} matches...")

    for index, (match_id, ordered_json, game_creation, game_version) in enumerate(
        _iter_training_matches(conn, queue_id),
        start=1,
    ):
        match_sort_key = (int(game_creation or 0), match_id)
        while current_participant_group is not None and current_participant_group[0] < match_sort_key:
            current_participant_group = _next_or_none(participant_rows_by_match)

        participant_rows = []
        if (
            current_participant_group is not None
            and current_participant_group[0] == match_sort_key
            and current_participant_group[1] == match_id
        ):
            participant_rows = current_participant_group[2]
            current_participant_group = _next_or_none(participant_rows_by_match)

        if len(participant_rows) != 10:
            participant_rows = _sort_participant_rows(_fetch_raw_participant_rows(conn, match_id))
            if len(participant_rows) == 10:
                fallback_rows += 1
        if len(participant_rows) != 10:
            if index % PROGRESS_UPDATE_EVERY == 0 or index == total_rows:
                _print_feature_progress(index, total_rows, built_rows, fallback_rows, started_at, force=index == total_rows)
            continue

        ordered_record = json.loads(ordered_json)
        feature_values = _build_dense_features_from_rows(
            participant_rows,
            history_store,
            int(game_creation or 0),
            game_version or ordered_record.get("game_version", ""),
        )

        blue_win = int(bool(ordered_record["blue_win"]))
        blue_side = int(ordered_record.get("blue_side", 1))
        champion_ids = [champion_list[champ] for champ in ordered_record["champions"]]
        feature_matrix[built_rows] = np.asarray([blue_win] + champion_ids + [blue_side] + feature_values, dtype=np.float64)
        built_rows += 1
        history_store.add_match_rows(participant_rows)

        if index % PROGRESS_UPDATE_EVERY == 0 or index == total_rows:
            _print_feature_progress(index, total_rows, built_rows, fallback_rows, started_at, force=index == total_rows)

    conn.close()

    if not built_rows:
        raise ValueError(
            "No rich feature rows could be built. Make sure the DB contains raw_match_json and ordered_match_json."
        )

    if fallback_rows:
        print(f"[Phase 2] participant_history missing for {fallback_rows:,} matches; fell back to raw_match_json.")

    df = pd.DataFrame(feature_matrix[:built_rows], columns=column_names)
    _print_phase_status(
        f"Feature dataframe ready with {len(df):,} matches x {len(df.columns):,} columns"
    )
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
