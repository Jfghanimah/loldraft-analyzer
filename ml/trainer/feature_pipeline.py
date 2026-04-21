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
REGION_LIST_PATH = "ml/save_data/region_list.json"
load_runtime_env()
DEFAULT_DB_PATH = get_db_path()
PROGRESS_UPDATE_EVERY = 1000
PROGRESS_BAR_WIDTH = 24
HISTORY_PROGRESS_UPDATE_EVERY = 25000
ROLE_SORT_ORDER = {role: index for index, role in enumerate(ROLE_ORDER)}
CHAMPION_COLUMNS = tuple(f"champion_{slot}" for slot in range(10))
AUX_TARGET_COLUMNS = (
    "target_gold_diff",
    "target_blue_dragons",
    "target_red_dragons",
    "target_game_length_minutes",
)
READ_PRAGMAS = (
    "PRAGMA cache_size = -65536",
    "PRAGMA mmap_size = 0",
    "PRAGMA read_uncommitted = ON",
    "PRAGMA temp_store = MEMORY",
)

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None


def _load_champion_list(champion_path):
    if not os.path.exists(champion_path):
        return {}
    with open(champion_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_region_list(region_path):
    if not os.path.exists(region_path):
        return {}
    with open(region_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_id_mapping(mapping, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ordered = dict(sorted(mapping.items(), key=lambda item: item[1]))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=4)


def _sync_champion_list(champion_list, observed_champions, champion_path):
    next_id = max(champion_list.values(), default=-1) + 1
    observed = sorted(set(observed_champions))
    missing = [champ for champ in observed if champ not in champion_list]
    if missing:
        for champ in missing:
            champion_list[champ] = next_id
            next_id += 1
        _write_id_mapping(champion_list, champion_path)
    return champion_list


def _sync_region_list(region_list, observed_regions, region_path):
    next_id = max(region_list.values(), default=-1) + 1
    observed = sorted({region for region in observed_regions if region})
    missing = [region for region in observed if region not in region_list]
    if missing:
        for region in missing:
            region_list[region] = next_id
            next_id += 1
        _write_id_mapping(region_list, region_path)
    return region_list


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
    if queue_id is None:
        row = conn.execute(
            """
            SELECT count(*)
            FROM matches
            WHERE ordered_match_json IS NOT NULL
            """
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT count(*)
            FROM matches
            WHERE ordered_match_json IS NOT NULL
              AND queue_id = ?
            """,
            (queue_id,),
        ).fetchone()
    return int(row[0] or 0) if row else 0


def _iter_training_ordered_match_json(conn, queue_id):
    if queue_id is None:
        return conn.execute(
            """
            SELECT ordered_match_json, region
            FROM matches
            WHERE ordered_match_json IS NOT NULL
            """
        )
    return conn.execute(
        """
        SELECT ordered_match_json, region
        FROM matches
        WHERE ordered_match_json IS NOT NULL
          AND queue_id = ?
        """,
        (queue_id,),
    )


def _iter_training_matches(conn, queue_id):
    if queue_id is None:
        return conn.execute(
            """
            SELECT match_id, ordered_match_json, region, game_creation, game_end_timestamp, game_version,
                   blue_dragons, red_dragons, gold_diff, game_length_minutes
            FROM matches
            WHERE ordered_match_json IS NOT NULL
            ORDER BY game_creation, match_id
            """
        )
    return conn.execute(
        """
        SELECT match_id, ordered_match_json, region, game_creation, game_end_timestamp, game_version,
               blue_dragons, red_dragons, gold_diff, game_length_minutes
        FROM matches
        WHERE ordered_match_json IS NOT NULL
          AND queue_id = ?
        ORDER BY game_creation, match_id
        """,
        (queue_id,),
    )


def _loads_json(payload):
    if orjson is not None:
        return orjson.loads(payload)
    return json.loads(payload)


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
    duration_minutes = max(float(((row[15] or 0) - (row[13] or 0)) / 60_000.0), 1.0)
    kills = int(row[5] or 0)
    deaths = int(row[6] or 0)
    assists = int(row[7] or 0)
    damage_to_champions = float(row[9] or 0.0)
    healing = float(row[10] or 0.0)
    gold_earned = float(row[11] or 0.0)
    cs = float(row[12] or 0.0)
    return {
        "puuid": row[1],
        "champion_name": row[2],
        "role": row[3],
        "win": int(row[4] or 0),
        "kills": kills,
        "deaths": deaths,
        "assists": assists,
        "vision_score": float(row[8] or 0.0),
        "damage_to_champions": damage_to_champions,
        "healing": healing,
        "gold_earned": gold_earned,
        "cs": cs,
        "game_creation": int(row[13] or 0),
        "team_id": int(row[14] or 0),
        "duration_minutes": duration_minutes,
        "kda_value": (kills + assists) / max(1, deaths),
        "dpm_value": damage_to_champions / duration_minutes,
        "gpm_value": gold_earned / duration_minutes,
        "cspm_value": cs / duration_minutes,
        "vspm_value": float(row[8] or 0.0) / duration_minutes,
        "hpm_value": healing / duration_minutes,
    }


def _sort_participant_rows(rows):
    rows.sort(key=lambda row: (row["team_id"], ROLE_SORT_ORDER.get(row["role"], 99)))
    return rows


def _iter_participant_rows_by_match(conn, queue_id):
    try:
        if queue_id is None:
            cursor = conn.execute(
                """
                SELECT ph.match_id, ph.puuid, ph.champion_name, ph.role, ph.win, ph.kills, ph.deaths, ph.assists,
                       ph.vision_score, ph.damage_to_champions, ph.healing, ph.gold_earned, ph.cs,
                       ph.game_creation, ph.team_id, m.game_end_timestamp
                FROM participant_history ph
                LEFT JOIN matches m ON m.match_id = ph.match_id
                ORDER BY ph.game_creation, ph.match_id, ph.team_id
                """
            )
        else:
            cursor = conn.execute(
                """
                SELECT ph.match_id, ph.puuid, ph.champion_name, ph.role, ph.win, ph.kills, ph.deaths, ph.assists,
                       ph.vision_score, ph.damage_to_champions, ph.healing, ph.gold_earned, ph.cs,
                       ph.game_creation, ph.team_id, m.game_end_timestamp
                FROM participant_history ph
                LEFT JOIN matches m ON m.match_id = ph.match_id
                WHERE ph.queue_id = ?
                ORDER BY ph.game_creation, ph.match_id, ph.team_id
                """,
                (queue_id,),
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


def _extract_multitask_targets(raw_match, game_creation, game_end_timestamp, participant_rows=None):
    info = raw_match.get("info", {}) if raw_match else {}
    teams = {team.get("teamId"): team for team in info.get("teams", [])}
    blue_team = teams.get(100, {})
    red_team = teams.get(200, {})
    blue_dragons = float(blue_team.get("objectives", {}).get("dragon", {}).get("kills", 0) or 0)
    red_dragons = float(red_team.get("objectives", {}).get("dragon", {}).get("kills", 0) or 0)

    blue_gold = 0.0
    red_gold = 0.0
    if participant_rows:
        for participant in participant_rows:
            team_id = int(participant.get("team_id", 0) or 0)
            gold = float(participant.get("gold_earned", 0.0) or 0.0)
            if team_id == 100:
                blue_gold += gold
            elif team_id == 200:
                red_gold += gold
    else:
        for participant in info.get("participants", []):
            team_id = int(participant.get("teamId", 0) or 0)
            gold = float(participant.get("goldEarned", 0.0) or 0.0)
            if team_id == 100:
                blue_gold += gold
            elif team_id == 200:
                red_gold += gold

    game_length_minutes = max(float(((game_end_timestamp or 0) - (game_creation or 0)) / 60_000.0), 1.0)
    return [
        blue_gold - red_gold,
        blue_dragons,
        red_dragons,
        game_length_minutes,
    ]


def _extract_compact_targets(
    *,
    blue_dragons,
    red_dragons,
    gold_diff,
    game_length_minutes,
    participant_rows,
    game_creation,
    game_end_timestamp,
):
    derived_gold_diff = None
    if participant_rows:
        blue_gold = 0.0
        red_gold = 0.0
        for participant in participant_rows:
            team_id = int(participant.get("team_id", 0) or 0)
            gold = float(participant.get("gold_earned", 0.0) or 0.0)
            if team_id == 100:
                blue_gold += gold
            elif team_id == 200:
                red_gold += gold
        derived_gold_diff = blue_gold - red_gold

    compact_game_length = game_length_minutes
    if compact_game_length is None:
        compact_game_length = max(float(((game_end_timestamp or 0) - (game_creation or 0)) / 60_000.0), 1.0)

    if blue_dragons is None or red_dragons is None:
        return None

    return [
        float(derived_gold_diff if derived_gold_diff is not None else (gold_diff or 0.0)),
        float(blue_dragons or 0.0),
        float(red_dragons or 0.0),
        float(compact_game_length or 1.0),
    ]


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
    region_path=REGION_LIST_PATH,
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

    champion_list = _load_champion_list(champion_path)
    region_list = _load_region_list(region_path)
    champion_mapping_updated = False
    region_mapping_updated = False
    _print_phase_status(
        f"Loaded champion map with {len(champion_list):,} champions and {max(len(region_list), 1):,} regions"
    )

    history_store = RecentHistoryStore()
    column_names = ["label", *CHAMPION_COLUMNS, "region_id", *AUX_TARGET_COLUMNS, *dense_feature_columns(ROLE_ORDER)]
    feature_matrix = np.empty((max(total_rows, 1), len(column_names)), dtype=np.float64)
    participant_rows_by_match = iter(_iter_participant_rows_by_match(conn, queue_id))
    current_participant_group = _next_or_none(participant_rows_by_match)
    built_rows = 0
    fallback_rows = 0
    started_at = time.time()
    _print_phase_status(f"Building dense pre-match features for {total_rows:,} matches...")

    raw_fallback_rows = 0
    target_raw_fallback_rows = 0

    for index, (match_id, ordered_json, region, game_creation, game_end_timestamp, game_version, blue_dragons, red_dragons, gold_diff, game_length_minutes) in enumerate(
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

        raw_match = None
        if len(participant_rows) != 10:
            raw_match = _fetch_raw_match(conn, match_id)
            participant_rows = _sort_participant_rows(extract_participant_history_rows(raw_match) if raw_match else [])
            if len(participant_rows) == 10:
                duration_minutes = max(float(((game_end_timestamp or 0) - (game_creation or 0)) / 60_000.0), 1.0)
                for row in participant_rows:
                    row["duration_minutes"] = duration_minutes
                fallback_rows += 1
                raw_fallback_rows += 1
        if len(participant_rows) != 10:
            if index % PROGRESS_UPDATE_EVERY == 0 or index == total_rows:
                _print_feature_progress(index, total_rows, built_rows, fallback_rows, started_at, force=index == total_rows)
            continue

        ordered_record = _loads_json(ordered_json)
        feature_values = _build_dense_features_from_rows(
            participant_rows,
            history_store,
            int(game_creation or 0),
            game_version or ordered_record.get("game_version", ""),
        )

        blue_win = int(bool(ordered_record["blue_win"]))
        region_key = region or ""
        if region_key and region_key not in region_list:
            region_list[region_key] = len(region_list)
            region_mapping_updated = True
        region_id = region_list.get(region_key, 0)
        champion_ids = []
        for champion_name in ordered_record["champions"]:
            champion_id = champion_list.get(champion_name)
            if champion_id is None:
                champion_id = len(champion_list)
                champion_list[champion_name] = champion_id
                champion_mapping_updated = True
            champion_ids.append(champion_id)
        targets = _extract_compact_targets(
            blue_dragons=blue_dragons,
            red_dragons=red_dragons,
            gold_diff=gold_diff,
            game_length_minutes=game_length_minutes,
            participant_rows=participant_rows,
            game_creation=int(game_creation or 0),
            game_end_timestamp=int(game_end_timestamp or 0),
        )
        if targets is None:
            if raw_match is None:
                raw_match = _fetch_raw_match(conn, match_id)
            targets = _extract_multitask_targets(
                raw_match,
                int(game_creation or 0),
                int(game_end_timestamp or 0),
                participant_rows=participant_rows,
            )
            target_raw_fallback_rows += 1
        feature_matrix[built_rows] = np.asarray(
            [blue_win] + champion_ids + [region_id] + targets + feature_values,
            dtype=np.float64,
        )
        built_rows += 1
        history_store.add_match_rows(participant_rows)

        if index % PROGRESS_UPDATE_EVERY == 0 or index == total_rows:
            _print_feature_progress(index, total_rows, built_rows, fallback_rows, started_at, force=index == total_rows)

    conn.close()

    if champion_mapping_updated:
        _write_id_mapping(champion_list, champion_path)
    if region_mapping_updated:
        _write_id_mapping(region_list, region_path)

    if not built_rows:
        raise ValueError(
            "No rich feature rows could be built. Make sure the DB contains raw_match_json and ordered_match_json."
        )

    if fallback_rows:
        print(f"[Phase 2] participant_history missing for {fallback_rows:,} matches; fell back to raw_match_json.")
    if target_raw_fallback_rows:
        print(
            f"[Phase 2] compact auxiliary targets missing for {target_raw_fallback_rows:,} matches; "
            "fell back to raw_match_json for target extraction."
        )

    df = pd.DataFrame(feature_matrix[:built_rows], columns=column_names)
    _print_phase_status(
        f"Feature dataframe ready with {len(df):,} matches x {len(df.columns):,} columns"
    )
    return df.sample(frac=1, random_state=42).reset_index(drop=True), champion_list


def fetch_recent_history_rows(conn, puuid, current_game_creation, queue_id=QUEUE_ID_SOLO, limit=20):
    try:
        rows = conn.execute(
            """
            SELECT ph.puuid, ph.champion_name, ph.role, ph.win, ph.kills, ph.deaths, ph.assists, ph.vision_score,
                   ph.damage_to_champions, ph.healing, ph.gold_earned, ph.cs, ph.game_creation, m.game_end_timestamp
            FROM participant_history ph
            LEFT JOIN matches m ON m.match_id = ph.match_id
            WHERE ph.puuid = ?
              AND ph.queue_id = ?
              AND ph.game_creation < ?
            ORDER BY ph.game_creation DESC, ph.match_id DESC
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
            "duration_minutes": max(float((((row[13] or 0) - (row[12] or 0)) / 60_000.0) if row[13] else 30.0), 1.0),
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
        if queue_id is None:
            row = conn.execute(
                """
                SELECT game_version FROM matches
                WHERE game_version IS NOT NULL AND game_version != ''
                ORDER BY game_creation DESC, match_id DESC LIMIT 1
                """
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT game_version FROM matches
                WHERE queue_id = ?
                  AND game_version IS NOT NULL AND game_version != ''
                ORDER BY game_creation DESC LIMIT 1
                """,
                (queue_id,),
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
