import json
import sqlite3
import time
from pathlib import Path

from ml.data.match_format import normalize_position


MATCH_COLUMNS = {
    "match_id": "TEXT PRIMARY KEY",
    "match_data": "TEXT",
    "region": "TEXT",
    "raw_match_json": "TEXT",
    "ordered_match_json": "TEXT",
    "rank_snapshot_json": "TEXT",
    "data_source": "TEXT",
    "collector_id": "TEXT",
    "game_version": "TEXT",
    "queue_id": "INTEGER",
    "game_creation": "INTEGER",
    "game_end_timestamp": "INTEGER",
    "blue_first_blood": "INTEGER",
    "blue_first_tower": "INTEGER",
    "blue_dragon_share": "REAL",
    "blue_gold_share": "REAL",
    "last_updated_ts": "INTEGER",
}
PARTICIPANT_HISTORY_BACKFILL_BATCH_ROWS = 10000
PARTICIPANT_HISTORY_INDEXES = (
    (
        "idx_participant_history_puuid_queue_game",
        "CREATE INDEX IF NOT EXISTS idx_participant_history_puuid_queue_game "
        "ON participant_history(puuid, queue_id, game_creation)",
    ),
    (
        "idx_participant_history_puuid_champion_game",
        "CREATE INDEX IF NOT EXISTS idx_participant_history_puuid_champion_game "
        "ON participant_history(puuid, champion_name, game_creation)",
    ),
    (
        "idx_participant_history_puuid_role_game",
        "CREATE INDEX IF NOT EXISTS idx_participant_history_puuid_role_game "
        "ON participant_history(puuid, role, game_creation)",
    ),
)


def _participant_history_insert_values(participant_rows):
    return [
        (
            row["match_id"],
            row["puuid"],
            row["queue_id"],
            row["game_creation"],
            row["champion_name"],
            row["role"],
            row["team_id"],
            row["win"],
            row["kills"],
            row["deaths"],
            row["assists"],
            row["vision_score"],
            row["damage_to_champions"],
            row["healing"],
            row["gold_earned"],
            row["cs"],
            row["game_version"],
        )
        for row in participant_rows
    ]


def _insert_participant_history_rows(conn: sqlite3.Connection, participant_rows):
    if not participant_rows:
        return
    conn.executemany(
        """
        INSERT INTO participant_history (
            match_id, puuid, queue_id, game_creation, champion_name, role, team_id, win,
            kills, deaths, assists, vision_score, damage_to_champions, healing, gold_earned, cs, game_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _participant_history_insert_values(participant_rows),
    )


def _create_participant_history_indexes(conn: sqlite3.Connection):
    for _, ddl in PARTICIPANT_HISTORY_INDEXES:
        conn.execute(ddl)


def _drop_participant_history_indexes(conn: sqlite3.Connection):
    for name, _ in PARTICIPANT_HISTORY_INDEXES:
        conn.execute(f"DROP INDEX IF EXISTS {name}")


def connect_sqlite(db_path, *, read_only=False):
    path = Path(db_path)
    if read_only:
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=5.0)
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    conn = sqlite3.connect(path, timeout=5.0)
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def ensure_match_schema(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS matches (
            match_id TEXT PRIMARY KEY,
            match_data TEXT,
            region TEXT,
            raw_match_json TEXT,
            ordered_match_json TEXT,
            rank_snapshot_json TEXT,
            data_source TEXT,
            collector_id TEXT,
            game_version TEXT,
            queue_id INTEGER,
            game_creation INTEGER,
            game_end_timestamp INTEGER,
            last_updated_ts INTEGER
        )
        """
    )

    existing = {
        row[1]
        for row in conn.execute("PRAGMA table_info(matches)").fetchall()
    }
    for column, column_type in MATCH_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE matches ADD COLUMN {column} {column_type}")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_matches_queue_id ON matches(queue_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_matches_game_version ON matches(game_version)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_matches_game_creation ON matches(game_creation)")
    ensure_participant_history_schema(conn)


def get_match_columns(conn: sqlite3.Connection):
    return {row[1] for row in conn.execute("PRAGMA table_info(matches)").fetchall()}


def ensure_rank_snapshot_schema(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS puuid_rank_cache (
            puuid TEXT PRIMARY KEY,
            platform TEXT,
            encrypted_summoner_id TEXT,
            snapshot_json TEXT,
            fetched_ts INTEGER
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_puuid_rank_cache_fetched_ts ON puuid_rank_cache(fetched_ts)"
    )


def ensure_participant_history_schema(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS participant_history (
            match_id TEXT NOT NULL,
            puuid TEXT NOT NULL,
            queue_id INTEGER,
            game_creation INTEGER,
            champion_name TEXT,
            role TEXT,
            team_id INTEGER,
            win INTEGER,
            kills INTEGER,
            deaths INTEGER,
            assists INTEGER,
            vision_score REAL,
            damage_to_champions REAL,
            healing REAL,
            gold_earned REAL,
            cs REAL,
            game_version TEXT,
            PRIMARY KEY (match_id, puuid)
        )
        """
    )
    existing = {
        row[1]
        for row in conn.execute("PRAGMA table_info(participant_history)").fetchall()
    }
    if "gold_earned" not in existing:
        conn.execute("ALTER TABLE participant_history ADD COLUMN gold_earned REAL")
    if "cs" not in existing:
        conn.execute("ALTER TABLE participant_history ADD COLUMN cs REAL")
    _create_participant_history_indexes(conn)


def extract_participant_history_rows(match):
    info = match.get("info", {})
    team_results = {team.get("teamId"): int(bool(team.get("win"))) for team in info.get("teams", [])}
    rows = []

    for participant in info.get("participants", []):
        role = normalize_position(
            participant.get("teamPosition"),
            participant.get("individualPosition"),
        )
        if role is None:
            return []

        team_id = participant.get("teamId")
        puuid = participant.get("puuid")
        if team_id not in team_results or not puuid:
            return []

        rows.append(
            {
                "match_id": match.get("metadata", {}).get("matchId"),
                "puuid": puuid,
                "queue_id": info.get("queueId"),
                "game_creation": info.get("gameCreation"),
                "champion_name": participant.get("championName", ""),
                "role": role,
                "team_id": team_id,
                "win": team_results[team_id],
                "kills": int(participant.get("kills", 0) or 0),
                "deaths": int(participant.get("deaths", 0) or 0),
                "assists": int(participant.get("assists", 0) or 0),
                "vision_score": float(participant.get("visionScore", 0.0) or 0.0),
                "damage_to_champions": float(participant.get("totalDamageDealtToChampions", 0.0) or 0.0),
                "healing": float(
                    (participant.get("totalHeal", 0.0) or 0.0)
                    + (participant.get("totalHealsOnTeammates", 0.0) or 0.0)
                ),
                "gold_earned": float(participant.get("goldEarned", 0.0) or 0.0),
                "cs": float(
                    (participant.get("totalMinionsKilled", 0.0) or 0.0)
                    + (participant.get("neutralMinionsKilled", 0.0) or 0.0)
                ),
                "game_version": info.get("gameVersion", ""),
            }
        )

    return rows if len(rows) == 10 else []


def extract_auxiliary_outcome_features(match):
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
    blue_dragon_share = blue_dragons / total_dragons if total_dragons > 0 else 0.5

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
    blue_gold_share = blue_gold / total_gold if total_gold > 0 else 0.5

    return {
        "blue_first_blood": int(bool(_objective(blue_objectives, "champion", "first", False))),
        "blue_first_tower": int(bool(_objective(blue_objectives, "tower", "first", False))),
        "blue_dragon_share": float(blue_dragon_share),
        "blue_gold_share": float(blue_gold_share),
    }


def extract_storage_payload(match, region, ordered_match_data, collector_id="", rank_snapshots=None):
    info = match["info"]
    now_ts = int(time.time())
    ordered_json = json.dumps(ordered_match_data) if ordered_match_data is not None else None
    rank_snapshot_json = json.dumps(rank_snapshots) if rank_snapshots is not None else None
    auxiliary_outcomes = extract_auxiliary_outcome_features(match)
    legacy_projection = json.dumps(
        [bool(info["teams"][0]["win"])] + [participant["championName"] for participant in info["participants"]]
    )

    return {
        "region": region,
        "match_data": ordered_json or legacy_projection,
        "raw_match_json": json.dumps(match),
        "ordered_match_json": ordered_json,
        "rank_snapshot_json": rank_snapshot_json,
        "data_source": "riot_api_match_v5",
        "collector_id": collector_id,
        "game_version": info.get("gameVersion", ""),
        "queue_id": info.get("queueId"),
        "game_creation": info.get("gameCreation"),
        "game_end_timestamp": info.get("gameEndTimestamp"),
        "blue_first_blood": auxiliary_outcomes["blue_first_blood"],
        "blue_first_tower": auxiliary_outcomes["blue_first_tower"],
        "blue_dragon_share": auxiliary_outcomes["blue_dragon_share"],
        "blue_gold_share": auxiliary_outcomes["blue_gold_share"],
        "last_updated_ts": now_ts,
        "participant_history_rows": extract_participant_history_rows(match),
    }


def upsert_match_record(conn: sqlite3.Connection, match_id: str, payload: dict):
    conn.execute(
        """
        INSERT INTO matches (
            match_id,
            match_data,
            region,
            raw_match_json,
            ordered_match_json,
            rank_snapshot_json,
            data_source,
            collector_id,
            game_version,
            queue_id,
            game_creation,
            game_end_timestamp,
            blue_first_blood,
            blue_first_tower,
            blue_dragon_share,
            blue_gold_share,
            last_updated_ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(match_id) DO UPDATE SET
            match_data = excluded.match_data,
            region = excluded.region,
            raw_match_json = excluded.raw_match_json,
            ordered_match_json = excluded.ordered_match_json,
            rank_snapshot_json = excluded.rank_snapshot_json,
            data_source = excluded.data_source,
            collector_id = excluded.collector_id,
            game_version = excluded.game_version,
            queue_id = excluded.queue_id,
            game_creation = excluded.game_creation,
            game_end_timestamp = excluded.game_end_timestamp,
            blue_first_blood = excluded.blue_first_blood,
            blue_first_tower = excluded.blue_first_tower,
            blue_dragon_share = excluded.blue_dragon_share,
            blue_gold_share = excluded.blue_gold_share,
            last_updated_ts = excluded.last_updated_ts
        """,
        (
            match_id,
            payload["match_data"],
            payload["region"],
            payload["raw_match_json"],
            payload["ordered_match_json"],
            payload["rank_snapshot_json"],
            payload["data_source"],
            payload["collector_id"],
            payload["game_version"],
            payload["queue_id"],
            payload["game_creation"],
            payload["game_end_timestamp"],
            payload["blue_first_blood"],
            payload["blue_first_tower"],
            payload["blue_dragon_share"],
            payload["blue_gold_share"],
            payload["last_updated_ts"],
        ),
    )
    participant_rows = payload.get("participant_history_rows", [])
    conn.execute("DELETE FROM participant_history WHERE match_id = ?", (match_id,))
    _insert_participant_history_rows(conn, participant_rows)


def rebuild_participant_history(
    conn: sqlite3.Connection,
    *,
    progress_callback=None,
    progress_update_every=1000,
):
    ensure_participant_history_schema(conn)
    total_rows = conn.execute(
        "SELECT count(*) FROM matches WHERE raw_match_json IS NOT NULL"
    ).fetchone()[0]
    conn.execute("DELETE FROM participant_history")
    matches_processed = 0
    participants_inserted = 0
    started_at = time.time()
    buffered_rows = []
    auxiliary_rows = []
    if progress_callback:
        progress_callback(
            processed=0,
            total=total_rows,
            matches_processed=0,
            participants_inserted=0,
            started_at=started_at,
            force=False,
        )

    _drop_participant_history_indexes(conn)
    try:
        rows = conn.execute(
            """
            SELECT raw_match_json
            FROM matches
            WHERE raw_match_json IS NOT NULL
            ORDER BY COALESCE(game_creation, 0), match_id
            """
        )

        for index, (raw_json,) in enumerate(rows, start=1):
            match = json.loads(raw_json)
            participant_rows = extract_participant_history_rows(match)
            auxiliary_outcomes = extract_auxiliary_outcome_features(match)
            if not participant_rows:
                if progress_callback and (index % progress_update_every == 0 or index == total_rows):
                    progress_callback(
                        processed=index,
                        total=total_rows,
                        matches_processed=matches_processed,
                        participants_inserted=participants_inserted,
                        started_at=started_at,
                        force=index == total_rows,
                    )
                continue
            participants_inserted += len(participant_rows)
            matches_processed += 1
            buffered_rows.extend(participant_rows)
            auxiliary_rows.append(
                (
                    auxiliary_outcomes["blue_first_blood"],
                    auxiliary_outcomes["blue_first_tower"],
                    auxiliary_outcomes["blue_dragon_share"],
                    auxiliary_outcomes["blue_gold_share"],
                    match.get("metadata", {}).get("matchId"),
                )
            )
            if len(buffered_rows) >= PARTICIPANT_HISTORY_BACKFILL_BATCH_ROWS:
                _insert_participant_history_rows(conn, buffered_rows)
                buffered_rows.clear()
            if len(auxiliary_rows) >= PARTICIPANT_HISTORY_BACKFILL_BATCH_ROWS:
                conn.executemany(
                    """
                    UPDATE matches
                    SET blue_first_blood = ?,
                        blue_first_tower = ?,
                        blue_dragon_share = ?,
                        blue_gold_share = ?
                    WHERE match_id = ?
                    """,
                    auxiliary_rows,
                )
                auxiliary_rows.clear()

            if progress_callback and (index % progress_update_every == 0 or index == total_rows):
                progress_callback(
                    processed=index,
                    total=total_rows,
                    matches_processed=matches_processed,
                    participants_inserted=participants_inserted,
                    started_at=started_at,
                    force=index == total_rows,
                )

        if buffered_rows:
            _insert_participant_history_rows(conn, buffered_rows)
        if auxiliary_rows:
            conn.executemany(
                """
                UPDATE matches
                SET blue_first_blood = ?,
                    blue_first_tower = ?,
                    blue_dragon_share = ?,
                    blue_gold_share = ?
                WHERE match_id = ?
                """,
                auxiliary_rows,
            )
    finally:
        _create_participant_history_indexes(conn)

    return matches_processed, participants_inserted
