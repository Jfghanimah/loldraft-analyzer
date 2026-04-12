import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from ml.data.match_format import ROLE_ORDER, try_build_ordered_match_record, try_build_ordered_participant_record


COMPACT_DATASET_SCHEMA_VERSION = 1
COMPACT_DATA_SOURCE = "riot_api_match_v5_compact"
MATCH_CHAMPION_COLUMNS = tuple(f"champion_{slot}" for slot in range(10))
MATCH_CHAMPION_ID_COLUMNS = tuple(f"champion_id_{slot}" for slot in range(10))
MATCH_COLUMNS = (
    "dataset_schema_version",
    "match_id",
    "platform",
    "queue_id",
    "game_version",
    "patch_major",
    "patch_minor",
    "game_creation",
    "game_end_timestamp",
    "game_date",
    "duration_minutes",
    "label",
    "blue_win",
    "blue_side",
    "position_source",
    *MATCH_CHAMPION_COLUMNS,
    *MATCH_CHAMPION_ID_COLUMNS,
    "blue_first_blood",
    "blue_first_tower",
    "blue_dragon_share",
    "blue_gold_share",
    "blue_dragons",
    "red_dragons",
    "blue_barons",
    "red_barons",
    "blue_rift_heralds",
    "red_rift_heralds",
    "blue_towers",
    "red_towers",
    "blue_inhibitors",
    "red_inhibitors",
    "gold_diff",
    "collector_id",
    "data_source",
    "collected_at_ts",
)
PARTICIPANT_COLUMNS = (
    "dataset_schema_version",
    "match_id",
    "platform",
    "queue_id",
    "game_creation",
    "game_date",
    "game_version",
    "duration_minutes",
    "slot",
    "side",
    "role",
    "team_id",
    "puuid",
    "champion_name",
    "champion_id",
    "win",
    "kills",
    "deaths",
    "assists",
    "vision_score",
    "damage_to_champions",
    "healing",
    "gold_earned",
    "cs",
    "kda_value",
    "dpm_value",
    "gpm_value",
    "cspm_value",
    "vspm_value",
    "hpm_value",
    "collector_id",
    "data_source",
    "collected_at_ts",
)


@dataclass(frozen=True)
class CompactMatchBatch:
    match: dict
    participants: list[dict]


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_patch_parts(game_version):
    parts = str(game_version or "").split(".")
    return _safe_int(parts[0] if len(parts) > 0 else None), _safe_int(parts[1] if len(parts) > 1 else None)


def game_date_from_creation(game_creation):
    timestamp_ms = _safe_int(game_creation)
    if timestamp_ms <= 0:
        return "unknown"
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).date().isoformat()


def _objective_kills(team, objective_name):
    return _safe_int(team.get("objectives", {}).get(objective_name, {}).get("kills", 0))


def _objective_first(team, objective_name):
    return int(bool(team.get("objectives", {}).get(objective_name, {}).get("first", False)))


def _participant_cs(participant):
    return _safe_float(participant.get("totalMinionsKilled")) + _safe_float(participant.get("neutralMinionsKilled"))


def _participant_healing(participant):
    return _safe_float(participant.get("totalHeal")) + _safe_float(participant.get("totalHealsOnTeammates"))


def _duration_minutes(game_creation, game_end_timestamp):
    return max(float((_safe_int(game_end_timestamp) - _safe_int(game_creation)) / 60_000.0), 1.0)


def _team_gold(participants, team_id):
    return sum(_safe_float(participant.get("goldEarned")) for participant in participants if participant.get("teamId") == team_id)


def _normalize_record(record, columns):
    return {column: record.get(column) for column in columns}


def normalize_match_record(record):
    return _normalize_record(record, MATCH_COLUMNS)


def normalize_participant_record(record):
    return _normalize_record(record, PARTICIPANT_COLUMNS)


def extract_compact_records(match, platform, collector_id="", collected_at_ts=None):
    info = match.get("info", {})
    metadata = match.get("metadata", {})
    ordered_match, reason = try_build_ordered_match_record(info)
    if ordered_match is None:
        return None, reason
    ordered_participants, reason = try_build_ordered_participant_record(info)
    if ordered_participants is None:
        return None, reason

    match_id = metadata.get("matchId", "")
    queue_id = _safe_int(info.get("queueId"))
    game_version = info.get("gameVersion", "")
    game_creation = _safe_int(info.get("gameCreation"))
    game_end_timestamp = _safe_int(info.get("gameEndTimestamp"))
    duration_minutes = _duration_minutes(game_creation, game_end_timestamp)
    game_date = game_date_from_creation(game_creation)
    patch_major, patch_minor = _parse_patch_parts(game_version)
    teams = {team.get("teamId"): team for team in info.get("teams", [])}
    blue_team = teams.get(100, {})
    red_team = teams.get(200, {})
    participants = info.get("participants", [])
    blue_gold = _team_gold(participants, 100)
    red_gold = _team_gold(participants, 200)
    total_gold = blue_gold + red_gold
    blue_dragons = _objective_kills(blue_team, "dragon")
    red_dragons = _objective_kills(red_team, "dragon")
    total_dragons = blue_dragons + red_dragons
    collected_at_ts = _safe_int(collected_at_ts, int(time.time()))

    match_record = {
        "dataset_schema_version": COMPACT_DATASET_SCHEMA_VERSION,
        "match_id": match_id,
        "platform": str(platform or "").lower(),
        "queue_id": queue_id,
        "game_version": game_version,
        "patch_major": patch_major,
        "patch_minor": patch_minor,
        "game_creation": game_creation,
        "game_end_timestamp": game_end_timestamp,
        "game_date": game_date,
        "duration_minutes": duration_minutes,
        "label": int(bool(ordered_match["blue_win"])),
        "blue_win": int(bool(ordered_match["blue_win"])),
        "blue_side": int(ordered_match.get("blue_side", 1)),
        "position_source": ordered_match.get("position_source", ""),
        "blue_first_blood": _objective_first(blue_team, "champion"),
        "blue_first_tower": _objective_first(blue_team, "tower"),
        "blue_dragon_share": float(blue_dragons / total_dragons) if total_dragons > 0 else 0.5,
        "blue_gold_share": float(blue_gold / total_gold) if total_gold > 0 else 0.5,
        "blue_dragons": blue_dragons,
        "red_dragons": red_dragons,
        "blue_barons": _objective_kills(blue_team, "baron"),
        "red_barons": _objective_kills(red_team, "baron"),
        "blue_rift_heralds": _objective_kills(blue_team, "riftHerald"),
        "red_rift_heralds": _objective_kills(red_team, "riftHerald"),
        "blue_towers": _objective_kills(blue_team, "tower"),
        "red_towers": _objective_kills(red_team, "tower"),
        "blue_inhibitors": _objective_kills(blue_team, "inhibitor"),
        "red_inhibitors": _objective_kills(red_team, "inhibitor"),
        "gold_diff": float(blue_gold - red_gold),
        "collector_id": collector_id or "",
        "data_source": COMPACT_DATA_SOURCE,
        "collected_at_ts": collected_at_ts,
    }
    for slot, participant in enumerate(ordered_participants):
        match_record[MATCH_CHAMPION_COLUMNS[slot]] = participant.get("championName", "")
        match_record[MATCH_CHAMPION_ID_COLUMNS[slot]] = _safe_int(participant.get("championId"))

    participant_records = []
    for slot, participant in enumerate(ordered_participants):
        team_id = _safe_int(participant.get("teamId"))
        side = "blue" if team_id == 100 else "red"
        role = ROLE_ORDER[slot % len(ROLE_ORDER)]
        kills = _safe_int(participant.get("kills"))
        deaths = _safe_int(participant.get("deaths"))
        assists = _safe_int(participant.get("assists"))
        vision_score = _safe_float(participant.get("visionScore"))
        damage_to_champions = _safe_float(participant.get("totalDamageDealtToChampions"))
        healing = _participant_healing(participant)
        gold_earned = _safe_float(participant.get("goldEarned"))
        cs = _participant_cs(participant)
        participant_records.append(
            {
                "dataset_schema_version": COMPACT_DATASET_SCHEMA_VERSION,
                "match_id": match_id,
                "platform": str(platform or "").lower(),
                "queue_id": queue_id,
                "game_creation": game_creation,
                "game_date": game_date,
                "game_version": game_version,
                "duration_minutes": duration_minutes,
                "slot": slot,
                "side": side,
                "role": role,
                "team_id": team_id,
                "puuid": participant.get("puuid", ""),
                "champion_name": participant.get("championName", ""),
                "champion_id": _safe_int(participant.get("championId")),
                "win": int(bool(teams.get(team_id, {}).get("win", False))),
                "kills": kills,
                "deaths": deaths,
                "assists": assists,
                "vision_score": vision_score,
                "damage_to_champions": damage_to_champions,
                "healing": healing,
                "gold_earned": gold_earned,
                "cs": cs,
                "kda_value": float((kills + assists) / max(1, deaths)),
                "dpm_value": damage_to_champions / duration_minutes,
                "gpm_value": gold_earned / duration_minutes,
                "cspm_value": cs / duration_minutes,
                "vspm_value": vision_score / duration_minutes,
                "hpm_value": healing / duration_minutes,
                "collector_id": collector_id or "",
                "data_source": COMPACT_DATA_SOURCE,
                "collected_at_ts": collected_at_ts,
            }
        )

    return CompactMatchBatch(
        match=normalize_match_record(match_record),
        participants=[normalize_participant_record(record) for record in participant_records],
    ), None


def match_record_from_ordered_json_row(row):
    (
        match_id,
        ordered_match_json,
        platform,
        game_version,
        queue_id,
        game_creation,
        game_end_timestamp,
        blue_first_blood,
        blue_first_tower,
        blue_dragon_share,
        blue_gold_share,
        blue_dragons,
        red_dragons,
        gold_diff,
        game_length_minutes,
        collector_id,
        last_updated_ts,
    ) = row
    ordered = json.loads(ordered_match_json)
    patch_major, patch_minor = _parse_patch_parts(game_version or ordered.get("game_version", ""))
    record = {
        "dataset_schema_version": COMPACT_DATASET_SCHEMA_VERSION,
        "match_id": match_id,
        "platform": str(platform or "").lower(),
        "queue_id": _safe_int(queue_id),
        "game_version": game_version or ordered.get("game_version", ""),
        "patch_major": patch_major,
        "patch_minor": patch_minor,
        "game_creation": _safe_int(game_creation),
        "game_end_timestamp": _safe_int(game_end_timestamp),
        "game_date": game_date_from_creation(game_creation),
        "duration_minutes": _safe_float(game_length_minutes, _duration_minutes(game_creation, game_end_timestamp)),
        "label": int(bool(ordered.get("blue_win"))),
        "blue_win": int(bool(ordered.get("blue_win"))),
        "blue_side": _safe_int(ordered.get("blue_side", 1), 1),
        "position_source": ordered.get("position_source", ""),
        "blue_first_blood": _safe_int(blue_first_blood),
        "blue_first_tower": _safe_int(blue_first_tower),
        "blue_dragon_share": _safe_float(blue_dragon_share, 0.5),
        "blue_gold_share": _safe_float(blue_gold_share, 0.5),
        "blue_dragons": _safe_int(blue_dragons),
        "red_dragons": _safe_int(red_dragons),
        "blue_barons": 0,
        "red_barons": 0,
        "blue_rift_heralds": 0,
        "red_rift_heralds": 0,
        "blue_towers": 0,
        "red_towers": 0,
        "blue_inhibitors": 0,
        "red_inhibitors": 0,
        "gold_diff": _safe_float(gold_diff),
        "collector_id": collector_id or "",
        "data_source": COMPACT_DATA_SOURCE,
        "collected_at_ts": _safe_int(last_updated_ts, int(time.time())),
    }
    champions = list(ordered.get("champions", []))
    if len(champions) != 10:
        raise ValueError(f"Expected 10 champion slots for match_id={match_id}")
    for slot, champion_name in enumerate(champions):
        record[MATCH_CHAMPION_COLUMNS[slot]] = champion_name
        record[MATCH_CHAMPION_ID_COLUMNS[slot]] = 0
    return normalize_match_record(record)


def participant_record_from_history_row(row):
    (
        match_id,
        platform,
        queue_id,
        game_creation,
        game_end_timestamp,
        game_version,
        puuid,
        champion_name,
        role,
        team_id,
        win,
        kills,
        deaths,
        assists,
        vision_score,
        damage_to_champions,
        healing,
        gold_earned,
        cs,
    ) = row
    duration_minutes = _duration_minutes(game_creation, game_end_timestamp)
    side_offset = 0 if _safe_int(team_id) == 100 else len(ROLE_ORDER)
    try:
        slot = side_offset + ROLE_ORDER.index(role)
    except ValueError:
        slot = side_offset
    kills = _safe_int(kills)
    deaths = _safe_int(deaths)
    assists = _safe_int(assists)
    vision_score = _safe_float(vision_score)
    damage_to_champions = _safe_float(damage_to_champions)
    healing = _safe_float(healing)
    gold_earned = _safe_float(gold_earned)
    cs = _safe_float(cs)
    return normalize_participant_record(
        {
            "dataset_schema_version": COMPACT_DATASET_SCHEMA_VERSION,
            "match_id": match_id,
            "platform": str(platform or "").lower(),
            "queue_id": _safe_int(queue_id),
            "game_creation": _safe_int(game_creation),
            "game_date": game_date_from_creation(game_creation),
            "game_version": game_version or "",
            "duration_minutes": duration_minutes,
            "slot": slot,
            "side": "blue" if _safe_int(team_id) == 100 else "red",
            "role": role or "",
            "team_id": _safe_int(team_id),
            "puuid": puuid or "",
            "champion_name": champion_name or "",
            "champion_id": 0,
            "win": _safe_int(win),
            "kills": kills,
            "deaths": deaths,
            "assists": assists,
            "vision_score": vision_score,
            "damage_to_champions": damage_to_champions,
            "healing": healing,
            "gold_earned": gold_earned,
            "cs": cs,
            "kda_value": float((kills + assists) / max(1, deaths)),
            "dpm_value": damage_to_champions / duration_minutes,
            "gpm_value": gold_earned / duration_minutes,
            "cspm_value": cs / duration_minutes,
            "vspm_value": vision_score / duration_minutes,
            "hpm_value": healing / duration_minutes,
            "collector_id": "",
            "data_source": COMPACT_DATA_SOURCE,
            "collected_at_ts": 0,
        }
    )
