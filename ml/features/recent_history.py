import math
from collections import deque


QUEUE_ID_SOLO = 420
RECENT_MATCH_LIMIT = 20
RECENT_WINDOWS_MS = {
    "games_last_3d": 3 * 24 * 60 * 60 * 1000,
    "games_last_7d": 7 * 24 * 60 * 60 * 1000,
}
HOURS_SINCE_LAST_CAP = 14 * 24.0

PARTICIPANT_FEATURES = (
    "games_played",
    "win_rate",
    "champ_games",
    "champ_win_rate",
    "role_games",
    "role_win_rate",
    "avg_kda",
    "games_last_3d",
    "avg_game_length_minutes",
    "avg_dpm",
    "avg_gpm",
    "avg_cspm",
    "avg_vspm",
    "avg_hpm",
)

GLOBAL_FEATURES = ("patch_major", "patch_minor")


def _normalize_count(value, scale=5.0):
    return min(math.log1p(value) / scale, 1.0)


def _normalize_rate(num, den):
    return float(num) / float(den) if den else 0.5


def _normalize_avg(total, count, scale):
    return min(total / count, scale) / scale if count else 0.0


def _normalize_hours_since_last(hours):
    if hours is None:
        return 1.0
    return min(hours / HOURS_SINCE_LAST_CAP, 1.0)


def _duration_minutes(row):
    return max(float(row.get("duration_minutes", 0.0) or 0.0), 1.0)


def _aggregate_recent_history(prior_rows, champion_name, role, current_game_creation):
    rows = list(prior_rows)[:RECENT_MATCH_LIMIT]
    total_games = len(rows)
    wins = sum(int(row["win"]) for row in rows)
    champ_rows = [row for row in rows if row["champion_name"] == champion_name]
    role_rows = [row for row in rows if row["role"] == role]
    games_last_3d = 0
    avg_game_length = _normalize_avg(sum(_duration_minutes(row) for row in rows), total_games, 45.0)
    avg_dpm = _normalize_avg(
        sum(float(row["damage_to_champions"]) / _duration_minutes(row) for row in rows),
        total_games,
        1500.0,
    )
    avg_gpm = _normalize_avg(
        sum(float(row.get("gold_earned", 0.0)) / _duration_minutes(row) for row in rows),
        total_games,
        900.0,
    )
    avg_cspm = _normalize_avg(
        sum(float(row.get("cs", 0.0)) / _duration_minutes(row) for row in rows),
        total_games,
        15.0,
    )
    avg_vspm = _normalize_avg(
        sum(float(row["vision_score"]) / _duration_minutes(row) for row in rows),
        total_games,
        6.0,
    )
    avg_hpm = _normalize_avg(
        sum(float(row["healing"]) / _duration_minutes(row) for row in rows),
        total_games,
        1200.0,
    )

    if current_game_creation:
        for row in rows:
            delta = max(0, int(current_game_creation) - int(row["game_creation"]))
            if delta <= RECENT_WINDOWS_MS["games_last_3d"]:
                games_last_3d += 1

    return [
        _normalize_count(total_games),
        _normalize_rate(wins, total_games),
        _normalize_count(len(champ_rows)),
        _normalize_rate(sum(int(row["win"]) for row in champ_rows), len(champ_rows)),
        _normalize_count(len(role_rows)),
        _normalize_rate(sum(int(row["win"]) for row in role_rows), len(role_rows)),
        _normalize_avg(sum((row["kills"] + row["assists"]) / max(1, row["deaths"]) for row in rows), total_games, 10.0),
        _normalize_count(games_last_3d),
        avg_game_length,
        avg_dpm,
        avg_gpm,
        avg_cspm,
        avg_vspm,
        avg_hpm,
    ]


def build_player_recent_feature_vector(prior_rows, champion_name, role, current_game_creation):
    return _aggregate_recent_history(prior_rows, champion_name, role, current_game_creation)


def parse_patch(game_version):
    if not game_version:
        return 0.0, 0.0
    parts = str(game_version).split(".")
    try:
        major = int(parts[0]) / 20.0
    except (ValueError, IndexError):
        major = 0.0
    try:
        minor = int(parts[1]) / 30.0
    except (ValueError, IndexError):
        minor = 0.0
    return major, minor


def dense_feature_columns(role_order):
    columns = []
    for slot in range(10):
        prefix = f"slot_{slot}_{'blue' if slot < 5 else 'red'}_{role_order[slot % 5].lower()}"
        for feature_name in PARTICIPANT_FEATURES:
            columns.append(f"{prefix}_{feature_name}")
    columns.extend(GLOBAL_FEATURES)
    return columns


class RecentHistoryStore:
    def __init__(self, limit=RECENT_MATCH_LIMIT):
        self.limit = limit
        self._rows_by_puuid = {}

    def feature_vector(self, puuid, champion_name, role, current_game_creation):
        history = self._rows_by_puuid.get(puuid)
        if not history:
            return build_player_recent_feature_vector([], champion_name, role, current_game_creation)
        return build_player_recent_feature_vector(history, champion_name, role, current_game_creation)

    def add_match_rows(self, rows):
        for row in rows:
            history = self._rows_by_puuid.setdefault(row["puuid"], deque(maxlen=self.limit))
            history.appendleft(row)
