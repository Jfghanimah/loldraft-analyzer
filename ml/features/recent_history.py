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


def _row_kda(row):
    return float(row.get("kda_value", (row["kills"] + row["assists"]) / max(1, row["deaths"])))


def _row_dpm(row):
    return float(row.get("dpm_value", float(row["damage_to_champions"]) / _duration_minutes(row)))


def _row_gpm(row):
    return float(row.get("gpm_value", float(row.get("gold_earned", 0.0)) / _duration_minutes(row)))


def _row_cspm(row):
    return float(row.get("cspm_value", float(row.get("cs", 0.0)) / _duration_minutes(row)))


def _row_vspm(row):
    return float(row.get("vspm_value", float(row["vision_score"]) / _duration_minutes(row)))


def _row_hpm(row):
    return float(row.get("hpm_value", float(row["healing"]) / _duration_minutes(row)))


def _aggregate_recent_history(prior_rows, champion_name, role, current_game_creation):
    rows = list(prior_rows)[:RECENT_MATCH_LIMIT]
    total_games = len(rows)
    wins = 0
    champ_games = 0
    champ_wins = 0
    role_games = 0
    role_wins = 0
    games_last_3d = 0
    sum_game_length = 0.0
    sum_kda = 0.0
    sum_dpm = 0.0
    sum_gpm = 0.0
    sum_cspm = 0.0
    sum_vspm = 0.0
    sum_hpm = 0.0

    for row in rows:
        wins += int(row["win"])
        if row["champion_name"] == champion_name:
            champ_games += 1
            champ_wins += int(row["win"])
        if row["role"] == role:
            role_games += 1
            role_wins += int(row["win"])
        if current_game_creation:
            delta = max(0, int(current_game_creation) - int(row["game_creation"]))
            if delta <= RECENT_WINDOWS_MS["games_last_3d"]:
                games_last_3d += 1

        sum_game_length += _duration_minutes(row)
        sum_kda += _row_kda(row)
        sum_dpm += _row_dpm(row)
        sum_gpm += _row_gpm(row)
        sum_cspm += _row_cspm(row)
        sum_vspm += _row_vspm(row)
        sum_hpm += _row_hpm(row)

    return [
        _normalize_count(total_games),
        _normalize_rate(wins, total_games),
        _normalize_count(champ_games),
        _normalize_rate(champ_wins, champ_games),
        _normalize_count(role_games),
        _normalize_rate(role_wins, role_games),
        _normalize_avg(sum_kda, total_games, 10.0),
        _normalize_count(games_last_3d),
        _normalize_avg(sum_game_length, total_games, 45.0),
        _normalize_avg(sum_dpm, total_games, 1500.0),
        _normalize_avg(sum_gpm, total_games, 900.0),
        _normalize_avg(sum_cspm, total_games, 15.0),
        _normalize_avg(sum_vspm, total_games, 6.0),
        _normalize_avg(sum_hpm, total_games, 1200.0),
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
        self._histories_by_puuid = {}

    def feature_vector(self, puuid, champion_name, role, current_game_creation):
        history = self._histories_by_puuid.get(puuid)
        if not history:
            return build_player_recent_feature_vector([], champion_name, role, current_game_creation)
        return history.feature_vector(champion_name, role, current_game_creation)

    def add_match_rows(self, rows):
        for row in rows:
            history = self._histories_by_puuid.setdefault(row["puuid"], _PlayerHistoryBuffer(self.limit))
            history.appendleft(row)


class _PlayerHistoryBuffer:
    def __init__(self, limit):
        self.limit = limit
        self.rows = deque()
        self.total_wins = 0
        self.sum_kda = 0.0
        self.sum_game_length = 0.0
        self.sum_dpm = 0.0
        self.sum_gpm = 0.0
        self.sum_cspm = 0.0
        self.sum_vspm = 0.0
        self.sum_hpm = 0.0
        self.champ_counts = {}
        self.champ_wins = {}
        self.role_counts = {}
        self.role_wins = {}

    def _adjust_counter(self, counter, key, delta):
        new_value = counter.get(key, 0) + delta
        if new_value:
            counter[key] = new_value
        else:
            counter.pop(key, None)

    def _apply_row(self, row, sign):
        win = int(row["win"])
        champion_name = row["champion_name"]
        role = row["role"]
        self.total_wins += sign * win
        self.sum_kda += sign * _row_kda(row)
        self.sum_game_length += sign * _duration_minutes(row)
        self.sum_dpm += sign * _row_dpm(row)
        self.sum_gpm += sign * _row_gpm(row)
        self.sum_cspm += sign * _row_cspm(row)
        self.sum_vspm += sign * _row_vspm(row)
        self.sum_hpm += sign * _row_hpm(row)
        self._adjust_counter(self.champ_counts, champion_name, sign)
        self._adjust_counter(self.champ_wins, champion_name, sign * win)
        self._adjust_counter(self.role_counts, role, sign)
        self._adjust_counter(self.role_wins, role, sign * win)

    def appendleft(self, row):
        if len(self.rows) >= self.limit:
            dropped = self.rows.pop()
            self._apply_row(dropped, -1)
        self.rows.appendleft(row)
        self._apply_row(row, 1)

    def feature_vector(self, champion_name, role, current_game_creation):
        total_games = len(self.rows)
        games_last_3d = 0
        if current_game_creation:
            for row in self.rows:
                delta = max(0, int(current_game_creation) - int(row["game_creation"]))
                if delta <= RECENT_WINDOWS_MS["games_last_3d"]:
                    games_last_3d += 1

        champ_games = self.champ_counts.get(champion_name, 0)
        role_games = self.role_counts.get(role, 0)
        return [
            _normalize_count(total_games),
            _normalize_rate(self.total_wins, total_games),
            _normalize_count(champ_games),
            _normalize_rate(self.champ_wins.get(champion_name, 0), champ_games),
            _normalize_count(role_games),
            _normalize_rate(self.role_wins.get(role, 0), role_games),
            _normalize_avg(self.sum_kda, total_games, 10.0),
            _normalize_count(games_last_3d),
            _normalize_avg(self.sum_game_length, total_games, 45.0),
            _normalize_avg(self.sum_dpm, total_games, 1500.0),
            _normalize_avg(self.sum_gpm, total_games, 900.0),
            _normalize_avg(self.sum_cspm, total_games, 15.0),
            _normalize_avg(self.sum_vspm, total_games, 6.0),
            _normalize_avg(self.sum_hpm, total_games, 1200.0),
        ]
