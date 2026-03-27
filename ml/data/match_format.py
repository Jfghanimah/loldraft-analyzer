ROLE_ORDER = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
VALID_POSITIONS = set(ROLE_ORDER)

PLATFORM_TO_REGION = {
    "BR1": "americas",
    "LA1": "americas",
    "LA2": "americas",
    "NA1": "americas",
    "OC1": "sea",
    "PH2": "sea",
    "SG2": "sea",
    "TH2": "sea",
    "TW2": "sea",
    "VN2": "sea",
    "EUN1": "europe",
    "EUW1": "europe",
    "ME1": "europe",
    "TR1": "europe",
    "RU": "europe",
    "JP1": "asia",
    "KR": "asia",
}


def normalize_position(team_position, individual_position=None):
    for value in (team_position, individual_position):
        if isinstance(value, str):
            upper = value.upper()
            if upper in VALID_POSITIONS:
                return upper
    return None


def is_ordered_match_record(match_data):
    return isinstance(match_data, dict) and match_data.get("format") == "role_order_v1"


def get_record_champions(match_data):
    if is_ordered_match_record(match_data):
        return match_data["champions"]
    raise ValueError("Unsupported match record format.")


def get_record_training_row(match_data):
    if is_ordered_match_record(match_data):
        champions = match_data["champions"]
        blue_side = int(match_data.get("blue_side", 1))
        return [int(bool(match_data["blue_win"]))] + champions + [blue_side]

    return None


def derive_regional_route(match_id):
    platform = match_id.split("_", 1)[0].upper()
    route = PLATFORM_TO_REGION.get(platform)
    if route is None:
        raise ValueError(f"Unsupported platform prefix in match ID '{match_id}'.")
    return route


def try_build_ordered_match_record(info):
    teams = {100: {}, 200: {}}
    position_sources = set()

    for participant in info["participants"]:
        team_id = participant.get("teamId")
        if team_id not in teams:
            return None, f"unexpected teamId {team_id}"

        position = normalize_position(
            participant.get("teamPosition"),
            participant.get("individualPosition"),
        )
        if position is None:
            return None, f"missing valid position for champion {participant.get('championName')}"

        if position in teams[team_id]:
            return None, f"duplicate position {position} on team {team_id}"

        source = "teamPosition" if normalize_position(participant.get("teamPosition")) else "individualPosition"
        position_sources.add(source)
        teams[team_id][position] = participant["championName"]

    expected_positions = set(ROLE_ORDER)
    for team_id, picks in teams.items():
        if set(picks) != expected_positions:
            return None, f"incomplete role coverage on team {team_id}: {sorted(picks)}"

    team_results = {team["teamId"]: bool(team["win"]) for team in info["teams"]}
    if 100 not in team_results:
        return None, "missing blue-side team result"

    ordered_champions = [teams[100][role] for role in ROLE_ORDER] + [teams[200][role] for role in ROLE_ORDER]
    return {
        "format": "role_order_v1",
        "blue_win": team_results[100],
        "blue_side": 1,
        "champions": ordered_champions,
        "role_order": list(ROLE_ORDER),
        "position_source": "teamPosition" if position_sources == {"teamPosition"} else "mixed",
        "game_version": info.get("gameVersion", ""),
        "queue_id": info.get("queueId"),
    }, None


def try_build_ordered_participant_record(info):
    teams = {100: {}, 200: {}}

    for participant in info["participants"]:
        team_id = participant.get("teamId")
        if team_id not in teams:
            return None, f"unexpected teamId {team_id}"

        position = normalize_position(
            participant.get("teamPosition"),
            participant.get("individualPosition"),
        )
        if position is None:
            return None, f"missing valid position for champion {participant.get('championName')}"

        if position in teams[team_id]:
            return None, f"duplicate position {position} on team {team_id}"

        teams[team_id][position] = participant

    expected_positions = set(ROLE_ORDER)
    for team_id, picks in teams.items():
        if set(picks) != expected_positions:
            return None, f"incomplete role coverage on team {team_id}: {sorted(picks)}"

    ordered_participants = [teams[100][role] for role in ROLE_ORDER] + [teams[200][role] for role in ROLE_ORDER]
    return ordered_participants, None
