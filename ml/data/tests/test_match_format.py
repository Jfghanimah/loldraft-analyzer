from ml.data.match_format import derive_regional_route, try_build_ordered_match_record


def test_try_build_ordered_match_record_sorts_participants_by_team_position():
    info = {
        "queueId": 420,
        "gameVersion": "15.1.1",
        "teams": [
            {"teamId": 100, "win": True},
            {"teamId": 200, "win": False},
        ],
        "participants": [
            {"teamId": 100, "championName": "Ashe", "teamPosition": "BOTTOM", "individualPosition": "BOTTOM"},
            {"teamId": 200, "championName": "LeeSin", "teamPosition": "JUNGLE", "individualPosition": "JUNGLE"},
            {"teamId": 100, "championName": "Aatrox", "teamPosition": "TOP", "individualPosition": "TOP"},
            {"teamId": 200, "championName": "Lux", "teamPosition": "MIDDLE", "individualPosition": "MIDDLE"},
            {"teamId": 100, "championName": "Ahri", "teamPosition": "MIDDLE", "individualPosition": "MIDDLE"},
            {"teamId": 200, "championName": "Jinx", "teamPosition": "BOTTOM", "individualPosition": "BOTTOM"},
            {"teamId": 100, "championName": "Amumu", "teamPosition": "JUNGLE", "individualPosition": "JUNGLE"},
            {"teamId": 100, "championName": "Braum", "teamPosition": "UTILITY", "individualPosition": "UTILITY"},
            {"teamId": 200, "championName": "Renekton", "teamPosition": "TOP", "individualPosition": "TOP"},
            {"teamId": 200, "championName": "Nami", "teamPosition": "UTILITY", "individualPosition": "UTILITY"},
        ],
    }

    record, reason = try_build_ordered_match_record(info)

    assert reason is None
    assert record["blue_win"] is True
    assert record["champions"] == [
        "Aatrox",
        "Amumu",
        "Ahri",
        "Ashe",
        "Braum",
        "Renekton",
        "LeeSin",
        "Lux",
        "Jinx",
        "Nami",
    ]


def test_try_build_ordered_match_record_rejects_duplicate_role():
    info = {
        "queueId": 420,
        "gameVersion": "15.1.1",
        "teams": [
            {"teamId": 100, "win": True},
            {"teamId": 200, "win": False},
        ],
        "participants": [
            {"teamId": 100, "championName": "Aatrox", "teamPosition": "TOP", "individualPosition": "TOP"},
            {"teamId": 100, "championName": "Darius", "teamPosition": "TOP", "individualPosition": "TOP"},
            {"teamId": 100, "championName": "Ahri", "teamPosition": "MIDDLE", "individualPosition": "MIDDLE"},
            {"teamId": 100, "championName": "Ashe", "teamPosition": "BOTTOM", "individualPosition": "BOTTOM"},
            {"teamId": 100, "championName": "Braum", "teamPosition": "UTILITY", "individualPosition": "UTILITY"},
            {"teamId": 200, "championName": "Renekton", "teamPosition": "TOP", "individualPosition": "TOP"},
            {"teamId": 200, "championName": "LeeSin", "teamPosition": "JUNGLE", "individualPosition": "JUNGLE"},
            {"teamId": 200, "championName": "Lux", "teamPosition": "MIDDLE", "individualPosition": "MIDDLE"},
            {"teamId": 200, "championName": "Jinx", "teamPosition": "BOTTOM", "individualPosition": "BOTTOM"},
            {"teamId": 200, "championName": "Nami", "teamPosition": "UTILITY", "individualPosition": "UTILITY"},
        ],
    }

    record, reason = try_build_ordered_match_record(info)

    assert record is None
    assert "duplicate position TOP" in reason


def test_derive_regional_route_uses_match_id_prefix():
    assert derive_regional_route("NA1_123") == "americas"
    assert derive_regional_route("EUW1_123") == "europe"
    assert derive_regional_route("KR_123") == "asia"
