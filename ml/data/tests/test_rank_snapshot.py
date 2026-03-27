import json
import sqlite3

from ml.data.match_storage import ensure_rank_snapshot_schema
from ml.data.rank_snapshot import extract_solo_queue_snapshot, get_or_fetch_rank_snapshots


class FakeSummonerApi:
    def __init__(self):
        self.calls = []

    def by_puuid(self, platform, puuid):
        self.calls.append((platform, puuid))
        return {"id": f"summ-{puuid}"}


class FakeLeagueApi:
    def __init__(self):
        self.calls = []

    def by_summoner(self, platform, summoner_id):
        self.calls.append((platform, summoner_id))
        return [
            {
                "queueType": "RANKED_SOLO_5x5",
                "tier": "EMERALD",
                "rank": "II",
                "leaguePoints": 77,
                "wins": 12,
                "losses": 9,
            }
        ]


class FakeWatcher:
    def __init__(self):
        self.summoner = FakeSummonerApi()
        self.league = FakeLeagueApi()


class MissingIdSummonerApi:
    def __init__(self):
        self.calls = []

    def by_puuid(self, platform, puuid):
        self.calls.append((platform, puuid))
        return {"puuid": puuid}


class MissingIdWatcher:
    def __init__(self):
        self.summoner = MissingIdSummonerApi()
        self.league = FakeLeagueApi()


def test_extract_solo_queue_snapshot_prefers_ranked_entry():
    snapshot = extract_solo_queue_snapshot(
        [
            {"queueType": "RANKED_FLEX_SR", "tier": "GOLD"},
            {"queueType": "RANKED_SOLO_5x5", "tier": "EMERALD", "rank": "III", "leaguePoints": 44, "wins": 9, "losses": 5},
        ],
        fetched_ts=123,
    )

    assert snapshot["status"] == "ranked"
    assert snapshot["tier"] == "EMERALD"
    assert snapshot["rank"] == "III"
    assert snapshot["snapshot_fetched_ts"] == 123


def test_get_or_fetch_rank_snapshots_uses_cache_after_first_fetch():
    conn = sqlite3.connect(":memory:")
    ensure_rank_snapshot_schema(conn)
    watcher = FakeWatcher()
    participants = [
        {"puuid": "p1", "teamId": 100, "teamPosition": "TOP", "championName": "Aatrox"},
        {"puuid": "p2", "teamId": 100, "teamPosition": "JUNGLE", "championName": "Amumu"},
    ]

    first = get_or_fetch_rank_snapshots(conn, watcher, "na1", participants, ttl_seconds=999999)
    second = get_or_fetch_rank_snapshots(conn, watcher, "na1", participants, ttl_seconds=999999)

    cached = conn.execute(
        "SELECT puuid, snapshot_json FROM puuid_rank_cache ORDER BY puuid"
    ).fetchall()
    conn.close()

    assert len(first) == 2
    assert len(second) == 2
    assert watcher.summoner.calls == [("na1", "p1"), ("na1", "p2")]
    assert watcher.league.calls == [("na1", "summ-p1"), ("na1", "summ-p2")]
    assert first[0]["tier"] == "EMERALD"
    assert second[1]["tier"] == "EMERALD"
    assert json.loads(cached[0][1])["status"] == "ranked"


def test_get_or_fetch_rank_snapshots_handles_missing_summoner_id():
    conn = sqlite3.connect(":memory:")
    ensure_rank_snapshot_schema(conn)
    watcher = MissingIdWatcher()
    participants = [
        {"puuid": "p1", "teamId": 100, "teamPosition": "TOP", "championName": "Aatrox"},
    ]

    snapshots = get_or_fetch_rank_snapshots(conn, watcher, "na1", participants, ttl_seconds=999999)
    conn.close()

    assert len(snapshots) == 1
    assert snapshots[0]["status"] == "missing_summoner_id"
    assert "returned keys" in snapshots[0]["error_message"]
    assert watcher.league.calls == []


def test_get_or_fetch_rank_snapshots_prefers_participant_summoner_id():
    conn = sqlite3.connect(":memory:")
    ensure_rank_snapshot_schema(conn)
    watcher = FakeWatcher()
    participants = [
        {
            "puuid": "p1",
            "summonerId": "summ-from-match",
            "teamId": 100,
            "teamPosition": "TOP",
            "championName": "Aatrox",
        },
    ]

    snapshots = get_or_fetch_rank_snapshots(conn, watcher, "na1", participants, ttl_seconds=999999)
    conn.close()

    assert len(snapshots) == 1
    assert snapshots[0]["summoner_id"] == "summ-from-match"
    assert watcher.summoner.calls == []
    assert watcher.league.calls == [("na1", "summ-from-match")]
