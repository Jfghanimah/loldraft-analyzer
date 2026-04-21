import json
import sqlite3

import pytest
from requests.exceptions import ConnectionError

import ml.data.data_api_sqlite as data_api_sqlite
from ml.data.compact_parquet import CompactParquetWriter
from ml.data.match_storage import ensure_match_schema


class DummyResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class DummyApiError(Exception):
    def __init__(self, status_code):
        self.response = DummyResponse(status_code)


class FakeAccountApi:
    def __init__(self, puuid="seed-puuid"):
        self.puuid = puuid
        self.calls = []

    def by_riot_id(self, region, game_name, tag_line):
        self.calls.append((region, game_name, tag_line))
        return {"puuid": self.puuid}


class FakeMatchApi:
    def __init__(self, match=None, history=None, by_id_error=None, history_error=None):
        self.match_payload = match
        self.history_payload = history or []
        self.by_id_error = by_id_error
        self.history_error = history_error
        self.by_id_calls = []
        self.history_calls = []

    def by_id(self, region, match_id):
        self.by_id_calls.append((region, match_id))
        if self.by_id_error is not None:
            raise self.by_id_error
        return self.match_payload

    def matchlist_by_puuid(self, region, puuid, queue, start_time, count):
        self.history_calls.append((region, puuid, queue, start_time, count))
        if self.history_error is not None:
            raise self.history_error
        return self.history_payload


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
                "tier": "DIAMOND",
                "rank": "IV",
                "leaguePoints": 12,
                "wins": 30,
                "losses": 25,
            }
        ]


class FakeRiotWatcher:
    def __init__(self, account_api):
        self.account = account_api


class FakeLolWatcher:
    def __init__(self, match_api, summoner_api=None, league_api=None):
        self.match = match_api
        self.summoner = summoner_api or FakeSummonerApi()
        self.league = league_api or FakeLeagueApi()


def make_conn():
    conn = sqlite3.connect(":memory:")
    ensure_match_schema(conn)
    data_api_sqlite.ensure_collector_state_schema(conn)
    conn.execute("CREATE TABLE match_queue_na1 (match_id TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE puuid_queue_na1 (puuid TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE processed_puuids_na1 (puuid TEXT PRIMARY KEY)")
    return conn


def sample_match():
    return {
        "metadata": {"matchId": "NA1_1"},
        "info": {
            "queueId": 420,
            "gameVersion": "15.1.1",
            "gameCreation": 111,
            "gameEndTimestamp": 222,
            "teams": [{"teamId": 100, "win": True}, {"teamId": 200, "win": False}],
            "participants": [
                {"teamId": 100, "championName": "Aatrox", "teamPosition": "TOP", "individualPosition": "TOP", "puuid": "p1"},
                {"teamId": 100, "championName": "Amumu", "teamPosition": "JUNGLE", "individualPosition": "JUNGLE", "puuid": "p2"},
                {"teamId": 100, "championName": "Ahri", "teamPosition": "MIDDLE", "individualPosition": "MIDDLE", "puuid": "p3"},
                {"teamId": 100, "championName": "Ashe", "teamPosition": "BOTTOM", "individualPosition": "BOTTOM", "puuid": "p4"},
                {"teamId": 100, "championName": "Braum", "teamPosition": "UTILITY", "individualPosition": "UTILITY", "puuid": "p5"},
                {"teamId": 200, "championName": "Renekton", "teamPosition": "TOP", "individualPosition": "TOP", "puuid": "p6"},
                {"teamId": 200, "championName": "LeeSin", "teamPosition": "JUNGLE", "individualPosition": "JUNGLE", "puuid": "p7"},
                {"teamId": 200, "championName": "Lux", "teamPosition": "MIDDLE", "individualPosition": "MIDDLE", "puuid": "p8"},
                {"teamId": 200, "championName": "Jinx", "teamPosition": "BOTTOM", "individualPosition": "BOTTOM", "puuid": "p9"},
                {"teamId": 200, "championName": "Nami", "teamPosition": "UTILITY", "individualPosition": "UTILITY", "puuid": "p10"},
            ],
        },
    }


@pytest.fixture(autouse=True)
def patch_watchers(monkeypatch):
    account_api = FakeAccountApi()
    match_api = FakeMatchApi(match=sample_match(), history=["NA1_2", "NA1_3"])
    monkeypatch.setattr(data_api_sqlite, "riot_watcher", FakeRiotWatcher(account_api))
    monkeypatch.setattr(data_api_sqlite, "lol_watcher", FakeLolWatcher(match_api))
    yield {"account_api": account_api, "match_api": match_api}


def test_seed_if_needed_adds_seed_player_when_region_queues_are_empty(patch_watchers):
    conn = make_conn()

    data_api_sqlite.seed_if_needed(
        conn,
        {"platform": "na1", "region": "americas", "seed_name": "Seed", "seed_tag": "NA1"},
    )

    rows = conn.execute("SELECT puuid FROM puuid_queue_na1").fetchall()
    conn.close()

    assert rows == [("seed-puuid",)]
    assert patch_watchers["account_api"].calls == [("americas", "Seed", "NA1")]


def test_process_region_fetches_match_stores_raw_and_ordered_data(patch_watchers):
    conn = make_conn()
    conn.execute("INSERT INTO match_queue_na1 VALUES (?)", ("NA1_1",))
    data_api_sqlite.STORAGE_MODE = "sqlite"

    did_work = data_api_sqlite.process_region(
        conn,
        {"platform": "na1", "region": "americas"},
    )

    stored = conn.execute(
        "SELECT region, raw_match_json, ordered_match_json, game_version, queue_id FROM matches WHERE match_id = ?",
        ("NA1_1",),
    ).fetchone()
    participant_rows = conn.execute(
        "SELECT count(*) FROM participant_history WHERE match_id = ?",
        ("NA1_1",),
    ).fetchone()
    puuids = conn.execute("SELECT puuid FROM puuid_queue_na1 ORDER BY puuid").fetchall()
    conn.close()

    assert did_work is True
    assert stored[0] == "na1"
    assert json.loads(stored[1])["metadata"]["matchId"] == "NA1_1"
    assert json.loads(stored[2])["format"] == "role_order_v1"
    assert stored[3] == "15.1.1"
    assert stored[4] == 420
    assert participant_rows[0] == 10
    assert len(puuids) == 10
    assert patch_watchers["match_api"].by_id_calls == [("americas", "NA1_1")]


def test_process_region_compact_mode_marks_seen_without_raw_sqlite_storage(patch_watchers, tmp_path):
    conn = make_conn()
    conn.execute("INSERT INTO match_queue_na1 VALUES (?)", ("NA1_1",))
    data_api_sqlite.STORAGE_MODE = "compact"
    writer = CompactParquetWriter(tmp_path)

    did_work = data_api_sqlite.process_region(
        conn,
        {"platform": "na1", "region": "americas"},
        compact_writer=writer,
    )

    stored = conn.execute(
        "SELECT raw_match_json, ordered_match_json FROM matches WHERE match_id = ?",
        ("NA1_1",),
    ).fetchone()
    seen = conn.execute(
        "SELECT platform, storage_mode FROM seen_matches WHERE match_id = ?",
        ("NA1_1",),
    ).fetchone()
    written = data_api_sqlite.flush_compact_writer(writer)
    conn.close()

    assert did_work is True
    assert stored is None
    assert seen == ("na1", "compact")
    assert {item["kind"] for item in written} == {"matches", "participants"}


def test_process_region_scans_player_and_queues_unseen_matches(patch_watchers):
    conn = make_conn()
    conn.execute("INSERT INTO puuid_queue_na1 VALUES (?)", ("seed-puuid",))
    conn.execute("INSERT INTO matches (match_id, match_data, region) VALUES (?, ?, ?)", ("NA1_2", "{}", "na1"))

    did_work = data_api_sqlite.process_region(
        conn,
        {"platform": "na1", "region": "americas"},
    )

    queued = conn.execute("SELECT match_id FROM match_queue_na1 ORDER BY match_id").fetchall()
    processed = conn.execute("SELECT puuid FROM processed_puuids_na1").fetchall()
    conn.close()

    assert did_work is True
    assert queued == [("NA1_3",)]
    assert processed == [("seed-puuid",)]
    assert patch_watchers["match_api"].history_calls == [
        ("americas", "seed-puuid", data_api_sqlite.QUEUE_ID, data_api_sqlite.START_TIME, 100)
    ]


def test_process_region_requeues_match_after_429(monkeypatch):
    conn = make_conn()
    conn.execute("INSERT INTO match_queue_na1 VALUES (?)", ("NA1_9",))

    match_api = FakeMatchApi(by_id_error=DummyApiError(429))
    monkeypatch.setattr(data_api_sqlite, "lol_watcher", FakeLolWatcher(match_api))
    monkeypatch.setattr(data_api_sqlite, "riot_watcher", FakeRiotWatcher(FakeAccountApi()))
    monkeypatch.setattr(data_api_sqlite, "ApiError", DummyApiError)

    did_work = data_api_sqlite.process_region(
        conn,
        {"platform": "na1", "region": "americas"},
    )

    queued = conn.execute("SELECT match_id FROM match_queue_na1").fetchall()
    conn.close()

    assert did_work is False
    assert queued == [("NA1_9",)]


def test_process_region_stops_and_requeues_match_after_auth_error(monkeypatch):
    conn = make_conn()
    conn.execute("INSERT INTO match_queue_na1 VALUES (?)", ("NA1_9",))

    match_api = FakeMatchApi(by_id_error=DummyApiError(401))
    monkeypatch.setattr(data_api_sqlite, "lol_watcher", FakeLolWatcher(match_api))
    monkeypatch.setattr(data_api_sqlite, "riot_watcher", FakeRiotWatcher(FakeAccountApi()))
    monkeypatch.setattr(data_api_sqlite, "ApiError", DummyApiError)
    monkeypatch.setattr(data_api_sqlite, "keep_running", True)

    did_work = data_api_sqlite.process_region(
        conn,
        {"platform": "na1", "region": "americas"},
    )

    queued = conn.execute("SELECT match_id FROM match_queue_na1").fetchall()
    conn.close()

    assert did_work is False
    assert data_api_sqlite.keep_running is False
    assert queued == [("NA1_9",)]


def test_process_region_requeues_match_after_transient_network_error(monkeypatch):
    conn = make_conn()
    conn.execute("INSERT INTO match_queue_na1 VALUES (?)", ("NA1_9",))

    match_api = FakeMatchApi(by_id_error=ConnectionError("remote closed"))
    monkeypatch.setattr(data_api_sqlite, "lol_watcher", FakeLolWatcher(match_api))
    monkeypatch.setattr(data_api_sqlite, "riot_watcher", FakeRiotWatcher(FakeAccountApi()))
    monkeypatch.setattr(data_api_sqlite.time, "sleep", lambda _seconds: None)

    did_work = data_api_sqlite.process_region(
        conn,
        {"platform": "na1", "region": "americas"},
    )

    queued = conn.execute("SELECT match_id FROM match_queue_na1").fetchall()
    conn.close()

    assert did_work is False
    assert queued == [("NA1_9",)]


def test_process_region_requeues_player_after_transient_network_error(monkeypatch):
    conn = make_conn()
    conn.execute("INSERT INTO puuid_queue_na1 VALUES (?)", ("seed-puuid",))

    match_api = FakeMatchApi(history_error=ConnectionError("remote closed"))
    monkeypatch.setattr(data_api_sqlite, "lol_watcher", FakeLolWatcher(match_api))
    monkeypatch.setattr(data_api_sqlite, "riot_watcher", FakeRiotWatcher(FakeAccountApi()))
    monkeypatch.setattr(data_api_sqlite.time, "sleep", lambda _seconds: None)

    did_work = data_api_sqlite.process_region(
        conn,
        {"platform": "na1", "region": "americas"},
    )

    queued = conn.execute("SELECT puuid FROM puuid_queue_na1").fetchall()
    processed = conn.execute("SELECT puuid FROM processed_puuids_na1").fetchall()
    conn.close()

    assert did_work is False
    assert queued == [("seed-puuid",)]
    assert processed == []


def test_process_region_stops_and_unmarks_player_after_auth_error(monkeypatch):
    conn = make_conn()
    conn.execute("INSERT INTO puuid_queue_na1 VALUES (?)", ("seed-puuid",))

    match_api = FakeMatchApi(history_error=DummyApiError(401))
    monkeypatch.setattr(data_api_sqlite, "lol_watcher", FakeLolWatcher(match_api))
    monkeypatch.setattr(data_api_sqlite, "riot_watcher", FakeRiotWatcher(FakeAccountApi()))
    monkeypatch.setattr(data_api_sqlite, "ApiError", DummyApiError)
    monkeypatch.setattr(data_api_sqlite, "keep_running", True)

    did_work = data_api_sqlite.process_region(
        conn,
        {"platform": "na1", "region": "americas"},
    )

    queued = conn.execute("SELECT puuid FROM puuid_queue_na1").fetchall()
    processed = conn.execute("SELECT puuid FROM processed_puuids_na1").fetchall()
    conn.close()

    assert did_work is False
    assert data_api_sqlite.keep_running is False
    assert queued == [("seed-puuid",)]
    assert processed == []


def test_process_region_only_fetches_match_data_when_rank_helpers_exist(monkeypatch):
    conn = make_conn()
    conn.execute("INSERT INTO match_queue_na1 VALUES (?)", ("NA1_1",))
    data_api_sqlite.STORAGE_MODE = "sqlite"

    match_api = FakeMatchApi(match=sample_match())
    summoner_api = FakeSummonerApi()
    league_api = FakeLeagueApi()
    monkeypatch.setattr(data_api_sqlite, "lol_watcher", FakeLolWatcher(match_api, summoner_api, league_api))
    monkeypatch.setattr(data_api_sqlite, "riot_watcher", FakeRiotWatcher(FakeAccountApi()))

    did_work = data_api_sqlite.process_region(
        conn,
        {"platform": "na1", "region": "americas"},
    )

    stored = conn.execute(
        "SELECT rank_snapshot_json FROM matches WHERE match_id = ?",
        ("NA1_1",),
    ).fetchone()
    conn.close()

    assert did_work is True
    assert stored[0] is None
    assert summoner_api.calls == []
    assert league_api.calls == []


def test_signal_handler_first_request_is_graceful_second_forces_exit(monkeypatch, capsys):
    monkeypatch.setattr(data_api_sqlite, "keep_running", True)
    monkeypatch.setattr(data_api_sqlite, "stop_requests", 0)

    forced = {"called": False}

    def fake_default_int_handler(sig, frame):
        forced["called"] = True
        raise KeyboardInterrupt()

    monkeypatch.setattr(data_api_sqlite.signal, "default_int_handler", fake_default_int_handler)

    data_api_sqlite.signal_handler(None, None)
    assert data_api_sqlite.keep_running is False
    assert data_api_sqlite.stop_requests == 1
    assert forced["called"] is False

    with pytest.raises(KeyboardInterrupt):
        data_api_sqlite.signal_handler(None, None)

    captured = capsys.readouterr()
    assert "Stopping safely" in captured.out
    assert "Force exiting now" in captured.out
    assert forced["called"] is True
