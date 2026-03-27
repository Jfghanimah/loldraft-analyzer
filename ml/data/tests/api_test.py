import os

import pytest
from ml.data.match_format import try_build_ordered_match_record
from ml.runtime_config import get_api_key, load_runtime_env

riotwatcher = pytest.importorskip("riotwatcher")
LolWatcher = riotwatcher.LolWatcher
RiotWatcher = riotwatcher.RiotWatcher

load_runtime_env()

PLATFORM = 'na1'
REGION = 'americas'
GAME_NAME = 'DrDoughnut'
TAG_LINE = 'GGG'


@pytest.mark.integration
@pytest.mark.smoke
def test_riot_api_lookup_smoke():
    if os.getenv("RUN_RIOT_API_TEST") != "1":
        pytest.skip("Set RUN_RIOT_API_TEST=1 to run the live Riot API smoke test.")

    riot_api_key = get_api_key()
    if not riot_api_key:
        pytest.skip("RIOT_API_KEY not found in environment.")

    riot_watcher = RiotWatcher(riot_api_key)
    lol_watcher = LolWatcher(riot_api_key)

    account = riot_watcher.account.by_riot_id(REGION, GAME_NAME, TAG_LINE)
    assert 'puuid' in account and account['puuid']

    summoner = lol_watcher.summoner.by_puuid(PLATFORM, account['puuid'])
    assert summoner['summonerLevel'] >= 1
    assert 'profileIconId' in summoner


@pytest.mark.integration
@pytest.mark.smoke
def test_riot_api_match_pipeline_smoke():
    if os.getenv("RUN_RIOT_API_TEST") != "1":
        pytest.skip("Set RUN_RIOT_API_TEST=1 to run the live Riot API smoke test.")

    riot_api_key = get_api_key()
    if not riot_api_key:
        pytest.skip("RIOT_API_KEY not found in environment.")

    riot_watcher = RiotWatcher(riot_api_key)
    lol_watcher = LolWatcher(riot_api_key)

    account = riot_watcher.account.by_riot_id(REGION, GAME_NAME, TAG_LINE)
    match_ids = lol_watcher.match.matchlist_by_puuid(
        REGION,
        account["puuid"],
        queue=420,
        count=5,
    )
    assert match_ids, "No ranked solo queue matches returned for smoke test."

    match = lol_watcher.match.by_id(REGION, match_ids[0])
    ordered_record, reason = try_build_ordered_match_record(match["info"])

    assert ordered_record is not None, reason
    assert ordered_record["format"] == "role_order_v1"
    assert len(ordered_record["champions"]) == 10
