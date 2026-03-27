import json
import os
import socket
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_SCRAPER_CONFIG = {
    "batch_size": 50,
    "queue_id": 420,
    "start_time": 1772323200,
    "status_interval_sec": 60,
    "targets": [
        {"platform": "na1", "region": "americas", "seed_name": "DrDoughnut", "seed_tag": "GGG"},
        {"platform": "euw1", "region": "europe", "seed_name": "Agurin", "seed_tag": "DND"},
        {"platform": "kr", "region": "asia", "seed_name": "Hide on bush", "seed_tag": "KR1"},
    ],
    "rank_snapshots": {
        "enabled": False,
        "ttl_seconds": 21600,
    },
}


_SCRAPER_CONFIG_CACHE = None


def load_runtime_env():
    load_dotenv()
    load_dotenv(".env.local", override=True)


def _deep_merge(base, override):
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_scraper_config(config_path="config/scraper.json"):
    global _SCRAPER_CONFIG_CACHE

    if _SCRAPER_CONFIG_CACHE is not None:
        return deepcopy(_SCRAPER_CONFIG_CACHE)

    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / config_path
    config = deepcopy(DEFAULT_SCRAPER_CONFIG)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        config = _deep_merge(config, loaded)

    _SCRAPER_CONFIG_CACHE = config
    return deepcopy(_SCRAPER_CONFIG_CACHE)


def get_api_key():
    return os.getenv("RIOT_API_KEY")


def get_db_path(default="league_data.db"):
    return os.getenv("LOL_DRAFT_DB_PATH", default)


def get_collector_id():
    return os.getenv("LOL_DRAFT_COLLECTOR", socket.gethostname())


def get_batch_size():
    return int(get_scraper_config()["batch_size"])


def get_queue_id():
    return int(get_scraper_config()["queue_id"])


def get_start_time():
    return int(get_scraper_config()["start_time"])


def get_status_interval_sec():
    return int(get_scraper_config()["status_interval_sec"])


def get_scraper_targets():
    return list(get_scraper_config()["targets"])


def should_capture_rank_snapshots():
    return bool(get_scraper_config()["rank_snapshots"]["enabled"])


def get_rank_snapshot_ttl_seconds():
    return int(get_scraper_config()["rank_snapshots"]["ttl_seconds"])
