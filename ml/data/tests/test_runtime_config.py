import ml.runtime_config as runtime_config


def test_get_scraper_config_reads_json_and_merges_defaults(tmp_path, monkeypatch):
    config_path = tmp_path / "scraper.json"
    config_path.write_text(
        """
        {
          "queue_id": 440,
          "rank_snapshots": {
            "enabled": true
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(runtime_config, "_SCRAPER_CONFIG_CACHE", None)
    config = runtime_config.get_scraper_config(str(config_path))

    assert config["queue_id"] == 440
    assert config["rank_snapshots"]["enabled"] is True
    assert config["rank_snapshots"]["ttl_seconds"] == 21600
    assert config["batch_size"] == 50


def test_get_scraper_config_uses_defaults_when_file_missing(monkeypatch):
    monkeypatch.setattr(runtime_config, "_SCRAPER_CONFIG_CACHE", None)
    config = runtime_config.get_scraper_config("does-not-exist.json")

    assert config["queue_id"] == 420
    assert config["targets"][0]["platform"] == "na1"
