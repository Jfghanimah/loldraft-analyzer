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
    assert config["storage"]["mode"] == "compact"


def test_get_scraper_config_uses_defaults_when_file_missing(monkeypatch):
    monkeypatch.setattr(runtime_config, "_SCRAPER_CONFIG_CACHE", None)
    config = runtime_config.get_scraper_config("does-not-exist.json")

    assert config["queue_id"] == 420
    assert config["targets"][0]["platform"] == "na1"
    assert config["storage"]["compact_dataset_dir"] == "ml/save_data/lol_dataset_v1"


def test_storage_accessors_read_config_and_allow_env_override(tmp_path, monkeypatch):
    config_path = tmp_path / "scraper.json"
    config_path.write_text(
        """
        {
          "storage": {
            "mode": "sqlite",
            "compact_dataset_dir": "custom/dataset"
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(runtime_config, "_SCRAPER_CONFIG_CACHE", None)
    monkeypatch.setenv("LOL_DRAFT_STORAGE_MODE", "")
    monkeypatch.delenv("LOL_DRAFT_STORAGE_MODE", raising=False)
    monkeypatch.delenv("LOL_DRAFT_DATASET_DIR", raising=False)
    runtime_config.get_scraper_config(str(config_path))

    assert runtime_config.get_storage_mode() == "sqlite"
    assert runtime_config.get_compact_dataset_dir() == "custom/dataset"

    monkeypatch.setenv("LOL_DRAFT_STORAGE_MODE", "compact")
    monkeypatch.setenv("LOL_DRAFT_DATASET_DIR", "override/dataset")

    assert runtime_config.get_storage_mode() == "compact"
    assert runtime_config.get_compact_dataset_dir() == "override/dataset"
