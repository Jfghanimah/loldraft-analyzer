import argparse
import sqlite3

from ml.data.match_storage import ensure_match_schema


def _connect(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    ensure_match_schema(conn)
    return conn


def _row_score(row):
    return (
        int(bool(row["ordered_match_json"])),
        int(bool(row["raw_match_json"])),
        int(bool(row["rank_snapshot_json"])),
        int(row["last_updated_ts"] or 0),
    )


def _choose_row(existing, incoming):
    if existing is None:
        return incoming
    return incoming if _row_score(incoming) >= _row_score(existing) else existing


def merge_matches(target_conn, source_conn):
    merged = 0
    inserted = 0
    updated = 0

    for row in source_conn.execute("SELECT * FROM matches"):
        existing = target_conn.execute(
            "SELECT * FROM matches WHERE match_id = ?",
            (row["match_id"],),
        ).fetchone()
        chosen = _choose_row(existing, row)
        if chosen is existing:
            continue

        target_conn.execute(
            """
            INSERT INTO matches (
                match_id,
                match_data,
                region,
                raw_match_json,
                ordered_match_json,
                rank_snapshot_json,
                data_source,
                collector_id,
                game_version,
                queue_id,
                game_creation,
                game_end_timestamp,
                blue_first_blood,
                blue_first_tower,
                blue_dragon_share,
                blue_gold_share,
                last_updated_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(match_id) DO UPDATE SET
                match_data = excluded.match_data,
                region = excluded.region,
                raw_match_json = excluded.raw_match_json,
                ordered_match_json = excluded.ordered_match_json,
                rank_snapshot_json = excluded.rank_snapshot_json,
                data_source = excluded.data_source,
                collector_id = excluded.collector_id,
                game_version = excluded.game_version,
                queue_id = excluded.queue_id,
                game_creation = excluded.game_creation,
                game_end_timestamp = excluded.game_end_timestamp,
                blue_first_blood = excluded.blue_first_blood,
                blue_first_tower = excluded.blue_first_tower,
                blue_dragon_share = excluded.blue_dragon_share,
                blue_gold_share = excluded.blue_gold_share,
                last_updated_ts = excluded.last_updated_ts
            """,
            (
                row["match_id"],
                row["match_data"],
                row["region"],
                row["raw_match_json"],
                row["ordered_match_json"],
                row["rank_snapshot_json"],
                row["data_source"],
                row["collector_id"],
                row["game_version"],
                row["queue_id"],
                row["game_creation"],
                row["game_end_timestamp"],
                row["blue_first_blood"],
                row["blue_first_tower"],
                row["blue_dragon_share"],
                row["blue_gold_share"],
                row["last_updated_ts"],
            ),
        )
        merged += 1
        inserted += int(existing is None)
        updated += int(existing is not None)

    return merged, inserted, updated


def main():
    parser = argparse.ArgumentParser(description="Merge one or more match SQLite databases.")
    parser.add_argument("sources", nargs="+", help="Source SQLite database paths")
    parser.add_argument("--target", required=True, help="Target SQLite database path")
    args = parser.parse_args()

    target_conn = _connect(args.target)
    total_merged = 0
    total_inserted = 0
    total_updated = 0

    try:
        for source_path in args.sources:
            if source_path == args.target:
                continue

            source_conn = _connect(source_path)
            try:
                merged, inserted, updated = merge_matches(target_conn, source_conn)
                total_merged += merged
                total_inserted += inserted
                total_updated += updated
                print(
                    f"{source_path}: merged={merged} inserted={inserted} updated={updated}"
                )
            finally:
                source_conn.close()

        target_conn.commit()
    finally:
        target_conn.close()

    print(
        f"Done. merged={total_merged} inserted={total_inserted} updated={total_updated}"
    )


if __name__ == "__main__":
    main()
