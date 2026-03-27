import sqlite3
import time

from ml.data.match_storage import connect_sqlite, ensure_match_schema, rebuild_participant_history
from ml.runtime_config import get_db_path, load_runtime_env

PROGRESS_BAR_WIDTH = 24


def _print_backfill_progress(processed, total, matches_processed, participants_inserted, started_at, *, force=False):
    if total <= 0:
        return
    percent = (processed / total) * 100.0
    filled = min(PROGRESS_BAR_WIDTH, int((processed / total) * PROGRESS_BAR_WIDTH))
    bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
    elapsed = time.time() - started_at
    rate = processed / elapsed if elapsed > 0 else 0.0
    remaining = total - processed
    eta = remaining / rate if rate > 0 else 0.0
    line = (
        f"\rBackfilling participant_history [{bar}] {processed:,}/{total:,} "
        f"({percent:5.1f}%) | matches={matches_processed:,} | participants={participants_inserted:,} | "
        f"elapsed={elapsed:.1f}s | eta={eta:.1f}s"
    )
    print(line.ljust(160), end="\n" if force else "", flush=True)


def main():
    load_runtime_env()
    db_path = get_db_path()
    conn = connect_sqlite(db_path)
    try:
        ensure_match_schema(conn)
        matches_processed, participants_inserted = rebuild_participant_history(
            conn,
            progress_callback=_print_backfill_progress,
        )
        conn.commit()
        print(
            f"Rebuilt participant history from {matches_processed:,} matches "
            f"into {participants_inserted:,} participant rows."
        )
    except sqlite3.OperationalError as exc:
        if "locked" in str(exc).lower():
            print(
                "Could not backfill participant_history because the database is locked. "
                "The live scraper is probably still writing to it. Stop the scraper and rerun "
                "'py -m ml.data.backfill_participant_history' if you want a full backfill. "
                "Training can still proceed; it now falls back to raw_match_json for matches "
                "that have not been backfilled yet."
            )
            raise SystemExit(1) from exc
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
