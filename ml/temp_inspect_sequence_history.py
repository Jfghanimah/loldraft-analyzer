import argparse
import sqlite3
from collections import deque

from ml.data.match_storage import connect_sqlite
from ml.runtime_config import get_db_path, load_runtime_env


HISTORY_LENGTH = 10
ROLE_ORDER = {"TOP": 0, "JUNGLE": 1, "MIDDLE": 2, "BOTTOM": 3, "UTILITY": 4}


def _load_matches(conn, queue_id):
    return conn.execute(
        """
        SELECT match_id, region, game_creation
        FROM matches
        WHERE ordered_match_json IS NOT NULL
          AND (? IS NULL OR queue_id = ?)
        ORDER BY COALESCE(game_creation, 0), match_id
        """,
        (queue_id, queue_id),
    ).fetchall()


def _load_participants_by_match(conn, queue_id):
    rows = conn.execute(
        """
        SELECT match_id, puuid, champion_name, role, team_id, win, game_creation, game_version
        FROM participant_history
        WHERE (? IS NULL OR queue_id = ?)
        ORDER BY COALESCE(game_creation, 0), match_id, team_id,
                 CASE role
                     WHEN 'TOP' THEN 0
                     WHEN 'JUNGLE' THEN 1
                     WHEN 'MIDDLE' THEN 2
                     WHEN 'BOTTOM' THEN 3
                     WHEN 'UTILITY' THEN 4
                     ELSE 99
                 END
        """,
        (queue_id, queue_id),
    ).fetchall()

    by_match = {}
    for row in rows:
        by_match.setdefault(row[0], []).append(
            {
                "puuid": row[1],
                "champion_name": row[2],
                "role": row[3],
                "team_id": row[4],
                "win": row[5],
                "game_creation": row[6],
                "game_version": row[7],
            }
        )
    return by_match


def main():
    parser = argparse.ArgumentParser(description="Temporary inspector for sequence-history population.")
    parser.add_argument("--db-path", default=None, help="SQLite DB path (defaults to LOL_DRAFT_DB_PATH or league_data.db)")
    parser.add_argument("--queue-id", type=int, default=420, help="Queue ID filter (default: 420)")
    parser.add_argument("--history-length", type=int, default=HISTORY_LENGTH, help="Recent history length to simulate")
    parser.add_argument("--samples", type=int, default=3, help="How many populated matches to print")
    args = parser.parse_args()

    load_runtime_env()
    db_path = args.db_path or get_db_path()
    conn = connect_sqlite(db_path, read_only=True)

    matches = _load_matches(conn, args.queue_id)
    participants_by_match = _load_participants_by_match(conn, args.queue_id)
    history_store = {}

    total_examples = 0
    examples_with_history = 0
    total_history_tokens = 0
    sample_count = 0

    print(f"db={db_path}")
    print(f"ordered_matches={len(matches):,}")
    print(f"participant_history_matches={len(participants_by_match):,}")
    print(f"history_length={args.history_length}")
    print("")

    for match_id, region, game_creation in matches:
        participant_rows = participants_by_match.get(match_id, [])
        if len(participant_rows) != 10:
            continue

        total_examples += 1
        slot_counts = []
        for participant in participant_rows:
            prior_rows = list(history_store.get(participant["puuid"], ()))[: args.history_length]
            slot_counts.append(len(prior_rows))

        filled = sum(slot_counts)
        total_history_tokens += filled
        if filled > 0:
            examples_with_history += 1
            if sample_count < args.samples:
                print(f"match_id={match_id} region={region} game_creation={game_creation} total_history_tokens={filled}")
                print("slot_counts=" + ",".join(str(count) for count in slot_counts))
                for slot_index, participant in enumerate(participant_rows):
                    prior_rows = list(history_store.get(participant["puuid"], ()))[: args.history_length]
                    if not prior_rows:
                        continue
                    recent = prior_rows[0]
                    print(
                        f"  slot={slot_index} role={participant['role']} champ={participant['champion_name']} "
                        f"recent_prior=({recent['champion_name']}, {recent['role']}, win={recent['win']})"
                    )
                print("")
                sample_count += 1

        for participant in participant_rows:
            history = history_store.setdefault(participant["puuid"], deque(maxlen=args.history_length))
            history.appendleft(participant)

    avg_tokens = (total_history_tokens / total_examples) if total_examples else 0.0
    print(f"usable_examples={total_examples:,}")
    print(f"examples_with_history={examples_with_history:,}")
    print(f"avg_history_tokens_per_example={avg_tokens:.2f}")

    conn.close()


if __name__ == "__main__":
    main()
