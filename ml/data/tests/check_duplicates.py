import json
from collections import Counter

import pandas as pd

from ml.data.pytorch_data import get_data_frames

# --- Configuration ---
CHAMP_LIST_PATH = 'ml/save_data/champion_list.json'

def check_repeated_matchups():
    print("Loading Data...")
    df, champ_dict = get_data_frames()

    # Invert dict for printing names
    id_to_name = {v: k for k, v in champ_dict.items()}

    print(f"Scanning {len(df)} games for duplicates...")

    # We use a Counter to store the "Signature" of every game
    # Signature = A tuple of two tuples, sorted.
    # This makes it Side Agnostic: (TeamA, TeamB) == (TeamB, TeamA)
    match_counter = Counter()

    # Iterate through the raw numpy array for speed
    # Col 0 is Win/Loss, Cols 1-5 Blue, Cols 6-10 Red
    data = df.iloc[:, 1:11].values

    for row in data:
        # Extract Teams (Assuming standard role order: Top, Jgl, Mid, Bot, Sup)
        # We assume strict role order for a "Duplicate".
        # (Garen Top is different from Garen Mid)
        blue_team = tuple(row[0:5])
        red_team = tuple(row[5:10])

        # Normalize: Sort the teams so Side doesn't matter
        if blue_team < red_team:
            signature = (blue_team, red_team)
        else:
            signature = (red_team, blue_team)

        match_counter[signature] += 1

    # --- Analysis ---
    total_games = len(df)
    unique_games = len(match_counter)
    repeated_games = total_games - unique_games

    print("\n" + "="*40)
    print(f"Total Games Scanned:   {total_games:,}")
    print(f"Unique Comps Found:    {unique_games:,}")
    print(f"Duplicate Occurrences: {repeated_games:,} ({(repeated_games/total_games)*100:.2f}%)")
    print("="*40 + "\n")

    print("--- TOP 20 MOST COMMON MATCHUPS ---")

    # Sort by frequency
    most_common = match_counter.most_common(20)

    if not most_common:
        print("No repeated games found! The combinations are too vast.")

    for idx, (sig, count) in enumerate(most_common, 1):
        team_a_ids, team_b_ids = sig

        # Convert IDs to Names
        team_a = [id_to_name.get(i, str(i)) for i in team_a_ids]
        team_b = [id_to_name.get(i, str(i)) for i in team_b_ids]

        print(f"{idx}. Seen {count} times:")
        print(f"   Team A: {team_a}")
        print(f"   Team B: {team_b}")
        print("-" * 30)

if __name__ == "__main__":
    check_repeated_matchups()
