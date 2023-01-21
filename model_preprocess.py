import json
import pandas as pd
import numpy as np



def champion_winrates(df_matches):
    cols_a = [1, 2, 3, 4, 5]
    cols_b = [6, 7, 8, 9, 10]

    champ_winloss = {} # { champ_id:[wins, losses] }
    i = 0
    for idx, row in df_matches.iterrows():

        win = row[0]
        team_a = row[cols_a]
        team_b = row[cols_b]
        for champ_id in team_a:
            if champ_id not in champ_winloss:
                champ_winloss[champ_id] = [0, 0]
            if win:
                champ_winloss[champ_id][0] += 1
            else:
                champ_winloss[champ_id][1] += 1
        for champ_id in team_b:
            if champ_id not in champ_winloss:
                champ_winloss[champ_id] = [0, 0]
            if not win:
                champ_winloss[champ_id][0] += 1
            else:
                champ_winloss[champ_id][1] += 1

    champ_winrates = {}
    for champ, data in champ_winloss.items():
        wins, losses = data
        winrate = wins / (wins + losses)
        champ_winrates[champ] = round(winrate,4)
    sorted_champ_winrates = sorted(champ_winrates.items(), key=lambda x: x[1], reverse=True)
    return champ_winloss, sorted_champ_winrates


def balance_winrates(df_matches):
    # TODO: tally champ winrates and sort by highest to lowest
    # currently runs, however it has a whack-a-mole issue of addressing high winrate champs
    # save columns of players within the dataset
    cols_a = [1, 2, 3, 4, 5]
    cols_b = [6, 7, 8, 9, 10]

    # Step 1 - Compute the win rate for each champion
    champ_win_loss, sorted_champ_winrates = champion_winrates(df_matches)
    #print("\n", sorted_champ_winrates)

    # Step 2 - calculate the number of games that need to be removed to get to 50%
    wins_to_remove = {} # { champ_id : # of games to remove}
    # for each champion in the winrates dictionary
    for champ_id, wins_losses in champ_win_loss.items():
        wins_to_remove[champ_id] = wins_losses[0] - wins_losses[1]

    print(f"\n Wins to remove: {wins_to_remove}")

    # Step 3 - Remove a number of wins (or losses if negative) for each champ respectively
    total_remove = 0
    for champ_id, games_to_remove in wins_to_remove.items():
        if games_to_remove > 0:
            total_remove += games_to_remove
            # Dropping the last x matches where champion is on the winning team
            # Get the indices of the rows where the champion is on the winning team
            champion_win_indices = df_matches[(df_matches[cols_a].eq(champ_id).sum(1) > 0) & (df_matches[0] == True)].index
            # Drop the last `games_to_remove` matches where the champion is on the winning team
            df_matches = df_matches.drop(champion_win_indices[-games_to_remove:])

    print("\n Total Remove: ",total_remove)
    print("\n", df_matches)
    # Step 4 pray that somehow the winrates are much closer to 50%
    champ_win_loss, sorted_champ_winrates = champion_winrates(df_matches)
    for champ_id, wins_losses in champ_win_loss.items():
        wins_to_remove[champ_id] = wins_losses[0] - wins_losses[1]

    print(f"\n Wins to remove: {wins_to_remove}")

    return df_matches



def preprocess_data(matches):
    #sample_match = [True, 'Mordekaiser', 'Trundle', 'Viktor', 'Ezreal', 'Sona', 'Volibear', 'Warwick', 'Veigar', 'Caitlyn', 'Lux']
    
    # Read the champion list from the champion_list.json file
    with open('save_data/champion_list.json') as f:
        champion_list = json.loads(f.read())

    # Convert the champion names to integer IDs in each match for embedding layer
    print("Converting champion names into IDs")
    matches_converted = {}
    for match_id, match in matches.items():
        matches_converted[match_id] = [champion_list[champ] if isinstance(champ, str) else champ for champ in match]

    # Create a pandas DataFrame from the matches dictionary
    df_matches = pd.DataFrame.from_dict(matches_converted, orient='index')
    #print(df_matches.head())
    # save columns of players within the dataset
    cols_a = [1, 2, 3, 4, 5]
    cols_b = [6, 7, 8, 9, 10]

    # Remove matches to balace the overall winrates of each champ to remove bias
    # df_matches = balance_winrates(df_matches)   

    # ----- PERMUTATION OF TWO TEAMS doubles dataset (also balances total wins and losses so model doesnt guess blindly)
    df_swapped = df_matches.copy()
    df_swapped[cols_a], df_swapped[cols_b] = df_matches[cols_b], df_matches[cols_a]
    # need to swap the wins and losses
    df_swapped[0] = ~df_swapped[0]
    df_matches = pd.concat([df_matches, df_swapped], ignore_index=True)

    # randomly shuffle the rows of a DataFrame
    length = len(df_matches)
    indices = np.arange(length)
    np.random.shuffle(indices)
    df_matches = df_matches.iloc[indices]

    # Split the DataFrame into a training set and a test set 
    stopping_point = round(length*0.95) #make it automatically use 95% of it for training?
    df_train = df_matches.iloc[:stopping_point, :]
    df_test = df_matches.iloc[stopping_point:, :]

    # Get the indices of the rows where win is True and drop last to make even
    win_indices = df_train[df_train[0] == True].index
    extra_wins = df_train[0].value_counts()[0]-df_train[0].value_counts()[1]
    df_train = df_train.drop(win_indices[-extra_wins:])
    print(df_train[0].value_counts())

    # Extract the features and labels from the DataFrames
    features = df_train.copy()
    labels = features.pop(0)
    test_features = df_test.copy()
    test_labels = test_features.pop(0)

    return features, labels, test_features, test_labels