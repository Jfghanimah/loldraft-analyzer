import json
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt



def champion_winrates(df_matches):
    cols_a = [1, 2, 3, 4, 5]
    cols_b = [6, 7, 8, 9, 10]

    champ_winloss = {} # { champ_id:[wins, losses] }
    i = 0
    for _, win, b1, b2, b3, b4, b5, r1, r2, r3, r4, r5 in df_matches.itertuples():
        i+=1
        if i % 10000 == 0:
            pass
            #print(i)

        #win = row[0]
        team_a = [b1, b2, b3, b4, b5]
        team_b =  [r1, r2, r3, r4, r5]
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
    # TODO: this function works and is bug free but does not seem to improve bias 
    # if Heimerdinger(#41) has 5500 wins and 5400 losses then we need to remove 100 games where ahri is on the winning team
    # JUST DO FOR 1 CHAMPION FOR THE TIME BEING
    # Doing this on multiple champions does not seem to work

    print("Balancing winrates of champions.....")

    cols_blueteam = [1, 2, 3, 4, 5]
    cols_redteam = [6, 7, 8, 9, 10]

    deviations = []
    len_matches = []

    # Iterations of gimping/normalizing the most OP champion
    for i in range(50): # ~100 Gets the variance of winrates down significantly without reducing total matches by much
        # Step 1 - Compute the win rate for each champion and standard deviation
        champ_win_loss, sorted_champ_winrates = champion_winrates(df_matches)
        winrates = [x[1] for x in sorted_champ_winrates]
        std_dev = np.std(winrates)
        deviations.append(std_dev)
        len_matches.append(len(df_matches))
        #print(sorted_champ_winrates[0:10])    
        #print(std_dev)
        #print(len(df_matches), "\n")

        # Step 2 - calculate the number of games that need to be removed (wins - losses)
        #most_op_id = sorted_champ_winrates[0][0]
        most_op_id = sorted_champ_winrates[0][0] if abs(sorted_champ_winrates[0][1] - 0.5) > abs(sorted_champ_winrates[-1][1] - 0.5) else sorted_champ_winrates[-1][0]
        wins_to_remove = champ_win_loss[most_op_id][0] - champ_win_loss[most_op_id][1]

        # Step 3 - Remove a number of wins (or losses if negative) for each champ respectively
        if wins_to_remove > 0: 
            champion_win_blue_idx = df_matches[(df_matches[cols_blueteam].eq(most_op_id).sum(1) > 0) & (df_matches[0] == True)].index            
            champion_win_red_idx = df_matches[(df_matches[cols_redteam].eq(most_op_id).sum(1) > 0) & (df_matches[0] == False)].index
            # Drop the last `games_to_remove` matches where the champion is on the winning team
            df_matches = df_matches.drop(champion_win_blue_idx[0:int(wins_to_remove/4)])
            df_matches = df_matches.drop(champion_win_red_idx[0:int(wins_to_remove/4)])
        elif wins_to_remove < 0: #Its really losses to remove now
            losses_to_remove = -wins_to_remove
            champion_lose_blue_idx = df_matches[(df_matches[cols_blueteam].eq(most_op_id).sum(1) > 0) & (df_matches[0] == False)].index            
            champion_lose_red_idx = df_matches[(df_matches[cols_redteam].eq(most_op_id).sum(1) > 0) & (df_matches[0] == True)].index
            # Drop the last `games_to_remove` matches where the champion is on the losing team
            df_matches = df_matches.drop(champion_lose_blue_idx[0:int(losses_to_remove/4)])
            df_matches = df_matches.drop(champion_lose_red_idx[0:int(losses_to_remove/4)])


    # Step 1 again to check the end result
    champ_win_loss, sorted_champ_winrates = champion_winrates(df_matches)
    winrates = [x[1] for x in sorted_champ_winrates]
    std_dev = np.std(winrates)
    #print(sorted_champ_winrates[0:10])    
    #print(std_dev)
    #print(len(df_matches), "\n")

    #plots to analyze how many iterations of normalization to run
    '''
    x_vals = range(len(deviations))
    y1_values = deviations
    y2_values = len_matches

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    ax1.plot(x_vals, y1_values, label='Winrate spread')
    ax1.set_ylim(0, 0.017)
    ax1.set_xlabel('X values')
    ax1.set_ylabel('y1 values')
    ax1.legend()

    ax2.plot(x_vals, y2_values, label='Matches left')
    ax2.set_ylim(0, 150000)
    ax2.set_xlabel('X values')
    ax2.set_ylabel('y2 values')
    ax2.legend()
    plt.show()
    quit()
    '''

    return df_matches



def preprocess_data(matches):
    print("Preprocessing Data...")
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

    # Remove some matches to balace the overall winrates of outlier champs to remove bias and improve training?
    # df_matches = balance_winrates(df_matches)   #doesnt really help at all

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
    stopping_point = round(length*0.90) #make it automatically use 90% of it for training?
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