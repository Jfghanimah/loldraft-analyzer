import csv
import json
import pandas as pd
from scorecard import team_scorecard


# The objective of this file is to convert the match_info.json files into CSVs with all the one-hot/multi-hot features
# I will have to test with how to condense 162 champions (162 categories) down (Binary, Features, etc.)
# First I will try to sum the whole teams features values together and see how that goes

# win , champ1, champ2, etc...
# match = [True, 'Ornn', 'Vi', 'Orianna', 'Kaisa', 'Nautilus', 'Fiora', 'LeeSin', 'Zed', 'Caitlyn', 'Senna']
def convert_champions(match):
    # This is the features method using the csv champion_features
    champion_features = {}
    with open("champion_features.csv") as f:
        for row in csv.DictReader(f):
            champion_features[row['Champion']] = row

    match_flattened = []
    if match[0] is True:
        match_flattened.append(1)
    else:
        match_flattened.append(0)

    for champ in match[1:]:
        # There are 20 attributes we want * 10 champs = 200 dense layer per game
        match_flattened += (list(champion_features[champ].values())[2:])

    return [float(x) for x in match_flattened]


def convert_all_matches():
        
    with open("save_data/match_info.json") as f:
        match_info = json.loads(f.read())

    match_df = pd.DataFrame(columns=list(range(201))) # 1 + 200 columns when flat
    # Create flattened matches of only the features for the model to use
    # Wow this is really slow gotta work on it
    i = 0
    for _, match in match_info.items():
        i += 1
        if i % 100 == 0: # progress
            print(i)

        match_flattened = convert_champions(match)
        match_df.loc[len(match_df.index)] = match_flattened #pretty sure this is what is making it slow( .index helps)

    match_df.to_csv("save_data/match_df.csv",index=False)

# Somehow add the name of the champion as well. Maybe just the champ with no features??? This probably needs way more data 500k+

if __name__ == "__main__":
    convert_all_matches()

