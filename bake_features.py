import csv
import json
import pandas as pd
from scorecard import team_scorecard


# The objective of this file is to convert the match_info.json files into CSVs with all the one-hot features
# I will have to expieriement with how to condense 162 champions (162 categories) down (Binary, Features, etc.)
# First I will try to sum the whole teams features values together and see how that goes

champion_features = {}
with open("champion_features.csv") as f:
    for row in csv.DictReader(f):
        champion_features[row['Champion']] = row


# Somehow get this data from API, dataframe would be nice.
# win , champ1, champ2, etc...
#[True, 'Wukong', 'Ekko', 'Irelia', 'Xayah', 'Rakan', 'Sett', 'Kindred', 'Swain', 'Ezreal', 'Senna']

with open("save_data/match_info.json") as f:
    match_info = json.loads(f.read())

match_df = pd.DataFrame()

for match_id, match in match_info.items():
   
    champion_data = []
    for champ in match[1:]:
        champion_features[champ]['Early']
        champion_data.append()



    


#We need to take the champions and bake their features into a 10*10 = 100 long entry per game



