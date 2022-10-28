import csv
import pandas as pd

api_key = "RGAPI-12ea6b09-b036-4a07-ae50-37e78e58ba39"

champion_features = {}

with open("champion_features.csv") as f:
    for row in csv.DictReader(f):
        champion_features[row['Champion']] = row


# Somehow get this data from API
game = ['Blue', 'Wukong', 'Ekko', 'Irelia', 'Xayah', 'Rakan', 'Sett', 'Kindred', 'Swain', 'Ezreal', 'Senna']


#We need to take the champions and bake their features into a 10*10 = 100 long entry per game

champion_data = []
for champ in game[1:]:
    champion_features[champ]['Early']
    champion_data.append()

