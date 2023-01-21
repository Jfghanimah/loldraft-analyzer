import os
import json
import numpy as np
import tensorflow as tf

with open('save_data/champion_list.json') as f:
    champion_list = json.loads(f.read())

model = tf.keras.models.load_model('models/draft-winner.h5')

sample_match = [True, 'Poppy', 'Maokai', 'Viktor', 'Tristana', 'Rell', 'Fiora', 'DrMundo', 'Gragas', 'Varus', 'Zac']
sample_match2 = [False, 'Fiora', 'DrMundo', 'Gragas', 'Varus', 'Zac', 'Poppy', 'Maokai', 'Viktor', 'Tristana', 'Rell']
sample_match = np.array([champion_list[champ] if isinstance(champ, str) else champ for champ in sample_match])
sample_match2 = np.array([champion_list[champ] if isinstance(champ, str) else champ for champ in sample_match2])
sample_match = tf.reshape(sample_match[1:], (1, 10))
sample_match2 = tf.reshape(sample_match2[1:], (1, 10))
print("Predicting on sample data...")
print(model.predict(sample_match, verbose=0)[0][0])  # Print the prediction
print(model.predict(sample_match2, verbose=0)[0][0])  # Print the prediction
quit()

def best_champs(match, role):
    champ_percents = {} # name : winrate
    for champ_name, _ in champion_list.items():
        match[role] = champ_name
        match_arr = np.array([champion_list[c] if isinstance(c, str) else c for c in match])
        match_tensor = tf.reshape(match_arr[1:], (1, 10))
        prediction = model.predict(match_tensor, verbose=0)[0][0]  # Print the prediction
        champ_percents[champ_name] = prediction

    sorted_champ_percents = dict(sorted(champ_percents.items(), key=lambda item: item[1], reverse=True))
    return sorted_champ_percents


role_to_check = 4
match_to_check = [True, 'Garen', 'Trundle', 'Viktor', 'Jinx', 'Maokai', 'Mordekaiser', 'Rengar', 'Diana', 'Ashe', 'Teemo']
print(best_champs(match_to_check, role_to_check))