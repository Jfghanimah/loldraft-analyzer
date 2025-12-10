import json
import numpy as np
import tensorflow as tf

with open('save_data/champion_list.json') as f:
    champion_list = json.loads(f.read())


model = tf.keras.models.load_model('models/model7a.h5')
model2 = tf.keras.models.load_model('models/model_1.0.0.h5')
model3 = tf.keras.models.load_model('models/model_1.0.1.h5')

match = ['Poppy', 'Nocturne', 'Orianna', 'Ashe', 'Taric', 'Ornn', 'Graves', 'Ahri', 'Draven', 'Nautilus']
match = ['Illaoi', 'Maokai', 'Orianna', 'Ashe', 'Renata', 'Ornn', 'Nocturne', 'Akali', 'Sivir', 'Senna']
match_rev = match[5:] + match[0:5]
match = np.array([champion_list[champ] if isinstance(champ, str) else champ for champ in match])
match_rev = np.array([champion_list[champ] if isinstance(champ, str) else champ for champ in match_rev])
match = tf.reshape(match, (1, 10))
match_rev = tf.reshape(match_rev, (1, 10))

print("Predicting on sample data...")
print("Model 1:")
print(model.predict(match, verbose=0)[0][0])  # Print the prediction
print(1-model.predict(match_rev, verbose=0)[0][0])  # Print the prediction

print("Model 2:")
print(model2.predict(match, verbose=0)[0][0])  # Print the prediction
print(1-model2.predict(match_rev, verbose=0)[0][0])  # Print the prediction

print("Model 3:")
print(model3.predict(match, verbose=0)[0][0])  # Print the prediction
print(1-model3.predict(match_rev, verbose=0)[0][0])  # Print the prediction

def best_champs(match, role):
    champ_percents = {} # name : winrate
    for champ_name, _ in champion_list.items():
        match[role] = champ_name
        match_arr = np.array([champion_list[c] if isinstance(c, str) else c for c in match])
        match_tensor = tf.reshape(match_arr, (1, 10))
        prediction = model2.predict(match_tensor, verbose=0)[0][0]  # Print the prediction
        champ_percents[champ_name] = prediction

    sorted_champ_percents = dict(sorted(champ_percents.items(), key=lambda item: item[1], reverse=True))
    return sorted_champ_percents


