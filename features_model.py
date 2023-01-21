import json
import os
import itertools
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector


# Read the champion list from the champion_list.json file
with open('save_data/champion_list.json') as f:
    champion_list = json.loads(f.read())


# Read the matches from the matches.json file
with open('save_data/match_info.json') as f:
    matches = json.loads(f.read())

# Pretend Match for final predict and to show how the input is pre-processed
sample_match = [True, 'Darius', 'Graves', 'Lissandra', 'Kaisa', 'Blitzcrank', 'Heimerdinger', 'Amumu', 'Vladimir', 'Neeko', 'Morgana']
sample_match2 = [False, 'Heimerdinger', 'Amumu', 'Vladimir', 'Neeko', 'Morgana', 'Darius', 'Graves', 'Lissandra', 'Kaisa', 'Blitzcrank']
sample_match3 = [True, 'Yorick', 'Shyvana', 'Viktor', 'Tristana', 'Sion', 'Nasus', 'Urgot', 'Vex', 'Kaisa', 'Rakan']
sample_match4 = [False,  'Nasus', 'Amumu', 'Vex', 'Kaisa', 'Rakan', 'Yorick', 'Shyvana', 'Viktor', 'Tristana', 'Sion']
sample_match5 = [True, "Quinn", "Nunu", "TwistedFate", "Kaisa", "JarvanIV", "Gangplank", "Ahri", "Fizz", "Ezreal", "Karma"]
sample_match6 = [False, "Gangplank", "Ahri", "Fizz", "Ezreal", "Karma", "Quinn", "Nunu", "TwistedFate", "Kaisa", "JarvanIV"]

sample_match = np.array([champion_list[champ] if isinstance(champ, str) else champ for champ in sample_match])
sample_match2 = np.array([champion_list[champ] if isinstance(champ, str) else champ for champ in sample_match2])
sample_match3 = np.array([champion_list[champ] if isinstance(champ, str) else champ for champ in sample_match3])
sample_match4 = np.array([champion_list[champ] if isinstance(champ, str) else champ for champ in sample_match4])
sample_match3 = np.array([champion_list[champ] if isinstance(champ, str) else champ for champ in sample_match5])
sample_match4 = np.array([champion_list[champ] if isinstance(champ, str) else champ for champ in sample_match6])

# Convert the champion names to integer IDs in each match
print("Converting champion names into IDs")
for match_id, match in matches.items():
    matches[match_id] = [champion_list[champ] if isinstance(champ, str) else champ for champ in match]

# Create a pandas DataFrame from the matches dictionary
df_matches = pd.DataFrame.from_dict(matches, orient='index')
#print(df_matches.head())
# save columns of players within the dataset
cols_a = [1, 2, 3, 4, 5]
cols_b = [6, 7, 8, 9, 10]

# ----- PERMUTATION OF TWO TEAMS doubles dataset
df_swapped = df_matches.copy()
df_swapped[cols_a], df_swapped[cols_b] = df_matches[cols_b], df_matches[cols_a]
# need to swap the win and loss HOW CAN IT LEARN WITHOUT THIS??? EXCELENT TEST
df_swapped[0] = ~df_swapped[0]
df_matches = pd.concat([df_matches, df_swapped], ignore_index=True)

# randomly shuffle the rows of a DataFrame
indices = np.arange(len(df_matches))
np.random.shuffle(indices)
df_matches = df_matches.iloc[indices]

# Split the DataFrame into a training set and a test set (make it automatically use 95% of it for training?)
df_train = df_matches.iloc[:230000, :]  # First 1M matches for training (120k * 2 = 240)
df_test = df_matches.iloc[230000:, :]  # Everything after 1M for testing

# Get the indices of the rows where win is True and drop last 2000
win_indices = df_train[df_train[0] == True].index
extra_wins = df_train[0].value_counts()[0]-df_train[0].value_counts()[1]
df_train = df_train.drop(win_indices[-extra_wins:])
print(df_train[0].value_counts())

# Extract the features and labels from the DataFrames
features = df_train.copy()
labels = features.pop(0)

test_features = df_test.copy()
test_labels = test_features.pop(0)

model = tf.keras.Sequential([
    #tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Embedding(162, 10, input_length=10),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(400, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.35),
    tf.keras.layers.Dense(400, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.35),
    tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.35),
    tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.35),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])

# Create a TensorBoard callback
log_dir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)           

model.fit(features, labels, epochs=20, batch_size=200, validation_data=(test_features, test_labels), callbacks=[tensorboard_callback])

print("Testing model.....")
results = model.evaluate(test_features, test_labels, batch_size=5)

print("Predicting on sample data...")
print(model.predict(tf.reshape(sample_match[1:], (1, 10))))  # Print the prediction
print(model.predict(tf.reshape(sample_match2[1:], (1, 10))))  # Print the prediction
print(model.predict(tf.reshape(sample_match3[1:], (1, 10))))  # Print the prediction
print(model.predict(tf.reshape(sample_match4[1:], (1, 10))))  # Print the prediction
print(model.predict(tf.reshape(sample_match5[1:], (1, 10))))  # Print the prediction
print(model.predict(tf.reshape(sample_match6[1:], (1, 10))))  # Print the prediction

# Embedding Code:
# Save Labels separately on a line-by-line manner.
with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
  for champ in champion_list:
    f.write(f"{champ}\n")

# Save the weights we want to analyze as a variable. Note that the first
# value represents any unknown word, which is not in the metadata, here
# we will remove this value.
weights = tf.Variable(model.layers[0].get_weights()[0])
# Create a checkpoint from embedding, the filename and key are the
# name of the tensor.
checkpoint = tf.train.Checkpoint(embedding=weights)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

# Set up config.
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
# The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = os.path.join(log_dir, 'metadata.tsv')
projector.visualize_embeddings(log_dir, config)