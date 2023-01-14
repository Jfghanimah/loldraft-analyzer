import json
import os
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
sample_match = [True, 'Ornn', 'Vi', 'Orianna', 'Kaisa', 'Nautilus', 'Teemo', 'MasterYi', 'Akshan', 'Caitlyn', 'Senna'] 
sample_match = [champion_list[champ] if isinstance(champ, str) else champ for champ in sample_match]
sample_match = np.array(sample_match)  # Convert the sample match to a NumPy array

# Convert the champion names to integer IDs in each match
print("Converting champion names into IDs")
for match_id, match in matches.items():
    matches[match_id] = [champion_list[champ] if isinstance(champ, str) else champ for champ in match]

# Create a pandas DataFrame from the matches dictionary
df = pd.DataFrame.from_dict(matches, orient='index')

# Split the DataFrame into a training set and a test set
df_train = df.iloc[:90000, :]  # First 90k matches for training
df_test = df.iloc[90000:, :]  # Everything after 90k for testing

# Get the indices of the rows where win is True and drop last 2000
win_indices = df_train[df_train[0] == True].index
df_train = df_train.drop(win_indices[-2080:])
print(df_train[0].value_counts())

'''
#I want to recreate more draining examples
# Create a list to store the permuted DataFrames
df_train_permuted = []
# Iterate over each match in the original DataFrame
for _, match in df_train.copy().iterrows():
    # Permute the champions in the match 10 times
    for i in range(1):
        # Convert the series to a NumPy array
        match_array = match.values
        # Shuffle the champions in the match
        np.random.shuffle(match_array[1:6])
        np.random.shuffle(match_array[6:])
        # Convert the array back to a series and append it to the list
        df_train_permuted.append(pd.Series(match_array))

# Concatenate the permuted matches with the original DataFrame
df_train = df_train.append(df_train_permuted)
'''

# Extract the features and labels from the DataFrames
features = df_train.copy()
labels = features.pop(0)

test_features = df_test.copy()
test_labels = test_features.pop(0)

model = tf.keras.Sequential([
    #tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Embedding(162, 16, input_length=10),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.45),
    tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.45),
    tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.45),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])

# Create a TensorBoard callback
log_dir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)           

model.fit(features, labels, epochs=50, batch_size=32, validation_data=(test_features, test_labels), callbacks=[tensorboard_callback])

print("Testing model.....")
results = model.evaluate(test_features, test_labels, batch_size=5)

print("Predicting on sample data...")
prediction = model.predict(tf.reshape(sample_match[1:], (1, 10)))
print(prediction)  # Print the prediction



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