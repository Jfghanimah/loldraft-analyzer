import json
import os
import numpy as np
import tensorflow as tf

from tensorboard.plugins import projector
from model_preprocess import preprocess_data

# Read the champion list from the champion_list.json file
with open('save_data/champion_list.json') as f:
    champion_list = json.loads(f.read())

# Read the matches from the matches.json file
with open('save_data/match_info.json') as f:
    matches = json.loads(f.read())

# Pretend Match for final predict and to show how the input is pre-processed
sample_match = [True, 'Teemo', 'Ekko', 'Syndra', 'Ezreal', 'Sona', 'Volibear', 'Warwick', 'Veigar', 'Caitlyn', 'Lux']
sample_match = np.array([champion_list[champ] if isinstance(champ, str) else champ for champ in sample_match])

features, labels, test_features, test_labels = preprocess_data(matches)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(162, 8, input_length=10),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Create a TensorBoard callback
log_dir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)           

model.fit(features, labels, epochs=30, batch_size=1024, validation_data=(test_features, test_labels), callbacks=[tensorboard_callback])

print("Testing model.....")
results = model.evaluate(test_features, test_labels, batch_size=100)
model.save('models/draft-winner.h5')

print("Predicting on sample data...")
print(model.predict(tf.reshape(sample_match[1:], (1, 10))))  # Print the prediction

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