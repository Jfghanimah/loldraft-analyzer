import json
import numpy as np
import tensorflow as tf

from model_preprocess import preprocess_data

# Read the champion list from the champion_list.json file
with open('save_data/champion_list.json') as f:
    champion_list = json.loads(f.read())

# Read the matches from the matches.json file
with open('save_data/match_info.json') as f:
    matches = json.loads(f.read())

configurations = [
                   {"name": "model_1.0.0", "lr":0.01, "momentum": 0.99}, 
                   {"name": "model_1.0.1", "lr":0.001, "momentum": 0.90},
                ]

for config in configurations:
    model_name = config['name']
    lr = config['lr']
    momentum = config['momentum']

    features, labels, test_features, test_labels = preprocess_data(matches)

    model = tf.keras.Sequential(name=model_name, layers=[
        tf.keras.layers.Embedding(162, 8, input_length=10),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(400, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(250, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum, nesterov=True), loss='binary_crossentropy', metrics=['accuracy'])

    # Create a TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{model_name}')

    model.fit(features, labels, epochs=40, batch_size=256, validation_data=(test_features, test_labels), callbacks=[tensorboard_callback])

    print("Testing model.....")
    results = model.evaluate(test_features, test_labels, batch_size=128)
    model.save(f'models/{model_name}.h5')

    sample_match = ['Ornn', 'Diana', 'Yasuo', 'Ashe', 'Amumu', 'Renekton', 'Zed', 'Yone', 'Caitlyn', 'Senna']
    sample_match2 = ['Renekton', 'Zed', 'Yone', 'Caitlyn', 'Senna', 'Ornn', 'Diana', 'Yasuo', 'Ashe', 'Amumu']
    sample_match = np.array([champion_list[champ] if isinstance(champ, str) else champ for champ in sample_match])
    sample_match2 = np.array([champion_list[champ] if isinstance(champ, str) else champ for champ in sample_match2])
    sample_match = tf.reshape(sample_match, (1, 10))
    sample_match2 = tf.reshape(sample_match2, (1, 10))
    print("Predicting on sample data...")
    print(model.predict(sample_match, verbose=0)[0][0])  # Print the prediction
    print(model.predict(sample_match2, verbose=0)[0][0])  # Print the prediction



