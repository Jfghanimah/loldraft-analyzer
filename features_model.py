import pandas as pd
import tensorflow as tf
from bake_features import convert_champions

# create one-hot for chess elo leading to wins, add random noise to test how hard it is to dig through noise
# elo-higher column
# test win every time and simulate elo higher = 75% chance to win.

sample_match = [True, 'Ornn', 'Vi', 'Orianna', 'Kaisa', 'Nautilus', 'Fiora', 'LeeSin', 'Zed', 'Caitlyn', 'Senna']
sample_data = [convert_champions(sample_match)[1:]]

#Getting csv in df
df = pd.read_csv('save_data/match_df2.csv')

df_train = df.iloc[:70000, :] # now ive got 79k
df_test = df.iloc[70000:, :]

features = df_train.copy()
labels = features.pop('0') # This is the win column

test_features = df_test.copy()
test_labels = test_features.pop('0') # This is the win column


model = tf.keras.Sequential([
tf.keras.layers.Input(shape=(200,)),
tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])

model.fit(features, labels, epochs=50, batch_size=10)

print("Testing model.....")
results = model.evaluate(test_features, test_labels, batch_size=5)

x = model(tf.convert_to_tensor(sample_data))
print(x)