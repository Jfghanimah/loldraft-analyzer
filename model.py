import pandas as pd
import tensorflow as tf

# create one-hot for chess elo leading to wins, add random noise to test how hard it is to dig through noise
# elo-higher column
# test win every time and simulate elo higher = 75% chance to win.

#Getting csv in df
df = pd.read_csv('test_data.csv')  
df = df.drop(['Rand'], axis=1)
#print(df.head())

df_train = df.iloc[:700, :]
df_test = df.iloc[700:, :]

features = df_train.copy()
labels = features.pop('Win')

test_features = df_test.copy()
test_labels = test_features.pop('Win')


model = tf.keras.Sequential([
tf.keras.layers.Dense(24, activation='relu'),
tf.keras.layers.Dropout(rate=0.2),
tf.keras.layers.Dense(10, activation='relu'),
tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])


model.fit(features, labels, epochs=50, batch_size=5)


print("Testing model.....")
results = model.evaluate(test_features, test_labels, batch_size=50)