from function import *
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import os

# Define paths and parameters
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['Help me', 'Thank you'])
no_sequences = 30
sequence_length = 30

# Label Mapping for each action
label_map = {label: num for num, label in enumerate(actions)}

# Load Data and Labels
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# Convert data to arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Define TensorBoard for logging
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback], validation_data=(X_test, y_test))

# Save the model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save("model.h5")
print("Model trained and saved!")
