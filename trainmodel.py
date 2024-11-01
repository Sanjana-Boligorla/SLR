import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Define actions and load data
actions = np.array([
    'Help me', 'Danger', 'Emergency', 'Stop', 'Go', 'Call Police', 'Call Ambulance', 
    'Fire', 'Accident', 'Need Doctor', 'Hurt/Injured', 'Stay Here', 'Follow Me', 
    'Safe/Safety', 'Threat', 'Run', 'Yes', 'No', 'Lost', 'Wait', 'Afraid', 
    'Pain', 'Protect', 'Rescue', 'Escape'
])
DATA_PATH = os.path.join('MP_Data')
no_sequences = 30
sequence_length = 30

# Load keypoint data and labels
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            res = np.load(npy_path)
            window.append(res)
        sequences.append(window)
        labels.append(np.where(actions == action)[0][0])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Build and compile LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, X.shape[2])),
    LSTM(128, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(actions.shape[0], activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=200, verbose=1)

# Save the model
model.save('action_model.h5')
print("Model trained and saved successfully.")
