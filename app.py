from function import *
import cv2
import numpy as np
from keras.models import model_from_json
import mediapipe as mp

# Load the trained model
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

# Define actions
actions = np.array(['Help me', 'Thank you'])

# Real-time phrase recognition setup
sequence = []
sentence = "ASL"  # Start with "ASL" as the default sentence
threshold = 0.8
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        image, results = mediapipe_detection(frame, hands)

        # Extract and save keypoints to sequence
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            # Check for prediction confidence
            if res[np.argmax(res)] > threshold:
                # Update sentence only if a high-confidence action is detected
                predicted_action = actions[np.argmax(res)]
                # Avoid repeated actions
                if sentence != predicted_action:
                    sentence = predicted_action
            else:
                # Keep the default sentence "ASL" if no confident prediction
                sentence = "ASL"  # Reset to default if confidence is below threshold

        # Display the output
        cv2.putText(frame, sentence, (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Phrase Recognition', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
