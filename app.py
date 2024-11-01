import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp  # Import mediapipe with alias 'mp'
from function import mediapipe_detection, draw_styled_landmarks, extract_keypoints

# Load the model and define actions
model = tf.keras.models.load_model('action_model.h5')
actions = np.array([
    'Help me', 'Danger', 'Emergency', 'Stop', 'Go', 'Call Police', 'Call Ambulance', 
    'Fire', 'Accident', 'Need Doctor', 'Hurt/Injured', 'Stay Here', 'Follow Me', 
    'Safe/Safety', 'Threat', 'Run', 'Yes', 'No', 'Lost', 'Wait', 'Afraid', 
    'Pain', 'Protect', 'Rescue', 'Escape'
])

# Initialize variables for real-time detection
sequence = []
current_phrase = "ASL"  # Default phrase when no confident recognition
threshold = 0.5

# Initialize webcam and Mediapipe model
cap = cv2.VideoCapture(0)
with mp.solutions.hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        image, results = mediapipe_detection(frame, hands)
        draw_styled_landmarks(image, results)

        # Extract keypoints and make predictions
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            if res[np.argmax(res)] > threshold:
                current_phrase = actions[np.argmax(res)]
            else:
                current_phrase = "ASL"  # Reset to default if confidence is below threshold

            # Display the recognized phrase or "ASL" if not confidently recognized
            cv2.putText(image, current_phrase, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Display the confidence score for the current prediction
            cv2.putText(image, f'Confidence: {res[np.argmax(res)]:.2f}', 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        # Show the video feed
        cv2.imshow('OpenCV Feed', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
