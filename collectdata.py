from function import *
import cv2
import numpy as np
import os
from time import sleep

# Define phrases and sequence settings
actions = np.array(['Help me', 'Thank you'])
no_sequences = 30
sequence_length = 30
DATA_PATH = os.path.join('MP_Data')

# Create directories for data storage
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# Initialize webcam and Mediapipe Hands model
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    for action in actions:
        print(f"Collecting data for: {action}")
        
        for sequence in range(no_sequences):
            print(f"Collecting sequence {sequence + 1}/{no_sequences} for {action}")
            
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                image, results = mediapipe_detection(frame, hands)
                draw_styled_landmarks(image, results)

                # Display phrase and frame number
                cv2.putText(image, f'Perform "{action}" {frame_num + 1}/{sequence_length}', 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)

                # Collect keypoints and save
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            sleep(1)

cap.release()
cv2.destroyAllWindows()
