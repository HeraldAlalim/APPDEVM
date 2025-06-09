import cv2
import mediapipe as mp
import numpy as np
import os

LABEL = "Hello"  # Change this to the label you're recording
SEQUENCE_LENGTH = 30
SAVE_DIR = "data"

os.makedirs(SAVE_DIR, exist_ok=True)
existing_files = [f for f in os.listdir(SAVE_DIR) if f.startswith(LABEL)]
sample_index = len(existing_files)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    if results.pose_landmarks:
        return np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten()
    return np.zeros(33 * 3)

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    sequence = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        cv2.imshow("Recording", image)

        if len(sequence) == SEQUENCE_LENGTH:
            filename = f"{LABEL}_{sample_index}.txt"
            np.savetxt(os.path.join(SAVE_DIR, filename), sequence, fmt="%.6f")
            print(f"[SAVED] {filename}")
            sample_index += 1
            sequence = []

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
