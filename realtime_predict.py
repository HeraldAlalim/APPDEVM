import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json

SEQUENCE_LENGTH = 30
model = tf.keras.models.load_model("best_model.h5")

with open("label_map.json") as f:
    label_map = json.load(f)
rev_label_map = {v: k for k, v in label_map.items()}

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

sequence = []
current_prediction = ""
prediction_timer = 0

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark] if results.pose_landmarks else np.zeros((33, 3))).flatten()
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark] if results.left_hand_landmarks else np.zeros((21, 3))).flatten()
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark] if results.right_hand_landmarks else np.zeros((21, 3))).flatten()
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark] if results.face_landmarks else np.zeros((468, 3))).flatten()
    return np.concatenate([pose, left_hand, right_hand, face])

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.face_landmarks:
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        if len(sequence) > SEQUENCE_LENGTH:
            sequence.pop(0)

        if len(sequence) == SEQUENCE_LENGTH:
            input_seq = np.expand_dims(sequence, axis=0)
            probs = model.predict(input_seq)[0]
            pred_class = np.argmax(probs)
            confidence = probs[pred_class]

            if confidence > 0.5:
                current_prediction = f"{rev_label_map[pred_class]} ({confidence*100:.1f}%)"
                prediction_timer = 30

        if prediction_timer > 0:
            prediction_timer -= 1
            cv2.putText(image, current_prediction, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Real-Time Sign Prediction", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
