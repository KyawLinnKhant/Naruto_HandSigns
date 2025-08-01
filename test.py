import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from deepface import DeepFace
import os
import time
from datetime import datetime

# === Load Hand Gesture Model ===
model = load_model("naruto_sign_finetuned.h5")
class_names = [
    "Hitsuji[Ram]", "I[Boar]", "Inu[Dog]", "Mi[Snake]", "Ne[Rat]",
    "Saru[Monkey]", "Tatsu[Dragon]", "Tora[Tiger]", "Tori[Bird]",
    "U[Hare]", "Uma[Horse]", "Ushi[Ox]"
]

# === Sequence Setup ===
sequence_order = ["Inu[Dog]", "Uma[Horse]", "Ne[Rat]"]
sequence_index = 0
hold_start_time = None
last_label = None
hold_duration = 1.0  # seconds
screenshot_dir = "screenshots"
os.makedirs(screenshot_dir, exist_ok=True)

# === MediaPipe Setup ===
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
hand_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# === Webcam Setup ===
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
input_size = 320

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # === Face Emotion Detection ===
    face_results = face_detector.process(rgb)
    if face_results.detections:
        for det in face_results.detections:
            bbox = det.location_data.relative_bounding_box
            fx, fy = int(bbox.xmin * w), int(bbox.ymin * h)
            fw, fh = int(bbox.width * w), int(bbox.height * h)
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 255), 2)
            try:
                face_crop = frame[fy:fy + fh, fx:fx + fw]
                result = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                cv2.putText(frame, f"Face: {emotion}", (fx, fy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            except:
                pass

    # === Hand Detection ===
    hand_results = hand_detector.process(rgb)
    if hand_results.multi_hand_landmarks:
        x_all, y_all = [], []
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x_all.append(int(lm.x * w))
                y_all.append(int(lm.y * h))

        if x_all and y_all:
            x1, y1 = max(min(x_all) - 40, 0), max(min(y_all) - 40, 0)
            x2, y2 = min(max(x_all) + 40, w), min(max(y_all) + 40, h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            try:
                roi = frame[y1:y2, x1:x2]
                roi_resized = cv2.resize(roi, (input_size, input_size))
                input_img = np.expand_dims(roi_resized / 255.0, axis=0)
                pred = model.predict(input_img, verbose=0)[0]
                label = class_names[np.argmax(pred)]
                conf = np.max(pred)

                # Show prediction
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

                expected_label = sequence_order[sequence_index]

                if label == expected_label:
                    if hold_start_time is None:
                        hold_start_time = time.time()
                    elif time.time() - hold_start_time >= hold_duration:
                        sequence_index += 1
                        print(f"[✔] Step {sequence_index}: {label}")
                        hold_start_time = None
                        last_label = label
                else:
                    hold_start_time = None
                    if label in sequence_order:
                        sequence_index = 1 if label == sequence_order[0] else 0
                    else:
                        sequence_index = 0
                    last_label = label

                # Screenshot if full sequence is complete
                if sequence_index == len(sequence_order):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{timestamp}.png"
                    path = os.path.join(screenshot_dir, filename)
                    cv2.imwrite(path, frame)
                    print(f"[📸] Screenshot saved: {path}")
                    cv2.putText(frame, "Screenshot Taken!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    sequence_index = 0
                    hold_start_time = None
                    last_label = None

            except Exception as e:
                print("[!] Error:", e)
                hold_start_time = None

    else:
        hold_start_time = None

    # === Show Window ===
    cv2.imshow("Naruto Sign Sequence Trigger", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()