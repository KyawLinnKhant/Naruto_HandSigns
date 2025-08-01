import sys
import os
import cv2
import time
import numpy as np
import mediapipe as mp
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QDialog, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from tensorflow.keras.models import load_model
from deepface import DeepFace
from ui_kyaw import Ui_Dialog

model = load_model("naruto_sign_finetuned.h5")
class_names = [
    "Hitsuji[Ram]", "I[Boar]", "Inu[Dog]", "Mi[Snake]", "Ne[Rat]",
    "Saru[Monkey]", "Tatsu[Dragon]", "Tora[Tiger]", "Tori[Bird]",
    "U[Hare]", "Uma[Horse]", "Ushi[Ox]"
]

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
hand_detector = mp_hands.Hands(False, 2, 1, 0.7, 0.5)
face_mesh = mp_face.FaceMesh(False, 1, True, 0.5)
pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class MainApp(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.input_size = 320
        self.running = False
        self.last_time = time.time()
        self.fps = 0
        self.zoom_factor = 1.0

        self.current_emotion = None
        self.current_label = None

        self.scene_mp = QGraphicsScene()
        self.ui.MPview.setScene(self.scene_mp)
        self.pixmap_item_mp = QGraphicsPixmapItem()
        self.scene_mp.addItem(self.pixmap_item_mp)

        self.ui.StartPB.clicked.connect(self.start_camera)
        self.ui.StopPB.clicked.connect(self.stop_camera)
        self.ui.QuitPB.clicked.connect(self.close)
        self.ui.SnapPB.clicked.connect(self.take_snapshot)
        self.ui.Slider.valueChanged.connect(self.update_threshold)

        self.conf_threshold = 0.7
        self.ui.Slider.setValue(int(self.conf_threshold * 100))
        self.ui.THtextEdit.setPlainText(str(self.conf_threshold))

        self.ui.Manual.setText(
            "Smile + Tori → Snapshot\n"
            "Angry + Inu → Volume\n"
            "Surprise + Monkey → Zoom"
        )
        self.ui.Manual.setStyleSheet("color: white; font-size: 14px;")

        self.snapshot_triggered = False
        self.countdown_overlay = ""
        self.countdown_step = 0
        self.countdown_active = False
        self.hold_start_time = None
        self.last_label = None
        self.hold_duration = 1.0

        self.ui.GD.stateChanged.connect(self.mode_toggle_logic)
        self.ui.BR.stateChanged.connect(self.mode_toggle_logic)
        self.ui.Vol.stateChanged.connect(self.mode_toggle_logic)

    def detect_naruto_gesture(self, frame, rgb, hands_result):
        h, w, _ = frame.shape

        x_all, y_all = [], []
        for hand_landmarks in hands_result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x_all.append(int(lm.x * w))
                y_all.append(int(lm.y * h))

        if x_all and y_all:
            x_min, x_max = min(x_all), max(x_all)
            y_min, y_max = min(y_all), max(y_all)

            # === Step 1: Square Box Centered ===
            box_size = int(1.5 * max(x_max - x_min, y_max - y_min))  # expand more if needed
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            x1 = max(center_x - box_size // 2, 0)
            y1 = max(center_y - box_size // 2, 0)
            x2 = min(x1 + box_size, w)
            y2 = min(y1 + box_size, h)

            # Adjust again to ensure square box if clipping occurred
            box_size = min(x2 - x1, y2 - y1)
            x2 = x1 + box_size
            y2 = y1 + box_size

            # === Step 2: Draw and Classify ===
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            try:
                roi = frame[y1:y2, x1:x2]
                roi_resized = cv2.resize(roi, (self.input_size, self.input_size))
                input_img = np.expand_dims(roi_resized / 255.0, axis=0)
                pred = model.predict(input_img, verbose=0)[0]
                label = class_names[np.argmax(pred)]
                conf = np.max(pred)
                self.current_label = label

                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

                expected_label = self.sequence_order[self.sequence_index]
                if label == expected_label:
                    if self.hold_start_time is None:
                        self.hold_start_time = time.time()
                    elif time.time() - self.hold_start_time >= self.hold_duration:
                        self.sequence_index += 1
                        print(f"[✔] Step {self.sequence_index}: {label}")
                        self.hold_start_time = None
                        self.last_label = label
                else:
                    self.hold_start_time = None
                    if label in self.sequence_order:
                        self.sequence_index = 1 if label == self.sequence_order[0] else 0
                    else:
                        self.sequence_index = 0
                    self.last_label = label

                if self.sequence_index == len(self.sequence_order):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    os.makedirs("screenshots", exist_ok=True)
                    path = os.path.join("screenshots", f"screenshot_{timestamp}.png")
                    cv2.imwrite(path, frame)
                    print(f"[📸] Screenshot saved: {path}")
                    cv2.putText(frame, "Screenshot Taken!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    self.sequence_index = 0
                    self.hold_start_time = None
                    self.last_label = None

            except Exception as e:
                print("[!] Error:", e)
                self.hold_start_time = None

    def mode_toggle_logic(self):
        sender = self.sender()
        if sender.isChecked():
            if sender == self.ui.GD:
                self.ui.BR.setChecked(False)
                self.ui.Vol.setChecked(False)
            elif sender == self.ui.BR:
                self.ui.GD.setChecked(False)
                self.ui.Vol.setChecked(False)
            elif sender == self.ui.Vol:
                self.ui.GD.setChecked(False)
                self.ui.BR.setChecked(False)

    def update_threshold(self):
        self.conf_threshold = self.ui.Slider.value() / 100
        self.ui.THtextEdit.setPlainText(str(round(self.conf_threshold, 2)))

    def start_camera(self):
        if not self.running:
            self.running = True
            self.timer.start(30)

    def stop_camera(self):
        self.running = False
        self.timer.stop()

    def take_snapshot(self):
        if hasattr(self, 'current_frame'):
            os.makedirs("screenshot", exist_ok=True)
            filename = os.path.join("screenshot", datetime.now().strftime("snap_%Y%m%d_%H%M%S.png"))
            cv2.imwrite(filename, self.current_frame)
            print(f"[📸] Saved snapshot: {filename}")

    def flash_gui(self):
        self.setStyleSheet("background-color: white;")
        QTimer.singleShot(100, lambda: self.setStyleSheet(""))

    def trigger_countdown(self):
        if self.snapshot_triggered or self.countdown_active:
            return
        self.countdown_step = 3
        self.countdown_active = True
        self.countdown_tick()

    def countdown_tick(self):
        if self.countdown_step > 0:
            self.countdown_overlay = str(self.countdown_step)
            self.countdown_step -= 1
            QTimer.singleShot(1000, self.countdown_tick)
        else:
            self.countdown_overlay = ""
            self.take_snapshot()
            self.flash_gui()
            self.ui.THtextEdit.append("📸 Snapshot triggered by Smile + Tori")
            QTimer.singleShot(3000, lambda: self.ui.Manual.setText(
                "Smile + Tori → Snapshot\n"
                "Angry + Inu → Volume\n"
                "Surprise + Monkey → Zoom"
            ))
            self.snapshot_triggered = True
            self.countdown_active = False
            QTimer.singleShot(3000, lambda: setattr(self, 'snapshot_triggered', False))

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # Default fallback center
        self.hand_center = (w // 2, h // 2)

        # === Hand center for zoom centering ===
        if self.ui.GD.isChecked() or self.ui.Vol.isChecked() or self.ui.BR.isChecked():
            hands = hand_detector.process(rgb)
            if hands.multi_hand_landmarks:
                for hand_landmarks in hands.multi_hand_landmarks:
                    x_all = [int(lm.x * w) for lm in hand_landmarks.landmark]
                    y_all = [int(lm.y * h) for lm in hand_landmarks.landmark]
                    cx = (min(x_all) + max(x_all)) // 2
                    cy = (min(y_all) + max(y_all)) // 2
                    self.hand_center = (cx, cy)
                    break

        # === Apply zoom if needed ===
        if self.zoom_factor > 1.0 and hasattr(self, 'hand_center'):
            center_x, center_y = self.hand_center
            new_w, new_h = int(w / self.zoom_factor), int(h / self.zoom_factor)
            x1 = max(center_x - new_w // 2, 0)
            y1 = max(center_y - new_h // 2, 0)
            x2 = min(x1 + new_w, w)
            y2 = min(y1 + new_h, h)
            x1 = max(x2 - new_w, 0)
            y1 = max(y2 - new_h, 0)
            frame = frame[y1:y2, x1:x2]
            frame = cv2.resize(frame, (w, h))
        else:
            self.hand_center = (w // 2, h // 2)

        self.current_frame = frame.copy()
        self.current_emotion = None
        self.current_label = None

        pose = pose_detector.process(rgb)
        black_mask = np.zeros_like(frame)
        if pose.segmentation_mask is not None:
            seg_mask = pose.segmentation_mask > 0.1
            black_mask[seg_mask] = (50, 50, 50)
            if pose.pose_landmarks:
                mp_draw.draw_landmarks(black_mask, pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                       mp_draw.DrawingSpec((255, 0, 0), 2, 2), mp_draw.DrawingSpec((0, 255, 255), 2))

        # === Facial emotion detection ===
        if self.ui.FR.isChecked():
            face_results = face_mesh.process(rgb)
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_draw.draw_landmarks(black_mask, face_landmarks, mp_face.FACEMESH_TESSELATION,
                                           mp_draw.DrawingSpec((0, 255, 0), 1, 1), mp_draw.DrawingSpec((0, 255, 0), 1))
                try:
                    bbox = face_results.multi_face_landmarks[0].landmark
                    xs = [int(p.x * w) for p in bbox]
                    ys = [int(p.y * h) for p in bbox]
                    x1, y1 = min(xs), min(ys)
                    x2, y2 = max(xs), max(ys)
                    face_crop = frame[y1:y2, x1:x2]
                    result = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
                    emotion = result[0]['dominant_emotion']
                    confidence = result[0]['emotion'][emotion] / 100
                    if confidence >= self.conf_threshold:
                        self.current_emotion = emotion.lower()
                        cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                except Exception as e:
                    print("[!] DeepFace Error:", e)

        # === Unified hand gesture recognition (combined hands) ===
        if self.ui.GD.isChecked() or self.ui.Vol.isChecked() or self.ui.BR.isChecked():
            hands = hand_detector.process(rgb)
            if hands.multi_hand_landmarks:

                if self.ui.GD.isChecked():
                    self.detect_naruto_gesture(frame, rgb, hands)

                # === Zoom or Volume control still uses single-hand logic ===
                if self.ui.BR.isChecked() or self.ui.Vol.isChecked():
                    lm = hands.multi_hand_landmarks[0].landmark
                    p1 = int(lm[4].x * w), int(lm[4].y * h)
                    p2 = int(lm[8].x * w), int(lm[8].y * h)
                    length = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
                    bar_value = int(np.interp(length, [30, 200], [0, 100]))

                    if self.ui.BR.isChecked():
                        self.zoom_factor = 1.0 + (bar_value / 100.0)
                        cv2.rectangle(frame, (50, 150), (85, 400), (0, 0, 0), 2)
                        vol_bar = int(np.interp(length, [30, 200], [400, 150]))
                        cv2.rectangle(frame, (50, vol_bar), (85, 400), (255, 0, 255), -1)
                        cv2.putText(frame, f"Zoom: {bar_value}%", (40, 430),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                    elif self.ui.Vol.isChecked():
                        try:
                            volume_percent = int(np.clip(bar_value, 0, 100))
                            os.system(f"osascript -e 'set volume output volume {volume_percent}'")
                            cv2.rectangle(frame, (50, 150), (85, 400), (0, 0, 0), 2)
                            vol_bar = int(np.interp(length, [30, 200], [400, 150]))
                            cv2.rectangle(frame, (50, vol_bar), (85, 400), (255, 255, 0), -1)
                            cv2.putText(frame, f"Vol: {volume_percent}%", (40, 430),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        except Exception as e:
                            print("[!] Volume Error:", e)

                    color = (255, 255, 0) if self.ui.Vol.isChecked() else (255, 0, 255)
                    mp_draw.draw_landmarks(black_mask, hands.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS,
                                           mp_draw.DrawingSpec(color, 2, 2),
                                           mp_draw.DrawingSpec((0, 0, 0), 2, 1))

        # === Trigger: Screenshot
        if self.current_emotion in ["happy", "smile"] and self.current_label == "Tori[Bird]":
            self.trigger_countdown()
            self.ui.THtextEdit.append("📸 Snapshot triggered by Smile + Tori")

        # === Trigger: Volume Control
        elif self.current_emotion == "angry" and self.current_label == "Inu[Dog]":
            if not self.ui.Vol.isChecked():
                self.ui.Vol.setChecked(True)
                self.ui.THtextEdit.append("📢 Volume control enabled by Angry + Inu")

        # === Trigger: Zoom Control
        elif self.current_emotion == "surprise" and self.current_label == "Saru[Monkey]":
            if not self.ui.BR.isChecked():
                self.ui.BR.setChecked(True)
                self.ui.THtextEdit.append("🔍 Zoom control enabled by Surprise + Monkey")

        if self.countdown_overlay:
            cv2.putText(frame, self.countdown_overlay, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (180, 0, 255), 4)

        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 * (1 / (now - self.last_time))
        self.last_time = now
        cv2.putText(frame, f"{self.fps:.1f} FPS", (w - 160, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 50, 50), 2, cv2.LINE_AA)

        q_img = QImage(frame.data, w, h, frame.strides[0], QImage.Format_BGR888)
        self.ui.MainView.setPixmap(QPixmap.fromImage(q_img))

        black_rgb = cv2.cvtColor(black_mask, cv2.COLOR_BGR2RGB)
        mp_qimg = QImage(black_rgb.data, black_rgb.shape[1], black_rgb.shape[0], black_rgb.strides[0],
                         QImage.Format_RGB888)
        self.pixmap_item_mp.setPixmap(QPixmap.fromImage(mp_qimg))
        self.ui.MPview.fitInView(self.pixmap_item_mp, Qt.KeepAspectRatio)

    def closeEvent(self, event):
        self.cap.release()
        self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainApp()
    win.show()
    sys.exit(app.exec_())