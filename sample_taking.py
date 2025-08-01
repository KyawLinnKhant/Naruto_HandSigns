import cv2
import os
import time

# === Settings ===
label = input("Enter gesture label (folder name): ").strip()
save_dir = os.path.join("hand-gesture", label)
os.makedirs(save_dir, exist_ok=True)

box_size = 1000         # Size of center crop square (pixels)
img_size = 320          # Final saved image size
capture_count = 510     # Total images to save
count = 0
started = False

# === Camera Init ===
cap = cv2.VideoCapture(0)
print("\n📷 Press 's' to start capture (3 sec delay)...")

# === Draw green square in center
def draw_center_square(frame, size=300):
    h, w, _ = frame.shape
    x1 = w // 2 - size // 2
    y1 = h // 2 - size // 2
    x2 = x1 + size
    y2 = y1 + size
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return (x1, y1, x2, y2)

# === Main Loop
while cap.isOpened() and count < capture_count:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror camera
    x1, y1, x2, y2 = draw_center_square(frame, size=box_size)

    cv2.putText(frame, f"Saved: {count}/{capture_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Naruto Hand Sign Capture", frame)

    key = cv2.waitKey(1)

    if not started and key == ord('s'):
        print("⏳ Starting in 3 seconds...")
        for i in range(3, 0, -1):
            print(i)
            time.sleep(1)
        started = True
        print("🎥 Capturing started...")

    # === Save cropped image
    if started:
        crop = frame[y1:y2, x1:x2]
        resized = cv2.resize(crop, (img_size, img_size))
        img_name = f"{label}_{count:03d}.jpg"
        cv2.imwrite(os.path.join(save_dir, img_name), resized)
        count += 1

    if key == 27:  # ESC to quit early
        print("🛑 Capture stopped.")
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Done. Saved {count} images to '{save_dir}'")