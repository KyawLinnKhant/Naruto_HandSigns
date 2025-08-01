import cv2
import os
import numpy as np
import random

base_dir = "hand-gesture"

for gesture_folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, gesture_folder)
    if os.path.isdir(folder_path):
        print(f"\nProcessing: {folder_path}")
        for filename in os.listdir(folder_path):
            if not filename.endswith(".jpg"):
                continue

            name_no_ext = os.path.splitext(filename)[0]
            if any(suffix in name_no_ext for suffix in [
                "_flip", "_rot", "_bright", "_contrast", "_blur", "_noise", "_zoom", "_translate", "_shear", "_hsv"
            ]):
                continue  # Skip already augmented

            path = os.path.join(folder_path, filename)
            img = cv2.imread(path)
            if img is None:
                continue

            h, w = img.shape[:2]

            # Horizontal Flip
            flipped = cv2.flip(img, 1)
            cv2.imwrite(os.path.join(folder_path, f"{name_no_ext}_flip.jpg"), flipped)

            # Rotation
            angle = random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
            rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            cv2.imwrite(os.path.join(folder_path, f"{name_no_ext}_rot.jpg"), rotated)

            # Brightness
            bright = cv2.convertScaleAbs(img, alpha=1, beta=40)
            cv2.imwrite(os.path.join(folder_path, f"{name_no_ext}_bright.jpg"), bright)

            # Contrast
            contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
            cv2.imwrite(os.path.join(folder_path, f"{name_no_ext}_contrast.jpg"), contrast)

            # Blur
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            cv2.imwrite(os.path.join(folder_path, f"{name_no_ext}_blur.jpg"), blur)

            # Noise
            noise = img.copy()
            noise = noise.astype(np.int16)
            noise += np.random.normal(0, 20, noise.shape).astype(np.int16)
            noise = np.clip(noise, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(folder_path, f"{name_no_ext}_noise.jpg"), noise)

            # Zoom (random crop and resize back)
            scale = random.uniform(1.1, 1.2)
            new_w, new_h = int(w / scale), int(h / scale)
            start_x = (w - new_w) // 2
            start_y = (h - new_h) // 2
            zoomed = img[start_y:start_y+new_h, start_x:start_x+new_w]
            zoomed = cv2.resize(zoomed, (w, h))
            cv2.imwrite(os.path.join(folder_path, f"{name_no_ext}_zoom.jpg"), zoomed)

            # Translation
            tx = random.randint(-20, 20)
            ty = random.randint(-20, 20)
            M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
            translated = cv2.warpAffine(img, M_trans, (w, h), borderMode=cv2.BORDER_REFLECT)
            cv2.imwrite(os.path.join(folder_path, f"{name_no_ext}_translate.jpg"), translated)

            # Shear
            shear_factor = random.uniform(-0.2, 0.2)
            M_shear = np.float32([[1, shear_factor, 0], [0, 1, 0]])
            sheared = cv2.warpAffine(img, M_shear, (w, h), borderMode=cv2.BORDER_REFLECT)
            cv2.imwrite(os.path.join(folder_path, f"{name_no_ext}_shear.jpg"), sheared)

            # HSV Color Jitter
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 0] += random.uniform(-10, 10)
            hsv[..., 1] *= random.uniform(0.9, 1.1)
            hsv[..., 2] *= random.uniform(0.9, 1.1)
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            hsv_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(os.path.join(folder_path, f"{name_no_ext}_hsv.jpg"), hsv_img)

            print(f"✅ Augmented: {name_no_ext}")