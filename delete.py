import os

base_dir = "hand-gesture"

for gesture_folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, gesture_folder)
    if os.path.isdir(folder_path):
        print(f"Cleaning: {folder_path}")

        for i in range(10):  # _000.jpg to _009.jpg
            suffix = f"_{i:03d}.jpg"
            for filename in os.listdir(folder_path):
                if filename.endswith(suffix):
                    file_path = os.path.join(folder_path, filename)
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")