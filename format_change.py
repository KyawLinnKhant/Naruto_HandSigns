import os
base_dir = "hand-gesture"
def clean_names(base_dir):
    for root, dirs, files in os.walk(base_dir, topdown=False):
        # Rename files
        for name in files:
            new_name = name.replace('[', '_').replace(']', '_')
            if name != new_name:
                os.rename(os.path.join(root, name), os.path.join(root, new_name))

        # Rename folders
        for name in dirs:
            new_name = name.replace('[', '_').replace(']', '_')
            if name != new_name:
                os.rename(os.path.join(root, name), os.path.join(root, new_name))

clean_names("hand-gesture")