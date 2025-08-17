import os
import shutil

def clear_directory(dir_path):
    """
    Deletes all files and subdirectories within a directory, and then deletes the directory itself.
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"Successfully cleared and removed {dir_path}")