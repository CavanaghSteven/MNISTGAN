import os
import cv2


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
