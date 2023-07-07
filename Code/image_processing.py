import cv2
from keras.applications.mobilenet import preprocess_input
import numpy as np


# Global variables
global img_width, img_height
img_width, img_height = 224, 224

def get_image_embedding(image_path, model):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (img_width, img_height)) 
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    embedding = model.predict(image)
    return embedding.flatten()


def calculate_histogram(image, color_mode):
    if color_mode == "hsv":
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # BGR to HSV
        hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8,8,8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()  # normalize and flatten the histogram
    elif color_mode == "rgb":
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR to RGB
        hist = cv2.calcHist([rgb_image], [0, 1, 2], None, [8,8,8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
    else:
        raise ValueError(f"Invalid color_mode: {color_mode}")
    return hist
