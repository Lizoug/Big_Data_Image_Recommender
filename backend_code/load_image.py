import os
import cv2

def image_generator(folder_path):
    """
    A generator that yields image arrays and their unique ids from a folder of images.
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(root, file)
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Failed to read image file: {image_path}")
                    else:
                        image_id = os.path.splitext(file)[0]  # use file name (without extension) as id
                        image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        yield image_id, image_array
                except Exception as e:
                    print(f"Error loading image file {image_path}: {str(e)}")


folder_path = r"C:\Users\lizak\Data_Science\Semester_4\Big_Data\Projekt\images\weather_image_recognition"

image_generator(folder_path)
