from PIL import Image
import matplotlib.pyplot as plt

def display_images(input_image_path, similar_image_paths):
    plt.figure(figsize=(20, 10))
    img = Image.open(input_image_path)
    plt.subplot(1, len(similar_image_paths) + 1, 1)
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis('off')

    for i, image_path in enumerate(similar_image_paths):
        img = Image.open(image_path)
        plt.subplot(1, len(similar_image_paths) + 1, i+2)
        plt.imshow(img)
        plt.title(f"Similar Image {i+1}")
        plt.axis('off')

    plt.show()
