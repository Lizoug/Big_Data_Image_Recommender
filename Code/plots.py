from PIL import Image
import matplotlib.pyplot as plt


def display_input_and_similar_images(input_image_path, similar_image_paths):
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


def display_dimension_reduction_images(similar_image_paths, similar_image_ids):
    plt.figure(figsize=(20, 10))

    for i, image_path in enumerate(similar_image_paths):
        img = Image.open(image_path)
        plt.subplot(1, len(similar_image_paths), i+1)
        plt.imshow(img)
        plt.title(f"Image ID: {similar_image_ids[i]}")  # Use the ID as the title
        plt.axis('off')
        
    plt.show()
