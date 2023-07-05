from image_similarity import update_paths_in_df, find_all_similarities
from data_management import load_dataframes
from plots import display_images

# Global variables
img_width, img_height = 224, 224
train_data_dir = "D:\images"
batch_size = 100

def main():
    # Load data
    path_df, embeddings_df, hsv_df, rgb_df = load_dataframes()

    # Update paths in DataFrames
    path_df = update_paths_in_df(path_df, 'Path')

    # Run the image similarity search
    similar_image_paths = find_all_similarities("wiese.jpg", top_n=6, type='hsv_manhattan')

    # Display images
    display_images("wiese.jpg", similar_image_paths)

if __name__ == '__main__':
    main()
