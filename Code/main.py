from image_similarity import find_all_similarities
from data_management import load_dataframes
from plots import display_input_and_similar_images
import sqlite3
from database import get_paths_from_db

# Global variables
img_width, img_height = 224, 224
train_data_dir = "D:\images"
batch_size = 100

def main():
    global embeddings_df, hsv_df, rgb_df

    # Load data
    embeddings_df, hsv_df, rgb_df = load_dataframes()
    
    # Run the image similarity search
    test_image_path = "wiese.jpg"
    similar_image_ids = find_all_similarities(test_image_path, top_n=6, type='rgb_cosine')
    print(similar_image_ids)
    
    # Get the paths from the SQLite database
    connection = sqlite3.connect('paths_db_drive_D.sqlite')
    similar_image_paths = get_paths_from_db(connection, similar_image_ids)
    
    # Display images
    display_input_and_similar_images(test_image_path, similar_image_paths)


if __name__ == '__main__':
    main()
