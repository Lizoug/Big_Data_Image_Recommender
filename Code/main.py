from image_similarity import update_paths_in_df, find_all_similarities
from data_management import load_dataframes
from plots import display_images

# Global variables
img_width, img_height = 224, 224
train_data_dir = "D:\images"
batch_size = 100

def main():
    global embeddings_df, hsv_df, rgb_df

    # Load data
    embeddings_df, hsv_df, rgb_df = load_dataframes()
    
    # Run the image similarity search
    similar_image_ids = find_all_similarities("test.jpg", top_n=6, type='embeddings')
    print(similar_image_ids)
    # Get the paths from the SQLite database
    connection = sqlite3.connect('paths_db_3.sqlite')
    similar_image_paths = get_paths_from_db(connection, similar_image_ids)
    
    # Display images
    display_images("test.jpg", similar_image_paths)

if __name__ == '__main__':
    main()
