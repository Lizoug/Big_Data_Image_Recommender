import cv2
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from image_processing import get_image_embedding, calculate_histogram
from data_management import merge_embeddings_and_paths, build_faiss_index
from data_management import embeddings_df, path_df,  hsv_df, rgb_df
from distance import euclidean_distance, manhattan_distance, cosine_similarity
from plots import display_images


def find_similar_images_faiss(index, image_paths, test_image_embedding, top_n):
    D, I = index.search(np.array([test_image_embedding]), top_n)
    top_faiss_paths = image_paths.iloc[I[0]].values
    return top_faiss_paths

def find_all_similarities(test_image_path, top_n=5, type='embeddings'):
    # Load image
    test_image = cv2.imread(test_image_path)

    # Load model
    model = MobileNet(weights='imagenet', include_top=False, pooling='avg')
    
    # get image embeddings
    test_image_embedding = get_image_embedding(test_image_path, model)
    print("Test image embedding dimension:", test_image_embedding.shape)
    
    if type == 'embeddings':
        merged_df = merge_embeddings_and_paths(embeddings_df, path_df)
        index, image_paths = build_faiss_index(merged_df)
        print("FAISS index dimension:", index.d)
        similar_image_paths = find_similar_images_faiss(index, image_paths, test_image_embedding, top_n)

        
    elif type in ['hsv_euclidean', 'hsv_manhattan', 'hsv_cosine']:
        test_color = calculate_histogram(test_image, "hsv")
        if type == 'hsv_euclidean':
            top_images = euclidean_distance(hsv_df, test_color, top_n)
        elif type == 'hsv_manhattan':
            top_images = manhattan_distance(hsv_df, test_color, top_n)
        elif type == 'hsv_cosine':
            top_images = cosine_similarity(hsv_df, test_color, top_n)
        similar_image_paths = path_df[path_df['ID'].isin(top_images['ID'])]['Path'].values.tolist()
    
    elif type in ['rgb_euclidean', 'rgb_manhattan', 'rgb_cosine']:
        test_color = calculate_histogram(test_image, "rgb")
        if type == 'rgb_euclidean':
            top_images = euclidean_distance(rgb_df, test_color, top_n)
        elif type == 'rgb_manhattan':
            top_images = manhattan_distance(rgb_df, test_color, top_n)
        elif type == 'rgb_cosine':
            top_images = cosine_similarity(rgb_df, test_color, top_n)
        similar_image_paths = path_df[path_df['ID'].isin(top_images['ID'])]['Path'].values.tolist()

    display_images(test_image_path, similar_image_paths)
