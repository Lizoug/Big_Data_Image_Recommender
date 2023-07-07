from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import imageio
import glob
import os
import cv2
from tqdm.auto import tqdm
import time
import faiss
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import distance
import sqlite3
import sys

#path = os.getcwd()
#sys.path.append(r"C:\Users\magra\Documents\HSD\4_Semester\Big_Data\Image_Recommender")

from image_recommender_cleaned_18 import euclidean_distance, manhattan_distance, \
    cosine_similarity, build_faiss_index, find_similar_images_faiss, calculate_histogram, \
        load_dataframes, get_paths_from_db, find_all_similarities

# from cleam_code_generator_test_17 import *

# Tests: Euclidean distance, Manhattan distance, Cosine similarity
# Generate dummy data for the input for the test functions
n_rows = 5
ids = np.arange(n_rows)
histograms = np.random.random((n_rows, 10)).tolist()

# Create DataFrame
data = {'ID': ids, 'Histogram': histograms}
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Generate 10 values in a similar range
color_values = np.random.random(10)

# Print the list of color values
print(color_values.tolist())

def test_euclidean_distance(df, test_color, top_n):
    df_copy = df.copy()  # Create a copy of the dataframe
    
    # Call the euclidean_distance function
    result_df = euclidean_distance(df_copy, test_color, top_n)
    
    # Check if the result has the correct length
    assert len(result_df) == top_n, f"Expected {top_n} similar images, but got {len(result_df)}"
    
    # Check if the distances are calculated correctly
    distances = result_df['hist_distance']
    for distance in distances:
        assert isinstance(distance, float), "Distance should be a float value"

    print("euclidean_distance test passed!")

def test_manhattan_distance(df, test_color, top_n):
    df_copy = df.copy()  # Create a copy of the dataframe
    
    # Call the euclidean_distance function
    result_df = manhattan_distance(df_copy, test_color, top_n)
    
    # Check if the result has the correct length
    assert len(result_df) == top_n, f"Expected {top_n} similar images, but got {len(result_df)}"
    
    # Check if the distances are calculated correctly
    distances = result_df['hist_distance']
    for distance in distances:
        assert isinstance(distance, float), "Distance should be a float value"

    print("manhattan_distance test passed!")

def test_cosine_distance(df, test_color, top_n):
    df_copy = df.copy()  # Create a copy of the dataframe
    
    # Call the euclidean_distance function
    result_df = cosine_similarity(df_copy, test_color, top_n)
    
    # Check if the result has the correct length
    assert len(result_df) == top_n, f"Expected {top_n} similar images, but got {len(result_df)}"
    
    # Check if the distances are calculated correctly
    distances = result_df['hist_similarity']
    for distance in distances:
        assert isinstance(distance, float), "Distance should be a float value"

    print("cosine_distance test passed!")

# Tests: functions associated with faiss module
def test_find_similar_images_faiss():
    # Generate some random data for testing
    embeddings_df = pd.DataFrame({
        'ID': [0, 1, 2, 3, 4, 5],
        'Embeddings': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 0.1]]
    })
    test_image_embedding = [0.2, 0.3, 0.4]
    top_n = 3
    
    # Create a sample index
    index = build_faiss_index(embeddings_df)
    
    # Call the find_similar_images_faiss function
    similar_image_ids = find_similar_images_faiss(index, embeddings_df, test_image_embedding, top_n)
    
    # Check if the result has the correct length
    assert len(similar_image_ids) == top_n, f"Expected {top_n} similar image IDs, but got {len(similar_image_ids)}"
    
    # Check if the result contains valid IDs
    assert isinstance(similar_image_ids, np.ndarray), "Similar image IDs should be a numpy array"
    
    print("find_similar_images_faiss test passed!")

# Define a test function to test 'build_faiss_index' function
def test_build_faiss_index():
    # Create a DataFrame with 10 rows of embeddings, each embedding is a list of 300 random numbers
    df = pd.DataFrame({
        'Embeddings': [np.random.rand(300).tolist() for _ in range(10)]
    })

    # Call the 'build_faiss_index' function with the created DataFrame
    index = build_faiss_index(df)

    # Assert that the output is an instance of 'faiss.IndexFlatL2', if not, raise an error with the given error message
    assert isinstance(index, faiss.IndexFlatL2), "Index is not a faiss.IndexFlatL2 instance"

    # Assert that the number of vectors in the index is equal to the number of embeddings in the DataFrame
    # If not, raise an error with the given error message
    assert index.ntotal == len(df), "Number of vectors in index does not match number of embeddings"
    
    print("build_faiss_index test passed!")

# Test: calculating histogram
def test_calculate_histogram():
    # Dummy data
    hsv_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    rgb_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    # Call calculate_histogram function for HSV image
    hist_hsv = calculate_histogram(hsv_image, "hsv")
    assert isinstance(hist_hsv, np.ndarray), "Histogram should be a NumPy array"
    assert hist_hsv.shape == (30 * 3,), "Incorrect shape of HSV histogram"

    # Call calculate_histogram function for RGB image
    hist_rgb = calculate_histogram(rgb_image, "rgb")
    assert isinstance(hist_rgb, np.ndarray), "Histogram should be a NumPy array"
    assert hist_rgb.shape == (30 * 3,), "Incorrect shape of RGB histogram"

    print("calculate_histogram test passed!")

# Test: Load dataframe from pickle file
def test_load_dataframes():
    # Call the load_dataframes function
    embeddings_df, hsv_df, rgb_df = load_dataframes()

    # Check that the outputs are all DataFrames
    assert isinstance(embeddings_df, pd.DataFrame), "embeddings_df is not a pandas DataFrame"
    assert isinstance(hsv_df, pd.DataFrame), "hsv_df is not a pandas DataFrame"
    assert isinstance(rgb_df, pd.DataFrame), "rgb_df is not a pandas DataFrame"

    # Check that the DataFrames have the expected columns
    assert 'Embeddings' in embeddings_df.columns, "embeddings_df does not have expected 'Embeddings' column"
    assert 'Histogram' in hsv_df.columns, "hsv_df does not have expected 'Histogram' column"
    assert 'Histogram' in rgb_df.columns, "rgb_df does not have expected 'Histogram' column"
    
    # Check that the files from which the dataframes are loaded exist
    assert os.path.isfile('ID_Embeddings_2.pkl'), "File 'ID_Embeddings_2.pkl' does not exist"
    assert os.path.isfile('ID_hsv_2.pkl'), "File 'ID_hsv_2.pkl' does not exist"
    assert os.path.isfile('ID_rgb_2.pkl'), "File 'ID_rgb_2.pkl' does not exist"

    # Check that the dataframes are not empty
    assert not embeddings_df.empty, "embeddings_df is empty"
    assert not hsv_df.empty, "hsv_df is empty"
    assert not rgb_df.empty, "rgb_df is empty"
    
    print("load_dataframes test passed!")

# Test: Load image paths from database
def test_get_paths_from_db():
    # Create a connection to the SQLite database
    connection = sqlite3.connect('paths_db_test.sqlite')
    
    # Prepare a test list of IDs
    ids = [111, 654, 0]
    
    # Call the get_paths_from_db function
    paths = get_paths_from_db(connection, ids)
    
    # Check that the output is a list
    assert isinstance(paths, list), "Output is not a list"
    
    # Check that the length of the list is correct
    assert len(paths) == len(ids), "Output list has incorrect length"
    
    # Check that all elements in the list are strings (assuming 'Path' column contains strings)
    assert all(isinstance(path, str) for path in paths), "Not all elements in the output list are strings"

    # Close the database connection
    connection.close()
    
    print("Get_paths_from_db test passed!")

# Test: Find similarities between input image and dataset images
def test_find_all_similarities():
    test_image_path = "test_image.jpg"
    top_n = 5
    type = 'embeddings'
    
    # Test case 1: embeddings
    similar_image_ids = find_all_similarities(test_image_path, top_n, type)
    assert len(similar_image_ids) == top_n, "Test case 1 failed"
    
    # Test case 2: hsv_euclidean
    type = 'hsv_euclidean'
    similar_image_ids = find_all_similarities(test_image_path, top_n, type)
    assert len(similar_image_ids) == top_n, "Test case 2 failed"
    
    # Test case 3: hsv_manhattan
    type = 'hsv_manhattan'
    similar_image_ids = find_all_similarities(test_image_path, top_n, type)
    assert len(similar_image_ids) == top_n, "Test case 3 failed"
    
    # Test case 4: hsv_cosine
    type = 'hsv_cosine'
    similar_image_ids = find_all_similarities(test_image_path, top_n, type)
    assert len(similar_image_ids) == top_n, "Test case 4 failed"
    
    # Test case 5: rgb_euclidean
    type = 'rgb_euclidean'
    similar_image_ids = find_all_similarities(test_image_path, top_n, type)
    assert len(similar_image_ids) == top_n, "Test case 5 failed"
    
    # Test case 6: rgb_manhattan
    type = 'rgb_manhattan'
    similar_image_ids = find_all_similarities(test_image_path, top_n, type)
    assert len(similar_image_ids) == top_n, "Test case 6 failed"
    
    # Test case 7: rgb_cosine
    type = 'rgb_cosine'
    similar_image_ids = find_all_similarities(test_image_path, top_n, type)
    assert len(similar_image_ids) == top_n, "Test case 7 failed"
    
    print("find_all_similarities test passed!")

# Run all tests at once
def run_tests():
    # print messages indicating which test is being run
    # then execute the test function
    
    print("Test euclidean_distance")
    test_euclidean_distance(df, color_values, 3)
    
    print("Test manhattan_distance")
    test_manhattan_distance(df, color_values, 3)
    
    print("Test cosine_distance")
    test_cosine_distance(df, color_values, 3)
    
    print("Test find_similar_images_faiss")
    test_find_similar_images_faiss()
    
    print("Testing build_faiss_index")
    test_build_faiss_index()
    
    print("Testing calculate_histogram")
    test_calculate_histogram()
    
    print("Testing load_dataframes")
    test_load_dataframes()
    
    print("Testing test_get_paths_from_db")
    test_get_paths_from_db()
    
    print("Testing find_all_similarities")
    test_find_all_similarities()

    # If no errors were raised by the assert statements, print a message indicating that all tests passed
    print("All tests passed")

test = run_tests()
