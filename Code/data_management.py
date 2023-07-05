import pandas as pd
import numpy as np
import faiss

def load_dataframes():
    path_df = pd.read_pickle('ID_path.pkl')
    embeddings_df = pd.read_pickle('ID_Embedings.pkl')
    hsv_df = pd.read_pickle('ID_hsv.pkl')
    rgb_df = pd.read_pickle('ID_rgb.pkl')
    return path_df, embeddings_df, hsv_df, rgb_df

def update_path(path):
    return path.replace('E:', 'D:')

def update_paths_in_df(df, path_column='Path'):
    df[path_column] = df[path_column].apply(lambda x: update_path(x))
    return df

def merge_embeddings_and_paths(embeddings_df, path_df):
    merged_df = pd.merge(embeddings_df, path_df, on='ID')
    return merged_df

def build_faiss_index(merged_df):
    image_paths = merged_df['Path']
    embeddings = np.array(merged_df['Embeddings'].to_list()).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, image_paths
