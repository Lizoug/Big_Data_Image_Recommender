import pandas as pd
import numpy as np
import faiss


def load_dataframes():
    embeddings_df = pd.read_pickle('ID_Embeddings_3.pkl')
    hsv_df = pd.read_pickle('ID_hsv_3.pkl')
    rgb_df = pd.read_pickle('ID_rgb_3.pkl')
    return embeddings_df, hsv_df, rgb_df


def update_path(path):
    return path.replace('E:', 'D:')


def update_paths_in_df(df, path_column='Path'):
    df[path_column] = df[path_column].apply(lambda x: update_path(x))
    return df


def build_faiss_index(embeddings_df):
    embeddings = np.array(embeddings_df['Embeddings'].to_list()).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index
