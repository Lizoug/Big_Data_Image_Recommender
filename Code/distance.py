from data_management import hsv_df, rgb_df
from scipy.spatial import distance


def euclidean_distance(df, test_color, top_n):
    df['hist_distance'] = df['Histogram'].apply(lambda x: distance.euclidean(test_color, x))
    df.sort_values('hist_distance', inplace=True, ascending=True)
    return df.head(top_n)

def manhattan_distance(df, test_color, top_n):
    df['hist_distance'] = df['Histogram'].apply(lambda x: distance.cityblock(test_color, x))
    df.sort_values('hist_distance', inplace=True, ascending=True)
    return df.head(top_n)

def cosine_similarity(df, test_color, top_n):
    df['hist_similarity'] = df['Histogram'].apply(lambda x: 1 - distance.cosine(test_color, x))
    df.sort_values('hist_similarity', inplace=True, ascending=False)
    return df.head(top_n)
