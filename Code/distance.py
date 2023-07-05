from data_management import hsv_df, rgb_df
from scipy.spatial import distance


def euclidean_distance(df, test_color, top_n):
    if df is hsv_df:
        df['hist_distance'] = df['HSV_Histogram'].apply(lambda x: distance.euclidean(test_color, x))
    elif df is rgb_df:
        df['hist_distance'] = df['RGB_Histogram'].apply(lambda x: distance.euclidean(test_color, x))
    df.sort_values('hist_distance', inplace=True)
    return df.head(top_n)


def manhattan_distance(df, test_color, top_n):
    if df is hsv_df:
        df['hist_distance'] = df['HSV_Histogram'].apply(lambda x: distance.cityblock(test_color, x))
    elif df is rgb_df:
        df['hist_distance'] = df['RGB_Histogram'].apply(lambda x: distance.cityblock(test_color, x))
    df.sort_values('hist_distance', inplace=True)
    return df.head(top_n)


def cosine_similarity(df, test_color, top_n):
    if df is hsv_df:
        df['hist_distance'] = df['HSV_Histogram'].apply(lambda x: distance.cosine(test_color, x))
    elif df is rgb_df:
        df['hist_distance'] = df['RGB_Histogram'].apply(lambda x: distance.cosine(test_color, x))
    df.sort_values('hist_distance', inplace=True)
    return df.head(top_n)
