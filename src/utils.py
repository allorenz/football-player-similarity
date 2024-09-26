import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PATH_PLAYER_STATS = '../data/fbref/player_stats.csv'
PATH_TEAM_STATS = '../data/fbref/squad_opponent_stats.csv'


def load_player_statistics():
    # load and set index
    df = pd.read_csv(PATH_PLAYER_STATS)
    df = df.set_index(df['Player'], drop=True)
    df = df.fillna(0)

    # descriptive information
    player_info_cols = ['unique_id','Season', 'League', 'Team', 'Player', 'Nation', 'Pos','Global Pos', 'Age','Matches Played','Playing Time_Starts','Playing Time_Min','Playing Time_90s']
    df_player_info = df[player_info_cols]

    # features
    features = [col for col in df.columns if col not in player_info_cols]
    df_features = df[features]

    return df_features, df_player_info

def load_team_statistics():
    cols_to_drop = ['Nation','Pos','Global Pos']

    # load 
    df = pd.read_csv(PATH_TEAM_STATS)
    df = df.set_index(df['Team'], drop=False)
    df = df.fillna(0)

    # fitler
    df_squad = df[df['Status'] == 'Squad Total']
    df_opponent = df[df['Status'] == 'Opponent Total']

    # descriptive information
    team_info_cols = ['Season', 'League', 'Team','Status',  'Age','Matches Played','Playing Time_Starts','Playing Time_Min','Playing Time_90s']
    df_team_info_squad = df_squad[team_info_cols]
    df_team_info_opponent = df_opponent[team_info_cols]

    # features
    features = [col for col in df.columns if col not in team_info_cols]
    df_squad_features = df_squad[features].drop(cols_to_drop,axis=1)
    df_opponent_features = df_opponent[features].drop(cols_to_drop,axis=1)

    return df_team_info_squad, df_team_info_opponent, df_squad_features, df_opponent_features


def get_top_k_similar_players(embeddings, query_index, player_info, top_k=10, distance_metric='cosine'):
    """
    Compute the top-k most similar players based on cosine similarity or Euclidean distance of embeddings.

    Args:
        embeddings (np.ndarray): Array of player embeddings, shape (num_players, embedding_dim).
        query_index (int): Index of the player to be used as the query point.
        player_info (pd.DataFrame): DataFrame containing player information.
        top_k (int): Number of top similar players to return.
        distance_metric (str): Metric to use for similarity, 'cosine' for cosine similarity or 'euclidean' for Euclidean distance.

    Returns:
        pd.DataFrame: DataFrame of the top_k similar players with their distances or similarities.
    """
    # Normalize the embeddings (L2 normalization) if using cosine similarity
    if distance_metric == 'cosine':
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        query_embedding = embeddings_norm[query_index]

        # Compute cosine similarities between the query and the rest
        similarities = np.dot(embeddings_norm, query_embedding)

        # Sort in descending order (higher cosine similarity means more similar)
        top_k_similar_indices = np.argsort(similarities)[::-1]
        metric_label = 'cosine_similarity'
    
    elif distance_metric == 'euclidean':
        query_embedding = embeddings[query_index]
        
        # Compute Euclidean distances between the query and the rest
        distances = np.linalg.norm(embeddings - query_embedding, axis=1)

        # Sort in ascending order (smaller Euclidean distance means more similar)
        top_k_similar_indices = np.argsort(distances)
        similarities = distances  # Treat distances as similarities for further usage
        metric_label = 'euclidean_distance'

    else:
        raise ValueError("Invalid distance_metric. Use 'cosine' or 'euclidean'.")

    # Get the top_k most similar players, including the query player itself
    top_k_similar_indices = top_k_similar_indices[:top_k]

    # Extract the player information for the top-k most similar players
    top_k_df = player_info.iloc[top_k_similar_indices].copy()

    # Add the similarity or distance column to the DataFrame
    top_k_df[metric_label] = similarities[top_k_similar_indices]

    # Sort the DataFrame by the similarity or distance in the correct order
    if distance_metric == 'cosine':
        top_k_df = top_k_df.sort_values(metric_label, ascending=False)  # Cosine similarity sorted descending
    else:
        top_k_df = top_k_df.sort_values(metric_label, ascending=True)  # Euclidean distance sorted ascending

    # Optionally drop the 'unique_id' column if it exists
    top_k_df = top_k_df.drop(columns='unique_id', errors='ignore')

    return top_k_df

def scatterplot_top_k(df, x_col, y_col, num_labels=10, top_size=200, rest_size=100, fontsize=9, dist_measure='', title=None):
    """
    Creates a scatter plot with custom coloring, labels, and different sizes for top and rest observations.
    
    Parameters:
    - df: pandas DataFrame containing the data.
    - x_col: string, name of the column for the x-axis.
    - y_col: string, name of the column for the y-axis.
    - num_labels: int, number of points to label (default: 10).
    - top_size: int, size of the scatter points for the first `num_labels` observations.
    - rest_size: int, size of the scatter points for the remaining observations.
    """
    # Ensure num_labels is less than the length of the dataframe
    num_labels = min(num_labels, len(df) - 1)

    # Define the color map: first one purple, next num_labels orange, and the rest grey
    colors = ['purple'] + ['orange'] * num_labels + ['grey'] * (len(df) - (num_labels + 1))

    # Define sizes for top and rest
    sizes = [top_size] * (num_labels + 1) + [rest_size] * (len(df) - (num_labels + 1))

    # Ensure sizes and colors are the same length as the DataFrame
    if len(sizes) != len(df):
        sizes = sizes[:len(df)]  # Trim sizes list to match DataFrame length
    if len(colors) != len(df):
        colors = colors[:len(df)]  # Trim colors list to match DataFrame length

    # Create the scatter plot
    plt.scatter(df[x_col], df[y_col], c=colors, s=sizes)

    # Label the first num_labels observations
    for i in range(num_labels + 1):
        plt.text(df[x_col].iloc[i], df[y_col].iloc[i], df.index[i], fontsize=fontsize)

    # Set labels and title
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if title is None:
        plt.title(f'{x_col} vs {y_col} - {dist_measure}')
    else:
        plt.title(title)

    # Show plot
    plt.show()


def get_feature_indices(df, selected_features):
    if type(selected_features) is list:
        start_idx = df.columns.get_loc(selected_features[0])
        end_idx = df.columns.get_loc(selected_features[-1])
        return start_idx, end_idx
    if type(selected_features) is str:
        idx = df.columns.get_loc(selected_features)
        return idx

# add mio and k to values 
def adjust_money_appearance(x):
    x = int(x)
    if(x >= 1000000):
        x = x / 1000000
        x = "€" + str(x) + " Mio"
        return x
    if(x < 1000000 and x >= 10000):
        x = x / 1000
        x = "€" + str(x) + "k"
        return x
    return "€" + str(x)


# Function to clean the column
def clean_nation(nation):
    pattern = r'\s([A-Z]+)$'
    match = re.search(pattern, str(nation))
    if match:
        return match.group(1)
    else:
        return nation
    
# Import all files all at once
def get_all_attributes(path_to_files):
    filenames = [
        'advanced_goalkeeping.csv',
        'defensive_actions.csv',
        'goalkeeping.csv',
        'goal_and_shot_creation.csv',
        'miscellaneous_stats.csv',
        'passing.csv',
        'playing_time.csv',
        'possession.csv',
        'shooting.csv',
        'standard_stats.csv'
    ]

    dataframes = {}

    for filename in filenames:
        filepath = os.path.join(path_to_files, filename)
        df_name = filename.replace('.csv', '') 
        dataframes[df_name] = pd.read_csv(filepath)

    return dataframes