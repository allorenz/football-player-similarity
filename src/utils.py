import re
import os
import pandas as pd
import numpy as np


PATH_PLAYER_STATS = '../data/fbref/player_stats.csv'


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


def get_top_k_similar_players(embeddings, query_index, player_info, top_k=10):
    """
    Compute the top-k most similar players based on cosine similarity of embeddings.

    Args:
        embeddings (np.ndarray): Array of player embeddings, shape (num_players, embedding_dim).
        query_index (int): Index of the player to be used as the query point.
        player_info (pd.DataFrame): DataFrame containing player information.
        top_k (int): Number of top similar players to return.

    Returns:
        pd.DataFrame: DataFrame of the top_k similar players with their cosine similarities.
    """
    # Normalize the embeddings (L2 normalization)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Get the query player's normalized embedding
    query_embedding = embeddings_norm[query_index]
    
    # Compute cosine similarities between the query and the rest
    cosine_similarities = np.dot(embeddings_norm, query_embedding)
    
    # Get the top-k most similar samples (including the query player itself)
    top_k_similar_indices = np.argsort(cosine_similarities)[::-1]  # Sort in descending order
    
    # Keep the top_k including the query player
    top_k_similar_indices = top_k_similar_indices[:top_k]
    
    # Extract the player information for the top-k most similar players
    top_k_df = player_info.iloc[top_k_similar_indices].copy()
    
    # Add cosine similarity column to the DataFrame
    top_k_df['cosine_similarity'] = cosine_similarities[top_k_similar_indices]
    
    # Sort the DataFrame by cosine similarity in descending order and drop the 'unique_id' column if it exists
    top_k_df = top_k_df.sort_values('cosine_similarity', ascending=False).drop(columns='unique_id', errors='ignore')
    
    return top_k_df


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