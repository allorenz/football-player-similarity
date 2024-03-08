import numpy as np
import pandas as pd
# Assuming you have a numpy array for each player with 12 features
# query_player_features and most_similar_player_features are assumed to be 1D numpy arrays

# Calculate cosine similarity
def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

# Find the most similar player
def find_most_similar(query_player_features, all_players_features):
    similarities = [cosine_similarity(query_player_features, player_features) for player_features in all_players_features]
    most_similar_index = np.argmax(similarities)
    return most_similar_index

# Example usage
# Replace query_player_features and all_players_features with your actual data
df = pd.read_csv("../data/2024/player_attributes.csv", sep=';')

# prepare feature labels
X = df.drop(columns=["short_name", "player_id"])
label_numbers = [i for i in range(len(X.columns))]
labels = [label for label in X.columns ]
label_dict = dict(zip(label_numbers, labels))

# prepare embeddings
df = df[df["player_id"]!= 241852]
all_players_features = df.drop(columns=["short_name", "player_id"]).values.tolist()
df = pd.read_csv("../data/2024/player_attributes.csv", sep=';')
df = df.drop(columns=["short_name"])
query_player_features = df[df["player_id"] == 241852].drop(columns=['player_id']).values.tolist()
query_player_features = np.array(query_player_features)

# compute cosine similarity
most_similar_index = find_most_similar(query_player_features, all_players_features)


# meaningful columns
df = pd.read_csv("../data/2024/player_attributes.csv", sep=';') 
omd_features = ['pace', 'shooting', 'passing', 'dribbling',
       'defending', 'physic', 'attacking_crossing', 'attacking_finishing',
       'attacking_heading_accuracy', 'attacking_short_passing',
       'attacking_volleys', 'skill_dribbling', 'skill_curve',
       'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
       'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
       'movement_reactions', 'movement_balance', 'power_shot_power',
       'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
       'mentality_aggression', 'mentality_interceptions',
       'mentality_positioning', 'mentality_vision', 'mentality_penalties',
       'defending_marking_awareness', 'defending_standing_tackle',
       'defending_sliding_tackle']

df_omd = df[omd_features]
most_similar_player_features = np.array(df_omd.iloc[[most_similar_index]])

df = pd.read_csv("../data/2024/player_attributes.csv", sep=';')
df = df[df["player_id"] == 241852]
df_omd = df[omd_features]
query_player_features = np.array(df_omd)

# check dimensions
query_player_features.shape, most_similar_player_features.shape

# Find the 4 most similar features
abs_dif = np.abs(query_player_features - most_similar_player_features )
sorted_indices = np.argsort(abs_dif)
sorted_indices = sorted_indices[0][:35]

# Display the results
print("Most similar player features:", most_similar_player_features)
print("Query player features:", query_player_features)

most_similar_player_features.shape, sorted_indices.shape

for i in sorted_indices:
    print(omd_features[i])



# conclusion: even the difference between offense, midfielder, and defensce
# dont show the similarities you might inspect.
# e.g query player diaby is known for speed but most similar players doesnt show
# the same attributes (having minor difference)
# try: different scaling, show radar plots based on set attributes
# let gpt check the argsort method