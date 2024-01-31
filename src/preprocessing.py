import pandas as pd
import spacy

# Import data
data = pd.read_csv('../data/2024/male_players.csv')


# Clean data
# interesting columns: league level
# columns_to_drop = [""]
data.head()
columns_to_keep = ['player_id', 'long_name', 'player_positions', 'overall', 
                   'value_eur',	'wage_eur'	,'age'	,'height_cm' ,'weight_kg',
                   'league_level',
                   'preferred_foot', 'weak_foot', 'skill_moves', 'international_reputation',
                    'work_rate', 'body_type', 'pace', 'shooting', 'passing', 'dribbling',
                    'defending', 'physic', 'attacking_crossing', 'attacking_finishing',
                    'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys',
                    'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing',
                    'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed',
                    'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power',
                    'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
                    'mentality_aggression', 'mentality_interceptions', 'mentality_positioning',
                    'mentality_vision', 'mentality_penalties', 'defending_marking_awareness',
                    'defending_standing_tackle', 'defending_sliding_tackle', 'goalkeeping_diving',
                    'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning',
                    'goalkeeping_reflexes', 'goalkeeping_speed']
df = data[columns_to_keep]

# one hot encoding
def assign_global_pos_offense(text):
    offense = ['RS','LS','RW','LW','ST','CF','LF','RF']
    pos_lst = text.split(",")
    for pos in pos_lst:
        if pos.strip() in offense:
            return 1
    return 0

def assign_global_pos_midfield(text):
    midfield = ['CAM','LDM','RCM','RDM','LCM','RM','LM','CDM','CM','RAM','LAM']
    pos_lst = text.split(",")
    for pos in pos_lst:
        if pos.strip() in midfield:
            return 1
    return 0

def assign_global_pos_defense(text):
    defense = ['RCB','CB','LWB','LCB','LB','RB','RWB']
    pos_lst = text.split(",")
    for pos in pos_lst:
        if pos.strip() in defense:
            return 1
    return 0

def assign_global_pos_goalkeeper(text):
    goalkeeper = ['GK']
    pos_lst = text.split(",")
    for pos in pos_lst:
        if pos.strip() in goalkeeper:
            return 1
    return 0

# apply encoding TODO: player_positions
df["Offense"] = df['player_positions'].apply(assign_global_pos_offense)
df["Midfield"] = df['player_positions'].apply(assign_global_pos_midfield)
df["Defense"] = df['player_positions'].apply(assign_global_pos_defense)
df["Goalkeeper"] = df['player_positions'].apply(assign_global_pos_goalkeeper)


# encode preferred foot TODO: drop preferred_foot
enc = pd.get_dummies(df['preferred_foot'], prefix='Preferred_Foot',dtype='int')#.astype(int)
df = pd.concat([df, enc] ,axis=1)

# clean work rate column: TODO: drop work_rate
enc = df['work_rate'].str.split('/', expand=True).rename(
                    columns = {0:'Work Rate Offense', 
                               1:'Work Rate Defense'}, 
                    inplace = False)
mapping = {'Low': 1, 'Medium': 2, 'High' : 3}
enc['Work Rate Offense'] = enc['Work Rate Offense'].map(mapping)
enc['Work Rate Defense'] = enc['Work Rate Defense'].map(mapping)
df = pd.concat([df, enc], axis=1)

df = df.drop(["player_positions","preferred_foot","work_rate"],axis=1, inplace=False)