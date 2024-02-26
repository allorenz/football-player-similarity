import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
import utils


### Import data
data = pd.read_csv('../data/2024/male_players.csv')
data = data[data['fifa_version'] == 24.0]

# interesting columns: league level
data.head()
columns_to_keep = ['player_id', 'short_name', 'player_positions', 
                   'value_eur',	'wage_eur'	,'height_cm' ,'weight_kg',
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

### Clean data

# field player column: fill 0 for NaN
df['goalkeeping_speed'] = df['goalkeeping_speed'].fillna(0)


# goalkeeper columns: fill 0 for NaN
gk_columns_to_fill_na = ['defending', 'pace', 'dribbling', 'physic', 'passing','shooting']
df[gk_columns_to_fill_na] = df[gk_columns_to_fill_na].fillna(0)


# clean body type, one hot encode
df['body_type'] = df['body_type'].str.lower().replace(r'\s*\(.*\)', '', regex=True)
df = pd.get_dummies(df, columns=['body_type'],prefix='body_type', dtype='int') # pd.get_dummies(df['body_type']


# global position - one hot encoding
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


# apply global position encoding
df["Offense"] = df['player_positions'].apply(assign_global_pos_offense)
df["Midfield"] = df['player_positions'].apply(assign_global_pos_midfield)
df["Defense"] = df['player_positions'].apply(assign_global_pos_defense)
df["Goalkeeper"] = df['player_positions'].apply(assign_global_pos_goalkeeper)


# apply one hot encoding on player_positions (e.g: CF, CAM)
df = df.assign(player_positions=df['player_positions'].str.split(', ')).explode('player_positions')
df = pd.get_dummies(df, columns=['player_positions'], prefix='', prefix_sep='',dtype='int')
df = df.groupby('player_id').max().reset_index()
df.shape


# encode preferred foott
df = pd.get_dummies(df, columns= ['preferred_foot'], prefix='Preferred_Foot',dtype='int')#.astype(int)


# clean work rate column
enc = df['work_rate'].str.split('/', expand=True).rename(
                    columns = {0:'Work Rate Offense', 
                               1:'Work Rate Defense'}, 
                    inplace = False)
mapping = {'Low': 1, 'Medium': 2, 'High' : 3}
enc['Work Rate Offense'] = enc['Work Rate Offense'].map(mapping)
enc['Work Rate Defense'] = enc['Work Rate Defense'].map(mapping)
df = pd.concat([df, enc], axis=1)
df = df.drop(["work_rate"], axis=1, inplace=False)


# check for missing values
missing_columns = df.isna().sum()
missing_columns[missing_columns > 0]


# missing price and wage values. they can get scrapped
missing_values_price_and_wage = list(df[df['value_eur'].isna()]['player_id'])
df_to_be_scraped = pd.DataFrame(missing_values_price_and_wage, columns = ['player_id'])
df_to_be_scraped.to_csv("../data/2024/idx_to_scrape_value_and_wage.csv", sep=';', index=False)


# load scrapped data 
scrapped_players = pd.read_json("../data/2024/scrapped_data.json")

# clean urls and keep the idx
pattern = re.compile(r'\d+')
def extract_id(url):
    match = pattern.search(url)
    if match:
        return match.group()
    else:
        return None

# clean data 
scrapped_players['id'] = scrapped_players['url'].apply(extract_id).astype(int)
scrapped_players['value_eur'] = scrapped_players['value_eur'].apply(lambda x: x.replace('€', '') if '€' in str(x) else x)
scrapped_players['value_eur'] = scrapped_players['value_eur'].apply(lambda x: x.replace('.', '') if '.' in str(x) else x).astype(float)
scrapped_players['wage_eur'] = scrapped_players['wage_eur'].apply(lambda x: x.replace('€', '') if '€' in str(x) else x)
scrapped_players['wage_eur'] = scrapped_players['wage_eur'].apply(lambda x: x.replace('.', '') if '.' in str(x) else x).astype(float)


# keep only players with values
scrapped_players = scrapped_players[scrapped_players['value_eur']>0].reset_index(drop=True)
scrapped_players = scrapped_players.drop(columns=['url'], inplace=False)


# merge scrapped data with main data set
merged_data = pd.merge(df, scrapped_players, left_on='player_id', right_on='id', how='left', suffixes=('_t', '_scrapped'))
df.update({'wage_eur': merged_data['wage_eur_scrapped']})
df.update({'value_eur': merged_data['value_eur_scrapped']})


# normalize 
columns_to_normalize = [
    'height_cm' ,'weight_kg','league_level', 'pace', 'shooting', 'passing', 'dribbling',
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
    'goalkeeping_reflexes', 'goalkeeping_speed'
]


# Standardize values
scaler = StandardScaler()
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])


# log transform 
df['wage_eur'] = np.log(df['wage_eur'])
df['value_eur'] = np.log(df['value_eur'])


# drop NaN: players that have no market value or wage 
df = df.dropna()


# store cleaned attributes
df.to_csv("../data/2024/player_attributes.csv", sep=';', index=False)


# prepare and store player information
data = pd.read_csv('../data/2024/male_players.csv')
data = data[data['fifa_version'] == 24.0]
data = data[data['player_id'].isin(df['player_id'])]
player_information = ['player_id', 'short_name','club_name' ,'age', 'overall', 'player_positions','preferred_foot' , 'nationality_name','value_eur','wage_eur','international_reputation' ]
df_player_information = data[player_information]


# add missing market values
df_player_information = pd.merge(df_player_information, scrapped_players[['id', 'value_eur']], left_on='player_id', right_on='id', how='left')
df_player_information['value_eur'] = df_player_information['value_eur_y'].combine_first(df_player_information['value_eur_x'])
df_player_information = df_player_information.drop(['value_eur_x', 'value_eur_y', 'id'], axis=1)
df_player_information = df_player_information[player_information]


# store player information
df_player_information.to_csv("../data/2024/player_information.csv", sep=';', index=False)