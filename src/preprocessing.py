import pandas as pd

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

# encode player positions
def assign_category(pos):
    offense = ['RS','LS','RW','LW','ST','CF','LF','RF']
    midfield = ['CAM','LDM','RCM','RDM','LCM','RM','LM','CDM','CM','RAM','LAM']
    defense = ['RCB','CB','LWB','LCB','LB','RB','RWB']
    goalkeeper = ['GK']

    pos = str(pos)
    if pos in offense:
        return 'offense'
    elif pos in midfield:
        return 'midfield'
    elif pos in defense:
        return 'defense'
    elif pos in goalkeeper:
        return 'goalkeeper'
    else:
        return 'unknown'

temp = df.player_positions.str.split(',', expand=True)
temp[0] = temp[0].apply(assign_category)
temp[1] = temp[1].str.strip().apply(assign_category)
temp[2] = temp[2].str.strip().apply(assign_category)
temp[3] = temp[3].str.strip().apply(assign_category)
enc_global_pos = pd.get_dummies(temp[[0]], dtype='int',prefix='global_position')
all(enc_global_pos['global_position_goalkeeper'] == 0)

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