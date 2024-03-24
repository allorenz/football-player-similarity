import pandas as pd
import utils

# loading the data
data = pd.read_csv('../data/fbref/standard_stats.csv')
data = data[data["Season"] == "2022-2023"]

data.columns
columns_to_drop = ['S_e', 'L_e',
       'T_e', 'U_n', 'U_n.1', 'U_n.2', 'U_n.3', 'P_l', 'P_l.1', 'P_l.2',
       'P_l.3', 'P_e', 'P_e.1', 'P_e.2', 'P_e.3', 'P_e.4', 'P_e.5', 'P_e.6',
       'P_e.7', 'E_x', 'E_x.1', 'E_x.2', 'E_x.3', 'P_r', 'P_r.1', 'P_r.2',
       'P_e.8', 'P_e.9', 'P_e.10', 'P_e.11', 'P_e.12', 'P_e.13', 'P_e.14',
       'P_e.15', 'P_e.16', 'P_e.17', 'U_n.4', 'U_n.5','Unnamed: 33_level_0_Matches',
       'Unnamed: 24_level_0_Matches']

data = data.drop(columns=columns_to_drop, inplace=False)


data = data.rename(columns={'Unnamed: 0_level_0_Player': 'Player',
                            'Unnamed: 1_level_0_Nation': 'Nation', 
                            'Unnamed: 2_level_0_Pos': 'Pos',
                            'Unnamed: 3_level_0_Age' : 'Age', 
                            'Unnamed: 4_level_0_MP': 'MP'})



# Clean the 'Nation' column
data["Nation"] = [utils.clean_nation(nation) for nation in data["Nation"]]

# Clean matches played
data['Matches Played'] = data['MP'].astype(str) + data['Playing Time_MP'].astype(str)
data['Matches Played'] = data['Matches Played'] .str.replace('nan', '')

data = data.drop(columns=["MP",'Playing Time_MP'], inplace=False)











