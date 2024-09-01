import pandas as pd
import utils



def load_standard_stats():
       # goalkeeper and field players
       data = pd.read_csv('../data/fbref/standard_stats.csv')
       data = data[data["Season"] == "2022-2023"]
       
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


       data = data[data["Player"] != "Squad Total"]
       data = data[data["Player"] != "Opponent Total"]

       # Clean the 'Nation' column
       data["Nation"] = [utils.clean_nation(nation) for nation in data["Nation"]]

       # Clean matches played
       data['Matches Played'] = data['MP'].astype(str) + data['Playing Time_MP'].astype(str)
       data['Matches Played'] = data['Matches Played'] .str.replace('nan', '')
       data = data.drop(columns=["MP",'Playing Time_MP'], inplace=False)

       data.columns = data.columns.str.replace('^Performance_', '', regex=True).str.replace('^Expected_', '', regex=True).str.replace('^Progression_', '', regex=True)

       data.to_csv('../data/fbref/cleaned/standard_stats.csv', index=False)

       return data   


def load_shooting():
       # goalkeeper and field players
       data = pd.read_csv("../data/fbref/shooting.csv")
       data = data[data["Season"] == "2022-2023"]
       

       columns_to_drop = ['S_e', 'L_e', 'T_e',
              'U_n', 'U_n.1', 'U_n.2', 'U_n.3', 'U_n.4', 'S_t', 'S_t.1', 'S_t.2',
              'S_t.3', 'S_t.4', 'S_t.5', 'S_t.6', 'S_t.7', 'S_t.8', 'S_t.9', 'S_t.10',
              'S_t.11', 'E_x', 'E_x.1', 'E_x.2', 'E_x.3', 'E_x.4', 'U_n.5','Unnamed: 16_level_0_Matches',
       'Unnamed: 22_level_0_Matches']
       
       data = data.drop(columns=columns_to_drop, inplace=False)

       data = data.rename(columns={'Unnamed: 0_level_0_Player': 'Player',
                                   'Unnamed: 1_level_0_Nation': 'Nation', 
                                   'Unnamed: 2_level_0_Pos': 'Pos',
                                   'Unnamed: 3_level_0_Age' : 'Age', 
                                   'Unnamed: 4_level_0_MP': 'MP',
                                   'Unnamed: 4_level_0_90s': '90s'})
       
       data = data[data["Player"] != "Squad Total"]
       data = data[data["Player"] != "Opponent Total"]

       # Clean the 'Nation' column
       data["Nation"] = [utils.clean_nation(nation) for nation in data["Nation"]]
       data.columns = data.columns.str.replace('^Standard_', '', regex=True).str.replace('^Expected_', '', regex=True)
       data.to_csv('../data/fbref/cleaned/shooting.csv', index=False)

       return data


def load_goalkeeping():
       data = pd.read_csv("../data/fbref/goalkeeping.csv")
       data = data[data["Season"] == "2022-2023"]
       

       columns_to_drop = ['S_e', 'L_e',
       'T_e', 'U_n', 'U_n.1', 'U_n.2', 'U_n.3', 'P_l', 'P_l.1', 'P_l.2',
       'P_l.3', 'P_e', 'P_e.1', 'P_e.2', 'P_e.3', 'P_e.4', 'P_e.5', 'P_e.6',
       'P_e.7', 'P_e.8', 'P_e.9', 'P_e.10', 'P_e.11', 'P_e.12', 'P_e.13',
       'P_e.14', 'U_n.4','Unnamed: 23_level_0_Matches']
       data = data.drop(columns=columns_to_drop, inplace=False)

       data = data.rename(columns={'Unnamed: 0_level_0_Player': 'Player',
                                          'Unnamed: 1_level_0_Nation': 'Nation', 
                                          'Unnamed: 2_level_0_Pos': 'Pos',
                                          'Unnamed: 3_level_0_Age' : 'Age', 
                                          'Playing Time_MP': 'MP'})

       data = data[data["Player"] != "Squad Total"]
       data = data[data["Player"] != "Opponent Total"]
       data.columns = data.columns.str.replace('^Performance_', '', regex=True)

       data["Nation"] = [utils.clean_nation(nation) for nation in data["Nation"]]
       data.to_csv('../data/fbref/cleaned/goalkeeping.csv', index=False)

       return data


def load_advanced_goalkeeping():
       # goalkeeper only
       data = pd.read_csv("../data/fbref/advanced_goalkeeping.csv")
       data = data[data["Season"] == "2022-2023"]
       

       columns_to_drop = ['S_e', 'L_e', 'T_e', 'U_n', 'U_n.1',
              'U_n.2', 'U_n.3', 'U_n.4', 'G_o', 'G_o.1', 'G_o.2', 'G_o.3', 'G_o.4',
              'E_x', 'E_x.1', 'E_x.2', 'E_x.3', 'L_a', 'L_a.1', 'L_a.2', 'P_a',
              'P_a.1', 'P_a.2', 'P_a.3', 'G_o.5', 'G_o.6', 'G_o.7', 'C_r', 'C_r.1',
              'C_r.2', 'S_w', 'S_w.1', 'S_w.2', 'U_n.5','Unnamed: 30_level_0_Matches']
       data = data.drop(columns=columns_to_drop, inplace=False)

       data = data.rename(columns={'Unnamed: 0_level_0_Player': 'Player',
                                          'Unnamed: 1_level_0_Nation': 'Nation', 
                                          'Unnamed: 2_level_0_Pos': 'Pos',
                                          'Unnamed: 3_level_0_Age' : 'Age', 
                                          'Unnamed: 4_level_0_MP': 'MP',
                                          'Unnamed: 4_level_0_90s': '90s'})

       data = data[data["Player"] != "Squad Total"]
       data = data[data["Player"] != "Opponent Total"]
       data["Nation"] = [utils.clean_nation(nation) for nation in data["Nation"]]
       data.columns = data.columns.str.replace('^Expected_', '', regex=True).str.replace('^Crosses_', '', regex=True).str.replace('^Sweeper_', '', regex=True)

       data.to_csv('../data/fbref/cleaned/advanced_goalkeeping.csv', index=False)

       return data


def load_defensive_actions():
       # goalkeeper and field players
       data = pd.read_csv("../data/fbref/defensive_actions.csv")
       data = data[data["Season"] == "2022-2023"]
       

       columns_to_drop = ['S_e', 'L_e', 'T_e', 'U_n', 'U_n.1',
              'U_n.2', 'U_n.3', 'U_n.4', 'T_a', 'T_a.1', 'T_a.2', 'T_a.3', 'T_a.4',
              'C_h', 'C_h.1', 'C_h.2', 'C_h.3', 'B_l', 'B_l.1', 'B_l.2', 'U_n.5',
              'U_n.6', 'U_n.7', 'U_n.8', 'U_n.9','Unnamed: 21_level_0_Matches']
       data = data.drop(columns=columns_to_drop, inplace=False)

       data = data.rename(columns={'Unnamed: 0_level_0_Player': 'Player',
                                   'Unnamed: 1_level_0_Nation': 'Nation', 
                                   'Unnamed: 2_level_0_Pos': 'Pos',
                                   'Unnamed: 3_level_0_Age' : 'Age', 
                                   'Unnamed: 4_level_0_90s': '90s',
                                   'Unnamed: 17_level_0_Int': 'Interceptions',
                                   'Unnamed: 18_level_0_Tkl+Int': 'Tkl+Interceptions',
                                   'Unnamed: 19_level_0_Clr': 'Clearances',
                                   'Unnamed: 20_level_0_Err' : 'Errors'})

       data = data[data["Player"] != "Squad Total"]
       data = data[data["Player"] != "Opponent Total"]
       data["Nation"] = [utils.clean_nation(nation) for nation in data["Nation"]]
       data.to_csv('../data/fbref/cleaned/defensive_actions.csv', index=False)

       return data


def load_goal_shot_creation():
       data = pd.read_csv("../data/fbref/goal_and_shot_creation.csv")
       data = data[data["Season"] == "2022-2023"]
       

       columns_to_drop = ['S_e', 'L_e', 'T_e', 'U_n', 'U_n.1',
              'U_n.2', 'U_n.3', 'U_n.4', 'S_C', 'S_C.1', 'S_C.2', 'S_C.3', 'S_C.4',
              'S_C.5', 'S_C.6', 'S_C.7', 'G_C', 'G_C.1', 'G_C.2', 'G_C.3', 'G_C.4',
              'G_C.5', 'G_C.6', 'G_C.7', 'U_n.5', 'Unnamed: 21_level_0_Matches']
       data = data.drop(columns=columns_to_drop, inplace=False)

       data = data.rename(columns={'Unnamed: 0_level_0_Player': 'Player',
                                   'Unnamed: 1_level_0_Nation': 'Nation', 
                                   'Unnamed: 2_level_0_Pos': 'Pos',
                                   'Unnamed: 3_level_0_Age' : 'Age', 
                                   'Unnamed: 4_level_0_90s': '90s'})

       data = data[data["Player"] != "Squad Total"]
       data = data[data["Player"] != "Opponent Total"]
       data["Nation"] = [utils.clean_nation(nation) for nation in data["Nation"]]
       data.to_csv('../data/fbref/cleaned/goal_and_shot_creation.csv', index=False)

       return data


def load_miscellaneous_stats():
       data = pd.read_csv("../data/fbref/miscellaneous_stats.csv")
       data = data[data["Season"] == "2022-2023"]
       


       columns_to_drop = ['S_e', 'L_e', 'T_e',
              'U_n', 'U_n.1', 'U_n.2', 'U_n.3', 'U_n.4', 'P_e', 'P_e.1', 'P_e.2',
              'P_e.3', 'P_e.4', 'P_e.5', 'P_e.6', 'P_e.7', 'P_e.8', 'P_e.9', 'P_e.10',
              'P_e.11', 'P_e.12', 'A_e', 'A_e.1', 'A_e.2', 'U_n.5','Unnamed: 21_level_0_Matches',
              'Unnamed: 17_level_0_Matches']
       data = data.drop(columns=columns_to_drop, inplace=False)

       data = data.rename(columns={'Unnamed: 0_level_0_Player': 'Player',
                                   'Unnamed: 1_level_0_Nation': 'Nation', 
                                   'Unnamed: 2_level_0_Pos': 'Pos',
                                   'Unnamed: 3_level_0_Age' : 'Age', 
                                   'Unnamed: 4_level_0_90s': '90s'})

       data = data[data["Player"] != "Squad Total"]
       data = data[data["Player"] != "Opponent Total"]
       data["Nation"] = [utils.clean_nation(nation) for nation in data["Nation"]]
       data.columns = data.columns.str.replace('^Performance_', '', regex=True)

       data.to_csv('../data/fbref/cleaned/miscellaneous_stats.csv', index=False)

       return data


def load_passing():
       # goalkeeper and field players
       data = pd.read_csv("../data/fbref/passing.csv")
       data = data[data["Season"] == "2022-2023"]
       

       columns_to_drop = ['S_e', 'L_e',
              'T_e', 'U_n', 'U_n.1', 'U_n.2', 'U_n.3', 'U_n.4', 'T_o', 'T_o.1',
              'T_o.2', 'T_o.3', 'T_o.4', 'S_h', 'S_h.1', 'S_h.2', 'M_e', 'M_e.1',
              'M_e.2', 'L_o', 'L_o.1', 'L_o.2', 'U_n.5', 'U_n.6', 'E_x', 'E_x.1',
              'U_n.7', 'U_n.8', 'U_n.9', 'U_n.10', 'U_n.11', 'U_n.12','Unnamed: 21_level_0_xA',
              'Unnamed: 22_level_0_A-xAG','Unnamed: 21_level_0_xA', 'Unnamed: 22_level_0_A-xAG',
              'Unnamed: 28_level_0_Matches']
       data = data.drop(columns=columns_to_drop, inplace=False)

       data = data.rename(columns={'Unnamed: 0_level_0_Player': 'Player',
                                   'Unnamed: 1_level_0_Nation': 'Nation', 
                                   'Unnamed: 2_level_0_Pos': 'Pos',
                                   'Unnamed: 3_level_0_Age' : 'Age', 
                                   'Unnamed: 4_level_0_90s': '90s',
                                   'Unnamed: 19_level_0_Ast': 'Assists',
                                   'Unnamed: 20_level_0_xAG': 'xAG',
                                   'Unnamed: 23_level_0_KP': 'Key Passes',
                                   'Unnamed: 24_level_0_1/3': 'Passes_to_1/3',
                                   'Unnamed: 25_level_0_PPA' : 'Passes_to_Penalt_Area',
                                   'Unnamed: 26_level_0_CrsPA':'Crosses_into_Penalty_Area',
                                   'Unnamed: 27_level_0_PrgP': 'Progressive Passes'})

       data = data[data["Player"] != "Squad Total"]
       data = data[data["Player"] != "Opponent Total"]
       data["Nation"] = [utils.clean_nation(nation) for nation in data["Nation"]]
       data.to_csv('../data/fbref/cleaned/passing.csv', index=False)

       return data


def load_playing_time():
       data = pd.read_csv("../data/fbref/playing_time.csv")
       data = data[data["Season"] == "2022-2023"]
       

       columns_to_drop = ['S_e', 'L_e', 'T_e', 'U_n', 'U_n.1',
       'U_n.2', 'U_n.3', 'P_l', 'P_l.1', 'P_l.2', 'P_l.3', 'P_l.4', 'S_t',
       'S_t.1', 'S_t.2', 'S_u', 'S_u.1', 'S_u.2', 'T_e.1', 'T_e.2', 'T_e.3',
       'T_e.4', 'T_e.5', 'T_e.6', 'T_e.7', 'T_e.8', 'T_e.9', 'T_e.10',
       'T_e.11', 'U_n.4', 'U_n.5', 'Unnamed: 4_level_0_MP', 'Unnamed: 26_level_0_Matches',
       'Unnamed: 21_level_0_Matches','Playing Time_MP']
       data = data.drop(columns=columns_to_drop, inplace=False)

       data = data.rename(columns={'Unnamed: 0_level_0_Player': 'Player',
                                          'Unnamed: 1_level_0_Nation': 'Nation', 
                                          'Unnamed: 2_level_0_Pos': 'Pos',
                                          'Unnamed: 3_level_0_Age' : 'Age', 
                                          'Unnamed: 4_level_0_90s': '90s',
                                          'Unnamed: 19_level_0_Ast': 'Assists',
                                          'Unnamed: 20_level_0_xAG': 'xAG',
                                          'Unnamed: 23_level_0_KP': 'Key Passes',
                                          'Unnamed: 24_level_0_1/3': 'Passes_to_1/3',
                                          'Unnamed: 25_level_0_PPA' : 'Passes_to_Penalt_Area',
                                          'Unnamed: 26_level_0_CrsPA':'Crosses_into_Penalty_Area',
                                          'Unnamed: 27_level_0_PrgP': 'Progressive Passes'})

       data = data[data["Player"] != "Squad Total"]
       data = data[data["Player"] != "Opponent Total"]
       data["Nation"] = [utils.clean_nation(nation) for nation in data["Nation"]]
       data.columns = data.columns.str.replace('^Starts_', '', regex=True).str.replace('^Subs_', '', regex=True).str.replace('^Team Success_', '', regex=True).str.replace('^Team Success ', '', regex=True)

       data.to_csv('../data/fbref/cleaned/playing_time.csv', index=False)
       
       return data


def load_possession():
       data = pd.read_csv("../data/fbref/possession.csv")
       data = data[data["Season"] == "2022-2023"]
       
       columns_to_drop = ['S_e',
              'L_e', 'T_e', 'U_n', 'U_n.1', 'U_n.2', 'U_n.3', 'U_n.4', 'T_o', 'T_o.1',
              'T_o.2', 'T_o.3', 'T_o.4', 'T_o.5', 'T_o.6', 'T_a', 'T_a.1', 'T_a.2',
              'T_a.3', 'T_a.4', 'C_a', 'C_a.1', 'C_a.2', 'C_a.3', 'C_a.4', 'C_a.5',
              'C_a.6', 'C_a.7', 'R_e', 'R_e.1', 'U_n.5', 'Unnamed: 27_level_0_Matches']
       data = data.drop(columns=columns_to_drop, inplace=False)

       data = data.rename(columns={'Unnamed: 0_level_0_Player': 'Player',
                                   'Unnamed: 1_level_0_Nation': 'Nation', 
                                   'Unnamed: 2_level_0_Pos': 'Pos',
                                   'Unnamed: 3_level_0_Age' : 'Age', 
                                   'Unnamed: 4_level_0_90s': '90s'})

       data = data[data["Player"] != "Squad Total"]
       data = data[data["Player"] != "Opponent Total"]
       data["Nation"] = [utils.clean_nation(nation) for nation in data["Nation"]]
       data.to_csv('../data/fbref/cleaned/possession.csv', index=False)
       
       return data

load_advanced_goalkeeping()
load_defensive_actions()
load_goal_shot_creation()
load_goalkeeping()
load_miscellaneous_stats()
load_passing()
load_playing_time()
load_possession()
load_shooting()
load_standard_stats()