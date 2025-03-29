import sys
import os
import numpy as np
import math
import pandas as pd
import json
from ast import literal_eval
from pathlib import Path
from feature_extraction.base_extractor import BaseDimensionFeatureExtractor

PI = math.pi
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent

def convert_to_list(input_data):
    if(isinstance(input_data, str)):
        try:
            return literal_eval(input_data)
        except (ValueError, SyntaxError):
            print(f"Error: The string {input_data} could not be converted to a list.")
            return None
    
    return input_data

def is_in_penalty_area(x, y):
    pitch_width = 120

    # Check if in larger penalty area but NOT in goal area
    x_axis = ((102 <= x) and (x <= pitch_width))  # Stop before goal area
    y_axis = ((40 - 22) <= y and y <= (40 + 22))
    return x_axis and y_axis

def is_in_goal_area(x, y):
    pitch_width = 120

    x_axis = ((pitch_width - 6 <= x) and (x <= pitch_width))
    y_axis = ((40 - 10) <= y and y <= (40 + 10))
    return x_axis and y_axis

def is_in_edge_of_the_box(x, y):
    pitch_width = 120

    x_axis = ((102 <= x) and (x <= pitch_width))
    y_axis = ((40 - 10) <= y and y <= (40 + 10))
    return x_axis and y_axis

def completed_passes(df):
    return (df["pass_outcome"] != "Incomplete") & \
           (df["pass_outcome"] != "Injury Clearance") & \
           (df["pass_outcome"] != "Out") & \
           (df["pass_outcome"] != "Pass Offside") & \
           (df["pass_outcome"] != "Unknown") 

def is_cut_back_into_edge_of_the_box(df):
    return ((df["pass_cut_back"] == True) & (df["pass_end_location"].apply(is_in_edge_of_the_box)))

def is_chip_ball(df):
    return ((df["pass_height"] == "High Pass"))

def is_vertical_pass(angle, max_tolerance=50):
    """
    Checks if the ball's movement is considered to be deep or vertical moving moving along side the x-axis.
    
    Parameters:
    - angle (float): The angle in radians (from arctan of y/x).
    - max_tolerance (float): Maximum angle in degrees .

    Returns:
    - bool: True if the angle is within the specified range, False otherwise.
    """
    normalized_angle = (angle * (180/PI))
    return  max_tolerance >= abs(normalized_angle)

def is_horizontal_pass(angle, min_tolerance=75, max_tolerance=105):
    """
    Checks if the ball's movement is primarily horizontal based on the angle between the pass direction and the x-axis.
    
    Parameters:
    - angle (float): The angle in radians (from arctan of y/x).
    - min_tolerance (float): Minimum angle in degrees (default is 75°).
    - max_tolerance (float): Maximum angle in degrees (default is 105°).

    Returns:
    - bool: True if the angle is within the specified range, False otherwise.
    """
    normalized_angle = (angle * (180/PI))
    return  max_tolerance >= abs(normalized_angle) >= min_tolerance

def is_progressive_pass(angle, tolerance=75):
    """
    Checks if the ball's movement angle (in radians) is a progressive pass and within the allowed range of angle towards the goal.
    
    Parameters:
    - angle (float): The angle in radians (from arctan of y/x).
    - tolerance (float): Maximum angle in degrees (default is 70°).
    
    Returns:
    - bool: True if the angle is within the tolerance, False otherwise.
    """
    normalized_angle = angle * (180 / PI)
    return abs(normalized_angle) < tolerance

def is_backward_pass(angle, tolerance=105):
    """
    Checks if the ball's movement angle (in radians) is a defensive pass and within the allowed range of angle towards the own goal.
    
    Parameters:
    - angle (float): The angle in radians (from arctan of y/x).
    - tolerance (float): Maximum angle in degrees (default is 70°).
    
    Returns:
    - bool: True if the angle is within the tolerance, False otherwise.
    """
    normalized_angle = angle * (180 / PI)
    return abs(normalized_angle) > tolerance

def is_ex_inside_goal_area(location):
    #print("Executing is_in_goal_area method")
    pitch_width = 120
    if isinstance(location, str):
            location = convert_to_list(location)

    x = location[0]
    y = location[1]

    x_axis = ((pitch_width - 6 <= x) and (x <= pitch_width))
    y_axis = ((40 - 10) <= y and y <= (40 + 10))
    return x_axis and y_axis

def is_ex_inside_penalty_area(location):
    #print("Executing is_in_penalty_area method")
    pitch_width = 120
    if isinstance(location, str):
        location = convert_to_list(location)
    x = location[0]
    y = location[1]

    if is_ex_inside_goal_area(location):
        return False
    else:
        # Check if in larger penalty area but NOT in goal area
        x_axis = ((102 <= x) and (x <= pitch_width))  # Stop before goal area
        y_axis = ((40 - 22) <= y and y <= (40 + 22))
        return x_axis and y_axis

def is_ex_inside_attacking_third(location):
    #print("Executing is_in_attacking_third method")
    pitch_width = 120
    if isinstance(location, str):
        location = convert_to_list(location)
    x = location[0]
    y = location[1]

    if is_ex_inside_goal_area(location) or is_ex_inside_penalty_area(location):
        return False
    else:
        attacking_third_start = 2 * (pitch_width / 3)
        return (attacking_third_start <= x <= pitch_width) 

def calculate_degree(radiant):
    return (radiant * (180/3.14))

def is_pass_completed(pass_outcome):
    return ( (pass_outcome != "Incomplete") & (pass_outcome != "Injury Clearance") & (pass_outcome != "Out") & (pass_outcome != "Pass Offside") &  (pass_outcome != "Unknown") )


class PassingFeatureExtractor(BaseDimensionFeatureExtractor):

    def __init__(self, df: pd.DataFrame, standard_stats: pd.DataFrame, league:str):
        self.df = df
        self.standard_stats = standard_stats
        self.league = league
        self.dim = "passing"

        with open(f"{PROJECT_ROOT_DIR}/config/columns_config.json", 'r') as f:
            self.columns = json.load(f)

    def filter_by_dimension(self):
        self.df = self.df.loc[self.df["type"].isin(self.columns[self.dim]["row_filter"]), self.columns[self.dim]["columns"]]

    def convert_columns(self):
        self.df["location"] = self.df["location"].apply(convert_to_list)
        self.df["pass_end_location"] = self.df["pass_end_location"].apply(convert_to_list)
        self.df[["x", "y"]] = self.df["location"].apply(pd.Series)
        self.df[["x_end_pass", "y_end_pass"]] = self.df["pass_end_location"].apply(pd.Series)
        self.df["under_pressure"] = self.df["under_pressure"].fillna(False)

    def extract(self):
        """
        This function Pre-compute all conditions for Vectorize operations.
        Returns results grouped by player and under_pressure  
        """
        # Pre-compute all conditions
        df_with_flags = self.df.copy()
        
        # Location-based flags
        df_with_flags['is_attacking_third'] = self.df['x'] >= 80
        df_with_flags['is_middle_third'] = (80 > self.df['x']) & (self.df['x'] > 40)
        df_with_flags['is_defending_third'] = self.df['x'] <= 40
        df_with_flags['is_in_box'] = self.df[["x","y"]].apply(lambda row: is_in_penalty_area(row['x'], row['y']), axis=1)
        df_with_flags['is_in_edge_box'] = self.df[["x","y"]].apply(lambda row: is_in_edge_of_the_box(row['x'], row['y']), axis=1)
        
        # End location flags
        df_with_flags['end_attacking_third'] = self.df['x_end_pass'] >= 80 # self.df[["x_end_pass", "y_end_pass"]]
        df_with_flags['end_in_box'] = self.df[["x_end_pass","y_end_pass"]].apply(lambda row: is_in_penalty_area(row['x_end_pass'], row['y_end_pass']), axis=1)
        df_with_flags['end_in_goal_area'] = self.df[["x_end_pass","y_end_pass"]].apply(lambda row: is_in_goal_area(row['x_end_pass'], row['y_end_pass']), axis=1)
        df_with_flags['end_in_edge_box'] = self.df[["x_end_pass","y_end_pass"]].apply(lambda row: is_in_edge_of_the_box(row['x_end_pass'], row['y_end_pass']), axis=1)
        
        # Pass type flags
        df_with_flags['is_completed'] = self.df['pass_outcome'].isna()
        df_with_flags['is_vertical'] = self.df['pass_angle'].apply(is_vertical_pass)
        df_with_flags['is_horizontal'] = self.df['pass_angle'].apply(is_horizontal_pass)
        df_with_flags['is_backward'] = self.df['pass_angle'].apply(is_backward_pass)
        df_with_flags['is_progressive'] = self.df['pass_angle'].apply(is_progressive_pass)
        df_with_flags['is_pass_shot_assist'] = self.df['pass_shot_assist'] == True
        df_with_flags['is_pass_goal_assist'] = self.df['pass_goal_assist'] == True
        df_with_flags['is_switch'] = self.df['pass_switch'] == True
        df_with_flags['is_cross'] = self.df['pass_cross'] == True
        df_with_flags['is_cut_back'] = self.df['pass_cut_back'] == True
        df_with_flags['is_through_ball'] = self.df['pass_technique'] == "Through Ball"

        # Pass length categories
        df_with_flags['is_short'] = (self.df['pass_length'] >= 5) & (self.df['pass_length'] <= 15)
        df_with_flags['is_medium'] = (self.df['pass_length'] >= 15) & (self.df['pass_length'] <= 30)
        df_with_flags['is_long'] = self.df['pass_length'] >= 30
        df_with_flags['is_short_and_completed'] = (df_with_flags['is_short']) & (df_with_flags['is_completed'])
        df_with_flags['is_medium_and_completed'] = (df_with_flags['is_medium']) & (df_with_flags['is_completed'])
        df_with_flags['is_long_and_completed'] = (df_with_flags['is_long']) & (df_with_flags['is_completed'])
        df_with_flags['is_progressive_and_completed'] = (df_with_flags['is_progressive']) & (df_with_flags['is_completed'])
        df_with_flags['is_horizontal_and_completed'] = (df_with_flags['is_horizontal']) & (df_with_flags['is_completed'])
        df_with_flags['is_backward_and_completed'] = (df_with_flags['is_backward']) & (df_with_flags['is_completed'])
        df_with_flags['is_long_and_vertical'] = (df_with_flags['is_long']) & (df_with_flags['is_vertical'])
        df_with_flags['is_long_vertical_and_from_defending_third'] = (df_with_flags['is_long']) & (df_with_flags['is_vertical']) 
        df_with_flags['is_long_vertical_and_from_mid_third'] = (df_with_flags['is_long']) & (df_with_flags['is_vertical']) & (df_with_flags['is_middle_third'])
        df_with_flags['is_long_vertical_and_from_mid_third_into_the_box'] = (df_with_flags['is_long']) & (df_with_flags['is_vertical']) & (df_with_flags['is_middle_third']) & (df_with_flags['end_in_box'])
        
        # Combined conditions
        df_with_flags['into_box_not_from_box'] = df_with_flags['end_in_box'] & ~df_with_flags['is_in_box']
        df_with_flags['into_box_not_from_box_completed'] = df_with_flags['into_box_not_from_box'] & df_with_flags['is_completed']
        df_with_flags['into_edge_box_not_from_edge_box'] = df_with_flags['end_in_edge_box'] & ~df_with_flags['is_in_edge_box']
        df_with_flags['into_edge_box_not_from_edge_box_completed'] = df_with_flags['into_edge_box_not_from_edge_box'] & df_with_flags['is_completed']
        df_with_flags['is_through_ball_and_completed'] = (df_with_flags['is_through_ball']) & (df_with_flags['is_completed'])
        
        df_with_flags['within_attacking_third'] = df_with_flags['is_attacking_third'] & df_with_flags['end_attacking_third']
        df_with_flags['within_attacking_third_completed'] = df_with_flags['within_attacking_third'] & df_with_flags['is_completed']
        
        # Pass type combinations
        df_with_flags['vertical_into_edge'] = (df_with_flags['is_vertical'] & 
                                            df_with_flags['end_in_edge_box'] & 
                                            (self.df['pass_type'] != "Corner") &
                                            ~df_with_flags['is_in_edge_box'])
        
        df_with_flags['vertical_into_box'] = (df_with_flags['is_vertical'] & 
                                            df_with_flags['end_in_box'] & 
                                            (self.df['pass_type'] != "Corner") &
                                            ~df_with_flags['is_in_box'])
        
        df_with_flags['horizontal_into_box'] = (df_with_flags['is_horizontal'] & 
                                            df_with_flags['end_in_box'] & 
                                            (self.df['pass_type'] != "Corner") &
                                            ~df_with_flags['is_in_box'])
        
        df_with_flags['horizontal_into_edge'] = (df_with_flags['is_horizontal'] & 
                                                df_with_flags['end_in_edge_box'] & 
                                                (self.df['pass_type'] != "Corner") &
                                                ~df_with_flags['is_in_edge_box'])
        
        # More specialized combinations
        df_with_flags['cut_back_into_edge'] = (df_with_flags['pass_cut_back'] == True) & (df_with_flags["end_in_edge_box"])
        
        df_with_flags['high_vertical_into_edge'] = ((self.df['pass_height'] == "High Pass") & df_with_flags['vertical_into_edge'])
        df_with_flags['high_vertical_into_box'] = ((self.df['pass_height'] == "High Pass") & df_with_flags['vertical_into_box'])
        df_with_flags['low_vertical_into_edge'] = ((self.df['pass_height'] == "Ground Pass") & df_with_flags['vertical_into_edge'])
        df_with_flags['low_vertical_into_box'] = ((self.df['pass_height'] == "Ground Pass") & df_with_flags['vertical_into_box'])

        df_with_flags['high_horizontal_into_edge'] = ((self.df['pass_height'] == "High Pass") & df_with_flags['horizontal_into_edge'])
        df_with_flags['high_horizontal_into_box'] = ((self.df['pass_height'] == "High Pass") & df_with_flags['horizontal_into_box'])
        df_with_flags['low_horizontal_into_edge'] = ((self.df['pass_height'] == "Ground Pass") & df_with_flags['horizontal_into_edge'])
        df_with_flags['low_horizontal_into_box'] = ((self.df['pass_height'] == "Ground Pass") & df_with_flags['horizontal_into_box'])
        
        
        player_under_pressure_grouping = df_with_flags.groupby(['player', 'under_pressure']).agg(
            passes_total=('player', 'count'),
            passes_completed=('is_completed', 'sum'),
            passes_from_attacking_third=('is_attacking_third', 'sum'),
            passes_from_middle_third=('is_middle_third', 'sum'),
            passes_from_defending_third=('is_defending_third', 'sum'),
            passes_from_the_box=('is_in_box', 'sum'),
            passes_into_attacking_third=('end_attacking_third', 'sum'),
            passes_into_box=('into_box_not_from_box', 'sum'),
            passes_into_box_completed=('into_box_not_from_box_completed', 'sum'),
            passes_into_goal_area=("end_in_goal_area", "sum"),
            passes_into_edge_of_the_box=("into_edge_box_not_from_edge_box", "sum"),
            passes_into_edge_of_the_box_completed=("into_edge_box_not_from_edge_box_completed", "sum"),
            passes_cuts_last_line_of_defence=("is_through_ball", "sum"),
            passes_cuts_last_line_of_defence_completed=("is_through_ball_and_completed", "sum"),
            passes_within_attacking_third=("within_attacking_third", "sum"),
            passes_within_attacking_third_completed=("within_attacking_third_completed", "sum"),
            passes_total_distance=("pass_length", "sum"),
            passes_short_total=("is_short", "sum"),
            passes_short_completed=("is_short_and_completed", "sum"),
            passes_medium_total=("is_medium", "sum"),
            passes_medium_completed=("is_medium_and_completed", "sum"),
            passes_long_total=("is_long", "sum"),
            passes_long_completed=("is_long_and_completed", "sum"),
            passes_shot_assist=("is_pass_shot_assist", "sum"),
            passes_goal_assist=("is_pass_goal_assist", "sum"),
            passes_switch=("is_switch", "sum"),
            passes_cross=("is_cross", "sum"),
            passes_cut_back=("is_cut_back", "sum"),
            passes_vertical_into_edge_of_the_box=("vertical_into_edge", "sum"),
            passes_vertical_into_the_box=("vertical_into_box", "sum"),
            passes_horizontal_into_edge_of_the_box=("horizontal_into_edge", "sum"),
            passes_horizontal_into_the_box=("horizontal_into_box", "sum"),
            passes_cut_back_into_edge_of_the_box=("cut_back_into_edge", "sum"),
            passes_high_vertical_into_edge_of_the_box=("high_vertical_into_edge", "sum"),
            passes_high_vertical_into_the_box=("high_vertical_into_box", "sum"),
            passes_low_vertical_into_edge_of_the_box=("low_vertical_into_edge", "sum"),
            passes_low_vertical_into_the_box=("low_vertical_into_box", "sum"),
            passes_high_horizontal_into_edge_of_the_box=("high_horizontal_into_edge", "sum"),
            passes_high_horizontal_into_the_box=("high_horizontal_into_box", "sum"),
            passes_low_horizontal_into_edge_of_the_box=("low_horizontal_into_edge", "sum"),
            passes_low_horizontal_into_the_box=("low_horizontal_into_box", "sum"),
            passes_progressive=("is_progressive", "sum"),
            passes_progressive_completed=("is_progressive_and_completed", "sum"),
            passes_horizontal=("is_horizontal_and_completed", "sum"),
            passes_horizontal_completed=("is_horizontal_and_completed", "sum"),
            passes_backward=("is_backward", "sum"),
            passes_backward_completed=("is_backward_and_completed", "sum"),
            passes_long_vertical=("is_long_and_vertical", "sum"),
            passes_long_vertical_from_defending_third=("is_long_vertical_and_from_defending_third", "sum"),
            passes_long_vertical_from_mid_third=("is_long_vertical_and_from_mid_third", "sum"),
            passes_long_vertical_from_mid_third_into_the_box=("is_long_vertical_and_from_mid_third_into_the_box", "sum"),
        )
        
        player_under_pressure_grouping["key_passes"] = player_under_pressure_grouping['passes_shot_assist'] + player_under_pressure_grouping['passes_goal_assist']
        
        total_stats = player_under_pressure_grouping.groupby('player').sum()
        player_under_pressure_grouping = player_under_pressure_grouping.add_prefix('up_')
        player_under_pressure_grouping = player_under_pressure_grouping.reset_index()
        player_under_pressure_grouping = player_under_pressure_grouping[player_under_pressure_grouping["under_pressure"] == True]
        player_under_pressure_grouping = player_under_pressure_grouping.drop("under_pressure", axis=1)

        player_stats = pd.merge(left=total_stats, right=player_under_pressure_grouping, on="player")

        ### calculate relative values ###

        calculation_pairs = [
            ('passes_completed', 'passes_total', "pass_accuracy_%"),
            ('up_passes_completed', 'up_passes_total', "up_pass_accuracy_%"),
            ("passes_progressive_completed", "passes_progressive","pass_accuracy_progressive_%"),
            ("up_passes_progressive_completed", "up_passes_progressive","up_pass_accuracy_progressive_%"),
            ("passes_backward_completed", "passes_backward","pass_accuracy_backward_%"),
            ("passes_horizontal_completed", "passes_horizontal","pass_horizontal_backward_%"),
            ("up_passes_backward_completed", "up_passes_backward","up_pass_accuracy_backward_%"),
            ("passes_into_box_completed", "passes_into_box", "pass_accuracy_into_box_%"),
            ("passes_into_edge_of_the_box_completed", "passes_into_edge_of_the_box", "pass_accuracy_into_edge_of_the_box_%"),
            ("up_passes_into_box_completed", "up_passes_into_box", "up_pass_accuracy_into_box_%"),
            ("passes_short_completed" , "passes_short_total", "pass_accuaracy_short_%"),
            ("passes_medium_completed" , "passes_medium_total", "pass_accuaracy_medium_%"),
            ("passes_long_completed" , "passes_long_total", "pass_accuaracy_long_%"),
            ("passes_within_attacking_third_completed","passes_within_attacking_third", "pass_accuracy_within_attacking_third_%"),       
            ("up_passes_within_attacking_third_completed","up_passes_within_attacking_third", "up_pass_accuracy_within_attacking_third_%"),   
            ("passes_shot_assist", "passes_progressive", "pass_progressive_lead_to_shot_%" ),
            ("passes_goal_assist", "passes_progressive", "pass_progressive_lead_to_goal_%" ),
            ("passes_total_distance", "passes_total", "pass_mean_distance_%"),
            ("passes_cuts_last_line_of_defence","passes_cuts_last_line_of_defence_completed", "pass_accuracy_cuts_last_line_of_defence_%")
        ] # pass_accuracy_cuts_last_line_of_defence

        for a, b, c in calculation_pairs:
            player_stats[c] = (player_stats[f'{a}'] / player_stats[f'{b}'])
            # replace inf
            player_stats[c] = player_stats[c].replace([np.inf, -np.inf], 0)


        ###  calcuate stats per match ###

        # merge standard stats with absolute values (result_df)
        absolute_column_values = [col for col in player_stats.columns if not col.endswith("_%") ]
        df_stats_per_game = pd.merge(left=self.standard_stats, right=player_stats[absolute_column_values],on="player",how="left")
        df_stats_per_game = df_stats_per_game.fillna(0)

        # calcuate stats per match and add to result_df
        for col in df_stats_per_game.drop(["player", "full_match_equivalents"], axis=1).columns:
            col_name = f"{col}_per_match"
            df_stats_per_game[col_name] = (df_stats_per_game[col] / 90).round(3)

        # keep only per match stats
        column_per_match = [col for col in df_stats_per_game.columns if col.endswith("_per_match") or col=="player" ]
        df_stats_per_game = df_stats_per_game[column_per_match]

        # merge: abosulte, relative, per game values
        player_stats = pd.merge(left=player_stats, right=df_stats_per_game, on="player", how="right")
        player_stats = player_stats.fillna(0)

        return player_stats
    
    def store_data(self, df):
        folder_dir = f"{PROJECT_ROOT_DIR}/data/processed/{self.league}"
        os.makedirs(folder_dir, exist_ok=True)
        file_path = f"{folder_dir}/{self.dim}.csv"
        df.to_csv(file_path,index=False)

    def run(self):
        self.filter_by_dimension()
        self.convert_columns()
        result_df = self.extract()
        self.store_data(result_df)