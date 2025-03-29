import os
import numpy as np
import pandas as pd
import json
from ast import literal_eval
from pathlib import Path
from feature_extraction.base_extractor import BaseDimensionFeatureExtractor


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
    x_axis = ((102 <= x) and (x <= pitch_width))  # Stop before goal area
    y_axis = ((40 - 22) <= y and y <= (40 + 22))
    return x_axis and y_axis

def is_in_defensive_penalty_area(x, y):
    x_axis = ((0 <= x) and (x <= 18))  # Stop before goal area
    y_axis = ((40 - 22) <= y and y <= (40 + 22))
    return x_axis and y_axis

def is_in_edge_of_the_box(x,y):
    pitch_width = 120
    x_axis = ((102 <= x) and (x <= pitch_width))
    y_axis = ((40 - 10) <= y and y <= (40 + 10))
    return x_axis and y_axis

def get_shot_distance(x_start,y_start,x_end,y_end):
    #print("Executing is_short_distance_shot method")

    start = np.array([x_start, y_start])
    end = np.array([x_end, y_end])

    return np.linalg.norm(start - end)

def is_progressive_carry(start_x, start_y, end_x, end_y):
    """
    Check if a carry is progressive towards the goal area.
    
    A carry is progressive if the end position is closer to the goal area than the start position.
    
    Args:
        start_x (float): x-coordinate of the carry start position
        start_y (float): y-coordinate of the carry start position
        end_x (float): x-coordinate of the carry end position
        end_y (float): y-coordinate of the carry end position
        pitch_width (float): width of the pitch
        
    Returns:
        bool: True if the carry is progressive, False otherwise
    """
    goal_center_x = 120
    goal_center_y = 40
    
    # Calculate distances from start and end points to the goal center
    start_distance = ((start_x - goal_center_x) ** 2 + (start_y - goal_center_y) ** 2) ** 0.5
    end_distance = ((end_x - goal_center_x) ** 2 + (end_y - goal_center_y) ** 2) ** 0.5
    
    # A carry is progressive if the end point is closer to the goal center than the start point
    return end_distance < start_distance


class PossessionFeatureExtractor(BaseDimensionFeatureExtractor):

    def __init__(self, df: pd.DataFrame, standard_stats: pd.DataFrame, league:str):
        self.df = df
        self.standard_stats = standard_stats
        self.league = league
        self.dim = "possession"

        with open(f"{PROJECT_ROOT_DIR}/config/columns_config.json", 'r') as f:
            self.columns = json.load(f)

    def filter_by_dimension(self):
        self.df = self.df.loc[self.df["type"].isin(self.columns[self.dim]["row_filter"]), self.columns[self.dim]["columns"]]

    def convert_columns(self):
        self.df["location"] = self.df["location"].apply(convert_to_list)
        self.df["carry_end_location"] = self.df["carry_end_location"].apply(convert_to_list)
        self.df[["x", "y"]] = self.df["location"].apply(pd.Series)
        self.df[["x_end_carry", "y_end_carry"]] = self.df["carry_end_location"].apply(pd.Series)
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
        df_with_flags['is_in_defending_box'] = self.df[["x","y"]].apply(lambda row: is_in_defensive_penalty_area(row['x'], row['y']), axis=1)
        df_with_flags['ends_attacking_third'] = self.df['x_end_carry'] >= 80
        df_with_flags['ends_in_box'] = self.df[['x_end_carry','y_end_carry']].apply(lambda row: is_in_penalty_area(row['x_end_carry'], row['y_end_carry']), axis=1)
        df_with_flags['ends_in_edge_box'] = self.df[['x_end_carry','y_end_carry']].apply(lambda row: is_in_edge_of_the_box(row['x_end_carry'], row['y_end_carry']), axis=1)
        df_with_flags["carry_distance"] = np.sqrt((self.df["x_end_carry"] - self.df["x"])**2 + (self.df["y_end_carry"] - self.df["y"])**2)
        

        # Check if progressive carry
        df_with_flags["start_to_goal_distance"] = np.sqrt( (self.df["x"] - 120)**2 + (self.df["y"] - 40)**2)
        df_with_flags["end_to_goal_distance"] = np.sqrt((self.df["x_end_carry"] - 120)**2 + (self.df["y_end_carry"] - 40)**2)
        df_with_flags["is_progressive_carry"] = df_with_flags["end_to_goal_distance"] < df_with_flags["start_to_goal_distance"]


        # Action flags
        df_with_flags['is_dribbling'] = self.df['type'] == "Dribble"
        df_with_flags['is_overrun'] = self.df['dribble_overrun'] == True
        df_with_flags['is_nutmeg'] = self.df['dribble_nutmeg'] == True
        df_with_flags['is_no_touch'] = self.df['dribble_no_touch'] == True
        df_with_flags['is_carry'] = self.df['type'] == "Carry"
        df_with_flags['is_touch'] = (self.df['type'] != "Dispossessed") & (self.df['type'] != "Error")
        df_with_flags['is_miscontrol'] = self.df['type'] == "Miscontrol"
        df_with_flags['is_miscontrol_in_aerial_duel'] = self.df['miscontrol_aerial_won'] == True
        df_with_flags['is_dispossessed'] =  self.df["type"]=='Dispossessed' # (self.df['duel_outcome'] == "Lost") | (self.df['duel_outcome'] == "Lost In Play") | (self.df['duel_outcome'] == "Lost Out")
        df_with_flags['balls_received'] = self.df["type"]=='Ball Receipt*'
        df_with_flags['balls_received_successful'] = (df_with_flags['balls_received']) & (self.df["ball_receipt_outcome"].isna())
        df_with_flags['is_error'] = self.df["type"]=='Error'
        df_with_flags['is_offside'] = self.df["type"]=='Offside'

        # Combination of flags
        df_with_flags['is_dribbling_and_completed'] = df_with_flags['is_dribbling'] & (self.df["dribble_outcome"] == "Complete")
        df_with_flags['is_dribbling_and_is_not_completed'] = df_with_flags['is_dribbling'] & (self.df["dribble_outcome"] == "Incomplete")
        df_with_flags['is_dribbling_and_overrun'] = df_with_flags['is_dribbling'] & (df_with_flags['is_overrun'])
        df_with_flags['is_dribbling_and_through_legs'] = df_with_flags['is_dribbling'] & (df_with_flags['is_nutmeg'])
        df_with_flags['is_dribbling_and_no_touch'] = df_with_flags['is_dribbling'] & (df_with_flags['is_no_touch'])
        df_with_flags['is_carry_from_outside_inside_attacking_third'] = (~df_with_flags['is_attacking_third']) & (df_with_flags['ends_attacking_third'])
        df_with_flags['is_carry_from_outside_inside_penalty_area'] = (~df_with_flags['is_in_box']) & (df_with_flags['ends_in_box'])
        df_with_flags['is_carry_from_outside_inside_edge_of_the_box'] = (~df_with_flags['is_in_edge_box']) & (df_with_flags['ends_in_edge_box'])
        df_with_flags['is_dispossessed_in_defending_penalty_area'] = df_with_flags['is_dispossessed'] & df_with_flags['is_in_defending_box']
        df_with_flags['is_dispossessed_in_defending_third'] = df_with_flags['is_dispossessed'] & df_with_flags['is_defending_third']
        df_with_flags['is_dispossessed_in_middle_third'] = df_with_flags['is_dispossessed'] & df_with_flags['is_middle_third']
        df_with_flags['is_dispossessed_in_attacking_third'] = df_with_flags['is_dispossessed'] & df_with_flags['is_attacking_third']
        df_with_flags['is_dispossessed_in_penalty_area'] = df_with_flags['is_dispossessed'] & df_with_flags['is_in_box']


        player_under_pressure_grouping = df_with_flags.groupby(['player', 'under_pressure']).agg(
            # touches
            touches_total=('is_touch', 'count'),
            touches_in_defending_penalty=("is_in_defending_box", "sum"),
            touches_in_defending_third=("is_defending_third", "sum"),
            touches_in_middle_third=("is_middle_third", "sum"),
            touches_in_attacking_third=("is_attacking_third", "sum"),
            touches_in_attacking_penalty=("is_in_box", "sum"),
            # dribblings
            dribbling_total=("is_dribbling", "sum"),
            dribbling_completed=("is_dribbling_and_completed", "sum"),
            dribbling_not_completed=("is_dribbling_and_is_not_completed", "sum"),
            dribbling_overrun=("is_dribbling_and_overrun", "sum"),
            dribbling_through_legs=("is_dribbling_and_through_legs", "sum"),
            dribbling_push_and_run=("is_dribbling_and_no_touch", "sum"),
            # carries
            carries_total=("is_carry","sum"),
            carry_into_attacking_third=("is_carry_from_outside_inside_attacking_third","sum"),
            carry_into_penalty_area=("is_carry_from_outside_inside_penalty_area","sum"),
            carry_into_edge_of_the_box=("is_carry_from_outside_inside_edge_of_the_box","sum"),
            carry_distance=("carry_distance", "sum"),
            # progressive carries
            progressive_carry_total=("is_progressive_carry","sum"),
            progressive_carry_distance=("carry_distance", lambda x: x[df_with_flags["is_progressive_carry"]].sum()),
            # miscontrols and dispossesions
            miscontrol_total=("is_miscontrol","sum"),
            miscontrol_in_aerial_duel=("is_miscontrol_in_aerial_duel", "sum"),
            dispossessed_total=("is_dispossessed", "sum"),
            dispossessed_in_defending_penalty_area=("is_dispossessed_in_defending_penalty_area", "sum"),
            dispossessed_in_defending_third=("is_dispossessed_in_defending_third", "sum"),
            dispossessed_in_middle_third=("is_dispossessed_in_middle_third", "sum"),
            dispossessed_in_attacking_third=("is_dispossessed_in_attacking_third", "sum"),
            dispossessed_in_penalty_area=("is_dispossessed_in_penalty_area", "sum"),
            # ball receive
            balls_received=("balls_received", "sum"),
            balls_received_successful=("balls_received_successful","sum"),
            # error an offside
            error_lead_to_goal=("is_error","sum"),
            offside=("is_offside", "sum")
        )

        total_stats = player_under_pressure_grouping.groupby('player').sum()
        player_under_pressure_grouping = player_under_pressure_grouping.add_prefix('up_')
        player_under_pressure_grouping = player_under_pressure_grouping.reset_index()
        player_under_pressure_grouping = player_under_pressure_grouping[player_under_pressure_grouping["under_pressure"] == True]
        player_under_pressure_grouping = player_under_pressure_grouping.drop("under_pressure", axis=1)

        player_stats = pd.merge(left=total_stats, right=player_under_pressure_grouping, on="player")

        ### calculate relative values ###

        calculation_pairs = [
            ("touches_in_defending_penalty","touches_total","touches_in_defending_penalty_%"),
            ("touches_in_defending_third","touches_total","touches_in_defending_third_%"),
            ("touches_in_middle_third","touches_total","touches_in_middle_third_%"),
            ("touches_in_attacking_third","touches_total","touches_in_attacking_third_%"),
            ("touches_in_attacking_penalty","touches_total","touches_in_attacking_penalty_%"),
            ("balls_received_successful", "balls_received", "ball_reception_%"),
            ("dribbling_completed", "dribbling_total", "dribblings_successful_%"),
            ("carry_distance", "carries_total", "distance_per_carry_%"),
            ("carry_into_attacking_third","carries_total", "carries_into_attacking_third_%"),
            ("carry_into_penalty_area","carries_total", "carries_into_penalty_area_%"),
            ("carry_into_edge_of_the_box","carries_total", "carries_into_edge_of_the_box_%"),
            ("progressive_carry_distance", "progressive_carry_total", "distance_per_progressive_carry"),
            ("dispossessed_total","carries_total","dispossessed_per_carry_%")
        ]

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