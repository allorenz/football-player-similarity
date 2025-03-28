from feature_extraction.base_extractor import BaseDimensionFeatureExtractor
from ast import literal_eval
import pandas as pd
import numpy as np
import os
from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent

def is_ex_inside_defending_goal_area(row):
    #print("Executing is_in_goal_area method")
    pitch_width = 120
    x = row["x"]
    y = row["y"]

    x_axis = ((0 <= x) and (x <= 6))
    y_axis = ((40 - 10) <= y and y <= (40 + 10))
    return x_axis and y_axis

def is_ex_inside_defending_penalty_area(row):
    #print("Executing is_in_penalty_area method")
    pitch_width = 120
    x = row["x"]
    y = row["y"]

    if is_ex_inside_defending_goal_area(row):
        return False
    else:
        x_axis = ((0 <= x) and (x <= 18)) 
        y_axis = ((40 - 22) <= y and y <= (40 + 22))
        return x_axis and y_axis

def is_ex_inside_defending_third(row):
    #print("Executing is_in_attacking_third method")
    pitch_width = 120
    x = row["x"]
    #y = row["y"]

    if is_ex_inside_defending_goal_area(row) or is_ex_inside_defending_penalty_area(row):
        return False
    else:
        return (x <= 40) 

def convert_to_list(input_data):
    if(isinstance(input_data, str)):
        try:
            return literal_eval(input_data)
        except (ValueError, SyntaxError):
            print(f"Error: The string {input_data} could not be converted to a list.")
            return None
    
    return input_data

class GoalKeepingFeatureExtractor(BaseDimensionFeatureExtractor):

    def __init__(self, df: pd.DataFrame, standard_stats: pd.DataFrame, league:str, dim:str):
        self.df = df
        self.standard_stats = standard_stats
        self.league = league
        self.dim = dim

    def convert_columns(self):
        # Implement passing-specific feature extraction
        self.df["location"] = self.df["location"].apply(convert_to_list)
        self.df[["x", "y"]] = self.df["location"].apply(pd.Series)
        self.df["under_pressure"] = self.df["under_pressure"].fillna(False)

    def extract(self):
        """
        This function Pre-compute all conditions for Vectorize operations.
        Returns results grouped by player and under_pressure  
        """
        # Pre-compute all conditions
        df_with_flags = self.df.copy()

        # Action flags
        df_with_flags["is_shot_on_target"] = (self.df["goalkeeper_type"] =="Goal Conceded") | (self.df["goalkeeper_type"] =="Shot Saved") # nan for "Shot Saved Off T", "Shot Saved To Post"
        df_with_flags["goal_conceded"] = self.df["goalkeeper_type"] =="Goal Conceded"
        df_with_flags["is_dive"] = self.df["goalkeeper_technique"] =="Diving"
        df_with_flags["is_shot_saved"] = self.df["goalkeeper_type"] =="Shot Saved"
        df_with_flags["is_sweeper"] = self.df["goalkeeper_type"] =="Keeper Sweeper"
        df_with_flags["is_standing"] = self.df["goalkeeper_technique"] =="Standing"
        df_with_flags["is_penalty"] = (self.df["goalkeeper_type"] =="Penalty Conceded") | (self.df["goalkeeper_type"] =="Penalty Saved") | (self.df["goalkeeper_type"] =="Penalty Saved To Post")
        

        df_with_flags["is_penalty_saved"] = (self.df["goalkeeper_type"] =="Penalty Saved") | (self.df["goalkeeper_type"] =="Penalty Saved To Post")
        df_with_flags["is_punch"] = self.df["goalkeeper_type"] =="Punch"
        df_with_flags["is_shot_faced"] = self.df["goalkeeper_type"] =="Shot Faced"
        df_with_flags["is_ball_collecting"] = self.df["goalkeeper_type"] =="Collected"
        df_with_flags["is_smother"] = self.df["goalkeeper_type"] =="Smother"

        # location flags
        df_with_flags["distance_to_goal"] = np.sqrt( (self.df["x"] - 0)**2 + (self.df["y"] - 40)**2)
        df_with_flags["is_in_goal_area"] = self.df[["x","y"]].apply(is_ex_inside_defending_goal_area, axis=1)
        df_with_flags["is_in_penalty_area"] = self.df[["x","y"]].apply(is_ex_inside_defending_penalty_area, axis=1)
        df_with_flags["is_in_defending_third"] = self.df[["x","y"]].apply(is_ex_inside_defending_third, axis=1)
        df_with_flags["is_in_middle_third"] = (80 > self.df['x']) & (self.df['x'] > 40)

        # combinations
        df_with_flags["is_saved_shot"] = df_with_flags["is_shot_on_target"] & df_with_flags["is_shot_saved"]
        df_with_flags["is_saved_from_dive"] = df_with_flags["is_saved_shot"] & df_with_flags["is_dive"]
        df_with_flags["is_saved_from_standing"] = df_with_flags["is_saved_shot"] & df_with_flags["is_standing"]
        df_with_flags["is_sweeper_and_clears_ball"] = (self.df["goalkeeper_type"] =="Keeper Sweeper") & (self.df["goalkeeper_outcome"] =="Clear")
        df_with_flags["is_sweeper_and_collects_ball"] = (self.df["goalkeeper_type"] =="Keeper Sweeper") & (self.df["goalkeeper_outcome"] =="Claim")
        df_with_flags["punch_that_cleared_situation"] = (df_with_flags["is_punch"] ) & ((self.df["goalkeeper_outcome"] =="Punched Out") | (self.df["goalkeeper_outcome"] =="In Play Safe") )
        df_with_flags["punch_that_creates_danger"] = (df_with_flags["is_punch"] ) & (self.df["goalkeeper_outcome"] =="In Play Danger")
        df_with_flags["save_that_creates_danger"] = (df_with_flags["is_shot_saved"] ) & (self.df["goalkeeper_outcome"] =="In Play Danger")
        df_with_flags["ball_collection_failed"] = (df_with_flags["is_ball_collecting"] ) & (self.df["goalkeeper_outcome"] =="Fail")
        df_with_flags["ball_collection_two_attemps_needed"] = (df_with_flags["is_ball_collecting"] ) & (self.df["goalkeeper_outcome"] =="Collected Twice")
        df_with_flags["saved_both_hands"] = (self.df["goalkeeper_body_part"] =="Both Hands") & (df_with_flags["is_shot_saved"])
        df_with_flags["saved_chest"] = (self.df["goalkeeper_body_part"] =="Chest") & (df_with_flags["is_shot_saved"])
        df_with_flags["saved_head"] = (self.df["goalkeeper_body_part"] =="Head") & (df_with_flags["is_shot_saved"])
        df_with_flags["saved_left_foot"] = (self.df["goalkeeper_body_part"] =="Left Foot") & (df_with_flags["is_shot_saved"])
        df_with_flags["saved_right_foot"] = (self.df["goalkeeper_body_part"] =="Right Foot") & (df_with_flags["is_shot_saved"])
        df_with_flags["saved_left_hand"] = (self.df["goalkeeper_body_part"] =="Left Hand") & (df_with_flags["is_shot_saved"])
        df_with_flags["saved_right_hand"] = (self.df["goalkeeper_body_part"] =="Right Hand") & (df_with_flags["is_shot_saved"])
        df_with_flags["smoother_is_successful"] = (df_with_flags["is_smother"]) & ((self.df["goalkeeper_outcome"] =="Won")|(self.df["goalkeeper_outcome"] =="Success")|(self.df["goalkeeper_outcome"] =="Success In Play")|
                                                    (self.df["goalkeeper_outcome"] =="Success Out"))
        df_with_flags["smoother_is_unsuccessful"] = (df_with_flags["is_smother"]) & (self.df["goalkeeper_outcome"] =="Lost Out")
        df_with_flags["is_sweeper_in_goal_area"] = (df_with_flags["is_sweeper"]) & (df_with_flags["is_in_goal_area"])
        df_with_flags["is_sweeper_in_penalty_area"] = (df_with_flags["is_sweeper"]) & (df_with_flags["is_in_penalty_area"])
        df_with_flags["is_sweeper_in_defending_third"] = (df_with_flags["is_sweeper"]) & (df_with_flags["is_in_defending_third"])
        df_with_flags["is_sweeper_in_middle_third"] = (df_with_flags["is_sweeper"]) & (df_with_flags["is_in_middle_third"])
        df_with_flags["action_outside_penalty_area"] = ((df_with_flags["is_in_defending_third"] ) | (df_with_flags["is_in_middle_third"])) & ((df_with_flags["is_smother"] ) | (df_with_flags["is_sweeper"]))



        total_stats = df_with_flags.groupby(['player']).agg(
                # general actions
                actions_total=('player', 'count'),
                shots_faced=("is_shot_faced","sum"),
                goals_conceded=("goal_conceded","sum"), # doesnt account own goals
                defensive_actions_outside_penalty_area=("action_outside_penalty_area","sum"),
                
                # shots and saves
                shots_on_target=("is_shot_on_target","sum"),
                saved_shots=("is_saved_shot","sum"),
                saved_shot_creates_danger=("save_that_creates_danger", "sum"),
                saved_shots_head=("saved_head","sum"),
                saved_shots_chest=("saved_chest","sum"),
                saved_shots_both_hands=("saved_both_hands","sum"),
                saved_shots_left_hand=("saved_left_hand","sum"),
                saved_shots_right_hand=("saved_right_hand","sum"),
                saved_shots_left_foot=("saved_left_foot","sum"),
                saved_shots_right_foot=("saved_right_foot","sum"),
                # diving
                dives_total=("is_dive","sum"),
                dives_saved_shots =("is_saved_from_dive","sum"),
                # standing - save from standing position
                standings_total=("is_standing","sum"),
                standing_saved_shots=("is_saved_from_standing","sum"),
                # penalties
                penalty_total=("is_penalty","sum"),
                penalty_saved=("is_penalty_saved","sum"),
                # punches
                punch_total=("is_punch","sum"),
                punch_clears_ball=("punch_that_cleared_situation", "sum"),
                punch_creates_danger=("punch_that_creates_danger", "sum"),
                # smother - comes out and tackles/dispossesses
                tackling_total = ("is_smother","sum"),
                tackling_successful=("smoother_is_successful","sum"),
                tackling_unsuccessful=("smoother_is_unsuccessful","sum"),
                # sweeper - comes out/off his line to collect the ball
                sweeper_total=("is_sweeper","sum"),
                sweeper_in_goal_area=("is_sweeper_in_goal_area","sum"),
                sweeper_in_penalty_area=("is_sweeper_in_penalty_area","sum"),
                sweeper_in_defending_third=("is_sweeper_in_defending_third","sum"),
                sweeper_in_middle_third=("is_sweeper_in_middle_third","sum"),
                sweeper_collects_ball=("is_sweeper_and_collects_ball","sum"),
                sweeper_clears_ball=("is_sweeper_and_clears_ball","sum"),
                # collecting the ball
                collecting_ball_total=("is_ball_collecting","sum"),
                collecting_ball_failed=("ball_collection_failed","sum"),
                collecting_ball_in_second_attempt=("ball_collection_two_attemps_needed", "sum"), # e.g. due to misjudgment or dense goal area
                # distance to goal
                avg_distance_to_goal=("distance_to_goal",lambda x: (x).mean()),
                avg_distance_to_goal_saved_shots=("distance_to_goal", lambda x: x[df_with_flags["is_saved_shot"] == True].mean()),
                avg_distance_to_goal_goals_conceded=("distance_to_goal", lambda x: x[df_with_flags["goal_conceded"] == True].mean()),

        )

        ### calculate relative values ###

        calculation_pairs = [
                ("shots_on_target","shots_faced","shots_on_target_%"),
                ("goals_conceded","shots_on_target","save_%"),
                ("penalty_saved","penalty_total","penalty_save_%"),
            ]

        for a, b, c in calculation_pairs:
            total_stats[c] = (total_stats[f'{a}'] / total_stats[f'{b}'])
            total_stats[c] = total_stats[c].replace([np.inf, -np.inf], 0)
        total_stats = total_stats.fillna(0)    
        total_stats = total_stats.reset_index(drop=False)

        ###  calcuate stats per match ###

        # merge standard stats with absolute values (result_df)
        absolute_column_values = [col for col in total_stats.columns if not col.endswith("_%") ]
        df_stats_per_game = pd.merge(left=self.standard_stats, right=total_stats[absolute_column_values],on="player",how="left")

        # calcuate stats per match and add to result_df
        for col in df_stats_per_game.drop(["player", "full_match_equivalents"], axis=1).columns:
            col_name = f"{col}_per_match"
            df_stats_per_game[col_name] = (df_stats_per_game[col] / 90).round(3)

        # keep only per match stats
        column_per_match = [col for col in df_stats_per_game.columns if col.endswith("_per_match") or col=="player" ]
        df_stats_per_game = df_stats_per_game[column_per_match]

        # merge: abosulte, relative, per game values
        total_stats = pd.merge(left=total_stats, right=df_stats_per_game, on="player", how="right")

        # many field players have no goal keeper attributes
        total_stats = total_stats.fillna(0)

        return total_stats

    def store_data(self, df):
        folder_dir = f"{PROJECT_ROOT_DIR}/data/processed/{self.league}"
        os.makedirs(folder_dir, exist_ok=True)
        file_path = f"{folder_dir}/{self.dim}.csv"
        df.to_csv(file_path,index=False)


    def run(self):
        self.convert_columns()
        result_df = self.extract()
        self.store_data(result_df)
