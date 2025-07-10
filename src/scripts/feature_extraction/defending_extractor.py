import sys
import os
import json
import pandas as pd
import numpy as np
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

def is_in_defensive_penalty_area(x, y):
    x_axis = ((0 <= x) and (x <= 18))  # Stop before goal area
    y_axis = ((40 - 22) <= y and y <= (40 + 22))
    return x_axis and y_axis

class DefendingFeatureExtractor(BaseDimensionFeatureExtractor):

    def __init__(self, df: pd.DataFrame, standard_stats: pd.DataFrame, league:str):
        self.df = df
        self.standard_stats = standard_stats
        self.league = league
        self.dim = "defending"

        with open(f"{PROJECT_ROOT_DIR}/config/columns_config.json", 'r') as f:
            self.columns = json.load(f)

    def filter_by_dimension(self):
        try:
            self.df = self.df.loc[self.df["type"].isin(self.columns[self.dim]["row_filter"]), self.columns[self.dim]["columns"]]
        except:
            """
            UCL event data doesn't provide all neccassry column due to not occuring event during the matches.
            Solution: Fill up missing columns with appropriate values (i.e. False, nan). 
            """
            required_columns = self.columns[self.dim]["columns"]
            available_columns = self.df.columns
            missing_columns = [col for col in required_columns if col not in available_columns]
            
            for col in missing_columns:
                self.df[col] = False


    def convert_columns(self):
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

        # Actions
        df_with_flags["is_ball_recovery"] = self.df["type"] == "Ball Recovery"
        df_with_flags["is_ball_recovery_successful"] = self.df["ball_recovery_recovery_failure"].isna()
        df_with_flags["is_offensive_ball_recovery"] = self.df["ball_recovery_offensive"] == True
        df_with_flags["is_ball_recovery_failed"] = self.df["ball_recovery_recovery_failure"] == True
        df_with_flags["is_bad_behaviour"] = self.df["type"]=="Bad Behaviour"
        df_with_flags["is_yellow_card_bad_behaviour"] = (self.df["bad_behaviour_card"] == "Yellow Card") 
        df_with_flags["is_second_yellow_card_bad_behaviour"] = self.df["bad_behaviour_card"] == "Second Yellow"
        df_with_flags["is_red_card_bad_behaviour"] = self.df["bad_behaviour_card"] == "Red Card"
        df_with_flags["is_yellow_card_foul"] = (self.df["foul_committed_card"] == "Yellow Card")
        df_with_flags["is_second_yellow_card_foul"] = (self.df["foul_committed_card"] == "Second Yellow")
        df_with_flags["is_red_card_foul"] = (self.df["foul_committed_card"] == "Red Card")    
        df_with_flags["is_counterpress"] = self.df["counterpress"].notna()
        df_with_flags["is_pressure_on_opponent"] = self.df["type"] == "Pressure"
        df_with_flags["is_block"] = self.df["type"] == "Block"
        df_with_flags["is_block_offensive"] = self.df["block_offensive"] == True
        df_with_flags["is_block_ball_deflection"] = self.df["block_deflection"] == True
        df_with_flags["is_block_shot_on_target"] = self.df["block_save_block"] == True
        df_with_flags["is_clearance"] = self.df["type"] == "Clearance"
        df_with_flags["is_duel"] = self.df["type"] == "Duel"
        df_with_flags["is_foul"] = self.df["type"] == "Foul Committed"
        df_with_flags["is_offensive_foul"] = self.df["foul_committed_offensive"] == True
        df_with_flags["is_foul_penalty_resulted"] = self.df["foul_committed_penalty"] == True
        df_with_flags["teammate_is_fouled_in_op_penalty"] = self.df["foul_won_penalty"] == True
        df_with_flags["is_interception"] =(self.df["interception_outcome"]=="Success In Play") | (self.df["interception_outcome"]=="Won") 
        df_with_flags["is_shield"] = self.df["type"] == "Shield"


        # Location-based flags
        df_with_flags['is_attacking_third'] = self.df['x'] >= 80
        df_with_flags['is_middle_third'] = (80 > self.df['x']) & (self.df['x'] > 40)
        df_with_flags['is_defending_third'] = self.df['x'] <= 40
        df_with_flags['is_in_defending_box'] = self.df[["x","y"]].apply(lambda row: is_in_defensive_penalty_area(row['x'], row['y']), axis=1)

        # Combinations
        df_with_flags["ball_recovery_successful"] = df_with_flags["is_ball_recovery"] & df_with_flags["is_ball_recovery_successful"]
        df_with_flags["ball_recovery_failed"] = (df_with_flags["is_ball_recovery"])  & (df_with_flags["is_ball_recovery_failed"])
        df_with_flags["ball_recovery_offensive_successful"] = df_with_flags["is_offensive_ball_recovery"] & df_with_flags["is_ball_recovery_successful"]
        df_with_flags["block_during_counterpress"] = df_with_flags["is_block"] & df_with_flags["is_block"]
        df_with_flags["is_clearance_with_head"] = (df_with_flags["is_clearance"]) & (self.df["clearance_body_part"] == "Head")
        df_with_flags["is_duel_tackling"] = (df_with_flags["is_duel"]) & (self.df["duel_type"] == "Tackle")
        df_with_flags["is_duel_tackling_won"] = (df_with_flags["is_duel"]) & (self.df["duel_type"] == "Tackle") & ((self.df["duel_outcome"] == "Won") | (self.df["duel_outcome"] == "Success In Play"))
        df_with_flags["is_pressure_on_opponent_defending_third"] = df_with_flags["is_pressure_on_opponent"] & df_with_flags['is_defending_third']
        df_with_flags["is_pressure_on_opponent_middle_third"] = df_with_flags["is_pressure_on_opponent"] & df_with_flags['is_middle_third']
        df_with_flags["is_pressure_on_opponent_attacking_third"] = df_with_flags["is_pressure_on_opponent"] & df_with_flags['is_attacking_third']
        df_with_flags["is_counterpress_on_opponent_defending_third"] = df_with_flags["is_counterpress"] & df_with_flags['is_defending_third']
        df_with_flags["is_counterpress_on_opponent_middle_third"] = df_with_flags["is_counterpress"] & df_with_flags['is_middle_third']
        df_with_flags["is_counterpress_on_opponent_attacking_third"] = df_with_flags["is_counterpress"] & df_with_flags['is_attacking_third']
        df_with_flags["is_tackling_in_defending_third"] = df_with_flags["is_duel_tackling"] & df_with_flags['is_defending_third']
        df_with_flags["is_tackling_in_middle_third"] = df_with_flags["is_duel_tackling"] & df_with_flags['is_middle_third']
        df_with_flags["is_tackling_in_attacking_third"] = df_with_flags["is_duel_tackling"] & df_with_flags['is_attacking_third']

        df_with_flags["is_foul_in_defending_third"] = df_with_flags["is_foul"] & df_with_flags['is_defending_third']
        df_with_flags["is_foul_in_middle_third"] = df_with_flags["is_foul"] & df_with_flags['is_middle_third']
        df_with_flags["is_foul_in_attacking_third"] = df_with_flags["is_foul"] & df_with_flags['is_attacking_third']

        df_with_flags["total_yellow_card"] = df_with_flags["is_yellow_card_foul"] | df_with_flags["is_yellow_card_foul"] | df_with_flags["is_second_yellow_card_bad_behaviour"] | df_with_flags["is_yellow_card_bad_behaviour"]
        df_with_flags["total_red_card"] = df_with_flags["is_red_card_foul"] | df_with_flags["is_red_card_bad_behaviour"]
        

        total_stats = df_with_flags.groupby('player_id').agg(
            # ball recovery
            ball_recovery_total=('is_ball_recovery', 'sum'),
            ball_recovery_successful=("ball_recovery_successful","sum"),
            ball_recovery_offensive_total=("is_offensive_ball_recovery", "sum"),
            ball_recovery_offensive_successful=("ball_recovery_offensive_successful", "sum"),
            ball_recovery_failed=("ball_recovery_failed", "sum"),
            # pressure 
            pressure_on_opponent=("is_pressure_on_opponent","sum"),
            pressure_on_opponent_defending_third=("is_pressure_on_opponent_defending_third","sum"),
            pressure_on_opponent_middle_third=("is_pressure_on_opponent_middle_third","sum"),
            pressure_on_opponent_attacking_third=("is_pressure_on_opponent_attacking_third","sum"),
            # counterpressing
            counterpressing_total = ("is_counterpress","sum"),
            counterpressing_opponent_defending_third = ("is_counterpress_on_opponent_defending_third","sum"),
            counterpressing_opponent_middle_third = ("is_counterpress_on_opponent_middle_third","sum"),
            counterpressing_attacking_third = ("is_counterpress_on_opponent_attacking_third","sum"),
            # shields
            shield_total=("is_shield","sum"),
            # blocking
            block_total = ("is_block", "sum"),
            block_offensive = ("is_block_offensive", "sum"),
            block_ball_deflection = ("is_block_ball_deflection", "sum"),
            block_counterpress = ("is_block_ball_deflection", "sum"),
            block_during_counterpress =("block_during_counterpress", "sum"),
            block_shot_on_target=("is_block_shot_on_target", "sum"),
            # clearance
            clearance_total=("is_clearance","sum"),
            clearance_with_head=("is_clearance_with_head","sum"),
            # interception
            interception=("is_interception","sum"),
            # duel
            tackling=("is_duel_tackling","sum"),
            tackling_won=("is_duel_tackling_won","sum"),
            tackling_in_defending_third=("is_tackling_in_defending_third","sum"),
            tackling_in_middle_third=("is_tackling_in_middle_third","sum"),
            tackling_in_attacking_third=("is_tackling_in_attacking_third","sum"),
            # fouls
            fouls_total=("is_foul","sum"),
            fouls_in_defending_third=("is_foul_in_defending_third","sum"),
            fouls_in_middle_third=("is_foul_in_middle_third","sum"),
            fouls_in_attacking_third=("is_foul_in_attacking_third","sum"),
            fouls_offensive=("is_offensive_foul","sum"),
            fouls_lead_to_penalty=("is_foul_penalty_resulted","sum"),
            fouls_wins_a_penalty=("teammate_is_fouled_in_op_penalty","sum"),
            fouls_yellow_card=("is_yellow_card_foul", "sum"),
            fouls_second_yellow_card=("is_second_yellow_card_foul", "sum"),
            fouls_red_card=("is_red_card_foul", "sum"),
            # bad_behaviour
            bad_behaviour_total=("is_bad_behaviour", "sum"),
            bad_behaviour_yellow_card=("is_yellow_card_bad_behaviour", "sum"),
            bad_behaviour_second_yellow_card=("is_second_yellow_card_bad_behaviour", "sum"),
            bad_behaviour_red_card=("is_red_card_bad_behaviour", "sum"),
            # total cards
            total_yellow = ("total_yellow_card", "sum"),
            total_red = ("total_red_card", "sum"),
        )

        ### calculate relative values ###

        calculation_pairs = [
            ("ball_recovery_successful","ball_recovery_total","ball_recovery_successful_%"),
            ("pressure_on_opponent_defending_third","pressure_on_opponent","pressure_on_opponent_defending_third_%"),
            ("pressure_on_opponent_middle_third","pressure_on_opponent","pressure_on_opponent_middle_third_%"),
            ("pressure_on_opponent_attacking_third","pressure_on_opponent","pressure_on_opponent_attacking_third_%"),
            ("counterpressing_opponent_defending_third","counterpressing_total","counterpressing_opponent_defending_third_%"),
            ("counterpressing_opponent_middle_third","counterpressing_total","counterpressing_opponent_middle_third_%"),
            ("counterpressing_attacking_third","counterpressing_total","counterpressing_opponent_attacking_third_%"),
            ("tackling_won","tackling","tackling_success_%"),
            ("tackling_in_defending_third","tackling","tackling_in_defending_third_%"),
            ("tackling_in_middle_third","tackling","tackling_in_middle_third_%"),
            ("tackling_in_attacking_third","tackling","tackling_in_attacking_third_%"),
            ("fouls_in_defending_third","fouls_total","fouls_in_defending_third_%"),
            ("fouls_in_middle_third","fouls_total","fouls_in_middle_third_%"),
            ("fouls_in_attacking_third","fouls_total","fouls_in_attacking_third_%"),
        ]

        for a, b, c in calculation_pairs:
            total_stats[c] = (total_stats[f'{a}'] / total_stats[f'{b}'])
            # replace inf
            total_stats[c] = total_stats[c].replace([np.inf, -np.inf], 0)
        total_stats = total_stats.fillna(0)

        ###  calcuate stats per match ###

        # merge standard stats with absolute values (result_df)
        absolute_column_values = [col for col in total_stats.columns if not col.endswith("_%") ]
        df_stats_per_game = pd.merge(left=self.standard_stats, right=total_stats[absolute_column_values],on="player_id",how="left")
        df_stats_per_game = df_stats_per_game.fillna(0)

        # calcuate stats per match and add to result_df
        for col in df_stats_per_game.drop(["player", "player_id","full_match_equivalents"], axis=1).columns:
            col_name = f"{col}_per_match"
            df_stats_per_game[col_name] = (df_stats_per_game[col] / 90).round(3)

        # keep only per match stats
        column_per_match = [col for col in df_stats_per_game.columns if col.endswith("_per_match") or col=="player" or col == "player_id"]
        df_stats_per_game = df_stats_per_game[column_per_match]

        # merge: abosulte, relative, per game values
        total_stats = pd.merge(left=total_stats, right=df_stats_per_game, on="player_id", how="right")
        total_stats = total_stats.fillna(0)


        return total_stats
    
    def store_data(self, df):
        folder_dir = f"{PROJECT_ROOT_DIR}/data/new_approach"
        os.makedirs(folder_dir, exist_ok=True)
        file_path = f"{folder_dir}/{self.dim}_ex.csv"
        df.to_csv(file_path,index=False)

    def run(self):
        self.filter_by_dimension()
        self.convert_columns()
        result_df = self.extract()
        self.store_data(result_df)