import sys
import os
import json
import pandas as pd
import numpy as np
from ast import literal_eval
from pathlib import Path
from collections import defaultdict
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

def is_ex_in_goal_area(row):
    pitch_width = 120
    x = row["x"]
    y = row["y"]

    x_axis = ((pitch_width - 6 <= x) and (x <= pitch_width))
    y_axis = ((40 - 10) <= y and y <= (40 + 10))
    return x_axis and y_axis

def is_ex_in_penalty_area(row):
    #print("Executing is_in_penalty_area method")
    pitch_width = 120

    x = row["x"]
    y = row["y"]

    if is_ex_in_goal_area(row):
        return False
    else:
        x_axis = ((102 <= x) and (x <= pitch_width))  
        y_axis = ((40 - 22) <= y and y <= (40 + 22))
        return x_axis and y_axis

def is_ex_in_attacking_third(row):
    pitch_width = 120
    x = row["x"]

    if is_ex_in_goal_area(row) or is_ex_in_penalty_area(row):
        return False
    else:
        attacking_third_start = 2 * (pitch_width / 3)
        return (attacking_third_start <= x <= pitch_width)  # Stop before penalty area

def is_in_middle_third(row):
    #print("Executing is_in_middle_third method")
    pitch_width = 120
    x = row["x"]

    return (1 * (pitch_width / 3)) <= x <= (2 * (pitch_width / 3))

def calculate_distance(row):
    return np.linalg.norm(np.array([row["x"], row["y"]]) - np.array([row["x_end_shot"], row["y_end_shot"]])) 

def count_important_goals(df):
    results = defaultdict(list)

    for match_id in df["match_id"].unique():
        match = df.loc[(df["match_id"] == match_id)  & (df["shot_outcome"] == "Goal"), ["match_id", "player","minute","possession_team"]]
        match = match.sort_values("minute", ascending=True)
        team_names = match['possession_team'].unique()
        team_scores = {team: 0 for team in team_names}
        is_equalized=False

        for i in range(match.shape[0]):
            team_scores[match.iloc[i]["possession_team"]]  += 1
            scores = list(team_scores.values())
            
            # first goal
            if (sum(scores) == 1):
                results[match.iloc[i]["player"]].append("first_goal")
            # equalizer
            elif (len(scores) > 1) and (scores[0] == scores[1]):
                is_equalized=True
                results[match.iloc[i]["player"]].append("equalizer")
            # taking the lead
            elif (len(scores) > 1) and (scores[0]>scores[1] or scores[1]>scores[0]) and is_equalized:
                is_equalized=False
                results[match.iloc[i]["player"]].append("leading_goal")

    df_results = pd.DataFrame([
        {"player": player, "event": event}
        for player, events in results.items()
        for event in events
    ])

    df_event_counts = df_results.groupby(["player", "event"]).size().unstack(fill_value=0).reset_index()
    return df_event_counts

class ShootingFeatureExtractor(BaseDimensionFeatureExtractor):

    def __init__(self, df: pd.DataFrame, standard_stats: pd.DataFrame, league:str):
        self.df = df
        self.standard_stats = standard_stats
        self.league = league
        self.dim = "shooting"

        with open(f"{PROJECT_ROOT_DIR}/config/columns_config.json", 'r') as f:
            self.columns = json.load(f)

    def filter_by_dimension(self):
        self.df = self.df.loc[self.df["type"].isin(self.columns[self.dim]["row_filter"]), self.columns[self.dim]["columns"]]

    def convert_columns(self):
        self.df["location"] = self.df["location"].apply(convert_to_list) #shot_end_location
        self.df["shot_end_location"] = self.df["shot_end_location"].apply(convert_to_list) #shot_end_location
        self.df[["x", "y"]] = self.df["location"].apply(pd.Series)
        self.df[["x_end_shot", "y_end_shot","z_end_shot"]] = self.df["shot_end_location"].apply(pd.Series)
        self.df["under_pressure"] = self.df["under_pressure"].fillna(False)

    def extract(self):
        """
        This function Pre-compute all conditions for Vectorize operations.
        Returns results grouped by player and under_pressure  
        """
        # tresholds
        long_goal_min_distance = 20
        short_goal_max_distance = 10
        treshold_xG = 0.3
        treshold_dist = 11
        threshold_last_minute_goal = 85

        # Pre-compute all conditions
        df_with_flags = self.df.copy()

        # action flags
        df_with_flags["is_goal"] = self.df["shot_outcome"] == "Goal"
        df_with_flags["is_shot"] = self.df["type"] == "Shot"
        df_with_flags["is_penalty"] = self.df["shot_type"] == "Penalty"
        df_with_flags["is_freekick"] = self.df["shot_type"] == "Free Kick"
        df_with_flags["is_left_foot"] = self.df["shot_body_part"] == "Left Foot"
        df_with_flags["is_right_foot"] = self.df["shot_body_part"] == "Right Foot"
        df_with_flags["is_head"] = self.df["shot_body_part"] == "Head"
        df_with_flags["is_other_part"] = self.df["shot_body_part"] == "Other"
        df_with_flags["is_first_touch"] = self.df["shot_first_time"] == True

        # location flags
        df_with_flags["is_in_goal_area"] = self.df[["x","y"]].apply(is_ex_in_goal_area, axis=1)
        df_with_flags["is_in_penalty_area"] = self.df[["x","y"]].apply(is_ex_in_penalty_area, axis=1)
        df_with_flags["is_in_attacking_third"] = self.df[["x","y"]].apply(is_ex_in_attacking_third, axis=1)
        df_with_flags["is_in_middle_third"] = (80 > self.df['x']) & (self.df['x'] > 40)
        df_with_flags["goal-shot_distance"] = self.df[["x","y","x_end_shot","y_end_shot"]].apply(calculate_distance, axis=1)
        df_with_flags["is_long_distance"] = (df_with_flags["goal-shot_distance"] >= long_goal_min_distance) 
        df_with_flags["is_short_distance"] = (df_with_flags["goal-shot_distance"] <= short_goal_max_distance) 
        df_with_flags["is_mid_distance"] = ~df_with_flags["is_short_distance"] & ~df_with_flags["is_long_distance"]

        # combinations
        df_with_flags["penalty_goal"] = (df_with_flags["is_penalty"]) & (df_with_flags["is_goal"])
        df_with_flags["shot_on_target"] = (self.df["shot_outcome"] != "Off T") & (self.df["shot_outcome"] != "Wayward") & (self.df["shot_outcome"] != "Saved Off T")
        df_with_flags["shot_on_target_left_foot"] = df_with_flags["is_left_foot"] & df_with_flags["shot_on_target"]
        df_with_flags["shot_on_target_right_foot"] = df_with_flags["is_right_foot"] & df_with_flags["shot_on_target"]
        df_with_flags["goal_left_foot"] = df_with_flags["is_left_foot"] & df_with_flags["is_goal"]
        df_with_flags["goal_right_foot"] = df_with_flags["is_right_foot"] & df_with_flags["is_goal"]
        df_with_flags["goal_with_head"] = df_with_flags["is_head"] & df_with_flags["is_goal"]
        df_with_flags["goal_with_other_part"] = df_with_flags["is_other_part"] & df_with_flags["is_goal"]
        df_with_flags["goal_with_first_touch"] = df_with_flags["is_first_touch"] & df_with_flags["is_goal"]
        df_with_flags["goal_in_goal_area"] = df_with_flags["is_in_goal_area"] & df_with_flags["is_goal"]
        df_with_flags["goal_in_penalty_area"] = df_with_flags["is_in_penalty_area"] & df_with_flags["is_goal"]
        df_with_flags["goal_in_attacking_third"] = df_with_flags["is_in_goal_area"] & df_with_flags["is_goal"]
        df_with_flags["goal_in_middle_third"] = df_with_flags["is_in_goal_area"] & df_with_flags["is_goal"]
        df_with_flags["goal_is_long_distance"] = df_with_flags["is_long_distance"] & df_with_flags["is_goal"]
        df_with_flags["goal_is_mid_distance"] = df_with_flags["is_mid_distance"] & df_with_flags["is_goal"]
        df_with_flags["goal_is_short_distance"] = df_with_flags["is_short_distance"] & df_with_flags["is_goal"]
        df_with_flags["shot_with_first_touch"] = df_with_flags["is_first_touch"] & df_with_flags["is_shot"]
        df_with_flags["shot_in_goal_area"] = df_with_flags["is_in_goal_area"] & df_with_flags["is_shot"]
        df_with_flags["shot_in_penalty_area"] = df_with_flags["is_in_penalty_area"] & df_with_flags["is_shot"]
        df_with_flags["shot_in_attacking_third"] = df_with_flags["is_in_goal_area"] & df_with_flags["is_shot"]
        df_with_flags["shot_in_middle_third"] = df_with_flags["is_in_goal_area"] & df_with_flags["is_shot"]
        df_with_flags["shot_is_long_distance"] = df_with_flags["is_long_distance"] & df_with_flags["is_shot"]
        df_with_flags["shot_is_mid_distance"] = df_with_flags["is_mid_distance"] & df_with_flags["is_shot"]
        df_with_flags["shot_is_short_distance"] = df_with_flags["is_short_distance"] & df_with_flags["is_shot"]
        df_with_flags["goal_normal"] = (self.df["shot_technique"] == "Normal") & (df_with_flags["is_goal"])
        df_with_flags["goal_volley"] = (self.df["shot_technique"] == "Volley") & (df_with_flags["is_goal"])
        df_with_flags["goal_half_volley"] = (self.df["shot_technique"] == "Half Volley") & (df_with_flags["is_goal"])
        df_with_flags["goal_lob"] = (self.df["shot_technique"] == "Lob") & (df_with_flags["is_goal"])
        df_with_flags["goal_backheel"] = (self.df["shot_technique"] == "Backheel") & (df_with_flags["is_goal"])
        df_with_flags["goal_overhead_kick"] = (self.df["shot_technique"] == "Overhead Kick") & (df_with_flags["is_goal"])
        df_with_flags["goal_diving_header"] =(self.df["shot_technique"] == "Diving Header") & (df_with_flags["is_goal"])
        df_with_flags["is_big_chance"] =(self.df["shot_statsbomb_xg"] >= treshold_xG) & (df_with_flags["goal-shot_distance"] <= treshold_dist)
        df_with_flags["is_big_chance_led_to_goal"] =(self.df["shot_statsbomb_xg"] >= treshold_xG) & (df_with_flags["goal-shot_distance"] <= treshold_dist) & (df_with_flags["is_goal"])
        df_with_flags["goal_inside_penalty_area"] = df_with_flags["is_in_goal_area"] & df_with_flags["is_in_penalty_area"] & df_with_flags["is_goal"]
        df_with_flags["goal_outside_penalty_area"] = ~df_with_flags["goal_inside_penalty_area"]
        df_with_flags["shot_inside_penalty_area"] = df_with_flags["is_in_goal_area"] & df_with_flags["is_in_penalty_area"] & df_with_flags["is_shot"]
        df_with_flags["shot_outside_penalty_area"] = ~df_with_flags["shot_inside_penalty_area"]
        df_with_flags["is_last_minute_goal"] = (self.df["minute"] >= threshold_last_minute_goal) & (df_with_flags["is_goal"])


        player_under_pressure_grouping = df_with_flags.groupby(['player',"under_pressure"]).agg(
            # goals 
            goals=('is_goal', "sum"),
            goals_penalty=("penalty_goal","sum"),
            goals_header=('goal_with_head', "sum"),
            goals_left_foot=('goal_left_foot', "sum"),
            goals_right_foot=('goal_right_foot', "sum"),
            goals_other_body_part=('goal_with_other_part', "sum"),
            goals_with_first_touch=('goal_with_first_touch', "sum"),
            goals_in_goal_area=("goal_in_goal_area", "sum"),
            goals_in_penalty_area=("goal_in_penalty_area", "sum"),
            goals_in_attacking_third=("goal_in_attacking_third", "sum"),
            goals_in_middle_third=("goal_in_middle_third", "sum"),
            goals_inside_penalty_area=("goal_inside_penalty_area","sum"),
            goals_outside_penalty_area=("goal_outside_penalty_area","sum"),
            goals_long_distance=("goal_is_long_distance","sum"),
            goals_mid_distance=("goal_is_mid_distance","sum"),
            goals_short_distance=("goal_is_short_distance","sum"),
            goals_avg_distance=("goal-shot_distance", lambda x: x[df_with_flags["is_goal"]==True].mean()),
            # shots
            shots_total=('player', "count"),
            shots_on_target=("shot_on_target","sum"),
            shots_from_freekick = ('is_freekick', "sum"),
            shots_from_penalty = ('is_penalty', "sum"),
            shots_on_target_left_foot = ("shot_on_target_left_foot", "sum"),
            shots_on_target_right_foot = ("shot_on_target_right_foot", "sum"),
            shots_with_first_touch=('shot_with_first_touch', "sum"),
            shots_in_goal_area=("shot_in_goal_area", "sum"),
            shots_in_penalty_area=("shot_in_penalty_area", "sum"),
            shots_in_attacking_third=("shot_in_attacking_third", "sum"),
            shots_in_middle_third=("shot_in_middle_third", "sum"),
            shots_inside_penalty_area=("shot_inside_penalty_area","sum"),
            shots_outside_penalty_area=("shot_outside_penalty_area","sum"),
            shots_long_distance=("shot_is_long_distance","sum"),
            shots_mid_distance=("shot_is_mid_distance","sum"),
            shots_short_distance=("shot_is_short_distance","sum"),
            shots_avg_distance=("goal-shot_distance", lambda x: x[df_with_flags["is_shot"]==True].mean()),
            # goal techniques
            goal_normal=("goal_normal", "sum"),
            goal_volley=("goal_volley", "sum"),
            goal_half_volley=("goal_half_volley", "sum"),
            goal_lob=("goal_lob", "sum"),
            goal_backheel=("goal_backheel", "sum"),
            goal_overhead_kick=("goal_overhead_kick","sum"),
            goal_diving_header=("goal_diving_header", "sum"),
            # big chances and important goals
            xG=("shot_statsbomb_xg", "sum"),
            big_chances_total=("is_big_chance","sum"),
            big_chances_led_to_goal=("is_big_chance_led_to_goal","sum"),
            last_minute_goals=("is_last_minute_goal","sum")

        )

        total_stats = player_under_pressure_grouping.groupby('player').sum()
        
        player_under_pressure_grouping = player_under_pressure_grouping.add_prefix('up_')
        player_under_pressure_grouping = player_under_pressure_grouping.reset_index()
        player_under_pressure_grouping = player_under_pressure_grouping[player_under_pressure_grouping["under_pressure"] == True]
        player_under_pressure_grouping = player_under_pressure_grouping.drop("under_pressure", axis=1)

        important_goal_counts = count_important_goals(self.df)
        important_goal_counts["goals_important"] = important_goal_counts["first_goal"] + important_goal_counts["equalizer"] + important_goal_counts["leading_goal"]

        player_stats = pd.merge(left=total_stats, right=player_under_pressure_grouping, on="player",how="left")
        player_stats = pd.merge(left=player_stats, right=important_goal_counts, on="player",how="left")

        ### calculate relative values ###

        calculation_pairs = [
            ('shots_on_target', 'shots_total', "shots_on_target_%"),
            ("goals", "shots_total", "shot_conversion_%"),
            ("goals", "shots_on_target", "shot_conversion_on_target_%"),
            ("big_chances_led_to_goal","big_chances_total","big_chances_conversion_%"),
            ("up_goals", "up_shots_total", "up_shot_conversion_%"),
            ("up_shots_on_target", "up_shots_total", "up_shots_on_target_%"),
            ("up_big_chances_led_to_goal","up_big_chances_total","up_big_chances_conversion_%"),
            ("shots_inside_penalty_area", "shots_total", "shots_inside_the_box_%"),
            ("shots_outside_penalty_area", "shots_total", "shots_outside_the_box_%"),
            ("goals_right_foot", "goals", "goals_right_foot_%"),
            ("goals_left_foot", "goals", "goals_left_foot_%"),
            ("goals_with_first_touch", "goals", "goals_with_first_touch_%"),
            ("big_chances_led_to_goal", "big_chances_total","big_chance_conversion_%"),
            ("goals_outside_penalty_area", "goals","goals_outside_penalty_area_%"),
            ("goals_inside_penalty_area", "goals","goals_inside_penalty_area_%"),
            ("goals_header", "goals", "goals_header_%")
        ]
        
        for a, b, c in calculation_pairs:
            player_stats[c] = (player_stats[f'{a}'] / player_stats[f'{b}']).round(3)
            # replace inf
            player_stats[c] = player_stats[c].replace([np.inf, -np.inf], 0)
        player_stats = player_stats.fillna(0)

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