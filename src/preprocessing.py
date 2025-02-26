import pandas as pd
import numpy as np
import time
from statsbombpy import sb
from ast import literal_eval
from dataloader import load_data
from datetime import datetime


def log_step(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")


def convert_to_list(input_data):
    try:
        return literal_eval(input_data)
    except (ValueError, SyntaxError):
        print(f"Error: The string {input_data} could not be converted to a list.")
        return None

def is_in_goal_area(location):
    #print("Executing is_in_goal_area method")
    pitch_width = 120
    if isinstance(location, str):
        location = convert_to_list(location)

    x = location[0]
    y = location[1]

    x_axis = ((pitch_width - 6 <= x) and (x <= pitch_width))
    y_axis = ((40 - 10) <= y and y <= (40 + 10))
    return x_axis and y_axis

def is_in_penalty_area(location):
    #print("Executing is_in_penalty_area method")
    pitch_width = 120

    if isinstance(location, str):
        location = convert_to_list(location)

    x = location[0]
    y = location[1]

    if is_in_goal_area(location):
        return False
    else:
        # Check if in larger penalty area but NOT in goal area
        x_axis = ((102 <= x) and (x <= pitch_width))  # Stop before goal area
        y_axis = ((40 - 22) <= y and y <= (40 + 22))
        return x_axis and y_axis

def is_in_attacking_third(location):
    #print("Executing is_in_attacking_third method")
    pitch_width = 120

    if isinstance(location, str):
        location = convert_to_list(location)

    x = location[0]
    y = location[1]

    if is_in_goal_area(location) or is_in_penalty_area(location):
        return False
    else:
        attacking_third_start = 2 * (pitch_width / 3)
        return (attacking_third_start <= x <= pitch_width)  # Stop before penalty area

def is_in_middle_third(location):
    #print("Executing is_in_middle_third method")
    pitch_width = 120

    if isinstance(location, str):
        location = convert_to_list(location)

    x = location[0]
    y = location[1]

    return (1 * (pitch_width / 3)) <= x <= (2 * (pitch_width / 3))

def is_long_distance_shot(start_location, end_location):
    #print("Executing is_long_distance_shot method")

    if isinstance(start_location, str):
        start_location = convert_to_list(start_location)

    if isinstance(end_location, str):
        end_location = convert_to_list(end_location)

    x_start = start_location[0]
    y_start = start_location[1]
    x_end = end_location[0]
    y_end = end_location[1]

    start = np.array([x_start, y_start])
    end = np.array([x_end, y_end])
    distance = np.linalg.norm(start - end)

    return (distance >= 20)

def is_mid_distance_shot(start_location, end_location):
    #print("Executing is_mid_distance_shot method")
    if isinstance(start_location, str):
        start_location = convert_to_list(start_location)

    if isinstance(end_location, str):
        end_location = convert_to_list(end_location)

    x_start = start_location[0]
    y_start = start_location[1]
    x_end = end_location[0]
    y_end = end_location[1]

    start = np.array([x_start, y_start])
    end = np.array([x_end, y_end])
    distance = np.linalg.norm(start - end)
    return (10 < distance < 20)

def is_short_distance_shot(start_location, end_location):
    #print("Executing is_short_distance_shot method")
    if isinstance(start_location, str):
        start_location = convert_to_list(start_location)

    if isinstance(end_location, str):
        end_location = convert_to_list(end_location)

    x_start = start_location[0]
    y_start = start_location[1]
    x_end = end_location[0]
    y_end = end_location[1]

    start = np.array([x_start, y_start])
    end = np.array([x_end, y_end])
    distance = np.linalg.norm(start - end)
    return (distance <= 10)

def get_shot_distance(start_location, end_location):
    #print("Executing is_short_distance_shot method")
    if isinstance(start_location, str):
        start_location = convert_to_list(start_location)

    if isinstance(end_location, str):
        end_location = convert_to_list(end_location)

    x_start = start_location[0]
    y_start = start_location[1]
    x_end = end_location[0]
    y_end = end_location[1]

    start = np.array([x_start, y_start])
    end = np.array([x_end, y_end])

    return np.linalg.norm(start - end)

def is_big_chance(xG, start_location, end_location):
    if isinstance(start_location, str):
        start_location = convert_to_list(start_location)

    if isinstance(end_location, str):
        end_location = convert_to_list(end_location)

    x_start = start_location[0]
    y_start = start_location[1]
    x_end = end_location[0]
    y_end = end_location[1]

    start = np.array([x_start, y_start])
    end = np.array([x_end, y_end])

    return (xG >= 0.5 and np.linalg.norm(start - end) <= 12)


def analyze_player_shots(df):
    """
    Analyze player shot statistics
    Returns:
    pandas.DataFrame: Player statistics for both under pressure and total shots
    """
    log_step("Change data types of certain columns")
    df['under_pressure'] = df['under_pressure'].fillna(False)

    log_step("Apply data aggregation")
    start = time.time()
    player_under_pressure_grouping = df[df['type'] == 'Shot'].groupby(['player',"under_pressure"]).agg(
        goals=('shot_outcome', lambda x: (x == 'Goal').sum()),
        goals_penalty=("shot_outcome", lambda x: ((x == 'Goal') & (df["shot_type"] == "Penalty")).sum()),
        shots_total=('shot_outcome', "count"), # optionally: sum
        shots_on_target=('shot_outcome', lambda x: ((x != 'Off T') | (x != 'Wayward') | (x != 'Saved Off T')).sum()),
        shots_from_freekick = ('shot_type', lambda x: (x == "Free Kick").sum()),
        shots_from_penalty = ('shot_type', lambda x: (x == "Penalty").sum()), 
        shots_on_target_left_foot = ("shot_body_part", lambda x: (x == "Left Foot").sum()),
        shots_on_target_right_foot = ("shot_body_part", lambda x: (x == "Right Foot").sum()),
        goals_from_left_foot=('shot_body_part', lambda x: ((df['shot_outcome'] == 'Goal') & (x == 'Left Foot')).sum()),
        goals_from_right_foot=('shot_body_part', lambda x: ((df['shot_outcome'] == 'Goal') & (x == 'Right Foot')).sum()),
        goals_from_head=('shot_body_part', lambda x: ((df['shot_outcome'] == 'Goal') & (x == 'Head')).sum()),
        goals_after_first_touch=('shot_first_time', lambda x: ((df['shot_outcome'] == 'Goal') & (x == True)).sum()),
        goals_in_penalty_area=("location", lambda x: x[df.loc[x.index, "shot_outcome"] == "Goal"].apply(is_in_penalty_area).sum()),
        goals_in_goal_area=("location", lambda x: x[df.loc[x.index, "shot_outcome"] == "Goal"].apply(is_in_goal_area).sum()),
        goals_in_attacking_third=("location", lambda x: x[df.loc[x.index, "shot_outcome"] == "Goal"].apply(is_in_attacking_third).sum()),
        goals_in_middle_third=("location", lambda x: x[df.loc[x.index, "shot_outcome"] == "Goal"].apply(is_in_middle_third).sum()),
        goals_long_distance=(("location"), lambda x: sum(
                                is_long_distance_shot(start, end) 
                                for start, end, idx in zip(x, df.loc[x.index, "shot_end_location"], x.index)
                                if df.loc[idx, "shot_outcome"] == "Goal"
                            )),
        goals_mid_distance=(("location"), lambda x: sum(
                                is_mid_distance_shot(start, end) 
                                for start, end, idx in zip(x, df.loc[x.index, "shot_end_location"], x.index)
                                if df.loc[idx, "shot_outcome"] == "Goal"
                            )),
        goals_short_distance=(("location"), lambda x: sum(
                                is_short_distance_shot(start, end) 
                                for start, end, idx in zip(x, df.loc[x.index, "shot_end_location"], x.index)
                                if df.loc[idx, "shot_outcome"] == "Goal"
                            )),
        goals_avg_distance=(("location"), lambda x: 
                                np.average([get_shot_distance(start, end) for start, end, idx in zip(x, df.loc[x.index, "shot_end_location"], x.index) if df.loc[idx, "shot_outcome"] == "Goal"]).round(2) 
                            ),  
        shots_in_penalty_area=("location", lambda x: x.apply(is_in_penalty_area).sum()),
        shots_in_goal_area=("location", lambda x: x.apply(is_in_goal_area).sum()),
        shots_in_attacking_third=("location", lambda x: x.apply(is_in_attacking_third).sum()),
        shots_in_middle_third=("location", lambda x: x.apply(is_in_middle_third).sum()),
        shots_long_distance=(("location"), lambda x: sum(
                                is_long_distance_shot(start, end) 
                                for start, end in zip(x, df.loc[x.index, "shot_end_location"])  # Use corresponding 'shot_end_location' for each 'location'
                            )),
        shots_mid_distance=(("location"), lambda x: sum(
                                is_mid_distance_shot(start, end) 
                                for start, end in zip(x, df.loc[x.index, "shot_end_location"])  # Use corresponding 'shot_end_location' for each 'location'
                            )),
        shots_short_distance=(("location"), lambda x: sum(
                                is_short_distance_shot(start, end) 
                                for start, end in zip(x, df.loc[x.index, "shot_end_location"])  # Use corresponding 'shot_end_location' for each 'location'
                            )),
        shots_avg_distance=(("location"), lambda x: 
                                np.average([get_shot_distance(start, end) for start, end in zip(x, df.loc[x.index, "shot_end_location"])]).round(2)   # Use corresponding 'shot_end_location' for each 'location'
                            ),                     
        shots_volley=("shot_technique", lambda x: (x == "Volley").sum()),
        shots_half_volley=("shot_technique", lambda x: (x == "Half Volley").sum()),
        shots_normal=("shot_technique", lambda x: (x == "Normal").sum()),
        shots_lob=("shot_technique", lambda x: (x == "Lob").sum()),
        shots_backheel=("shot_technique", lambda x: (x == "Backheel").sum()),
        shots_overhead_kick=("shot_technique", lambda x: (x == "Overhead Kick").sum()),
        shots_diving_header=("shot_technique", lambda x: (x == "Diving Header").sum()),
        shots_after_aerial_duel_won=("shot_aerial_won", lambda x: (x == True).sum()),
        shots_after_dribbling=("shot_follows_dribble", lambda x: (x == True).sum()),
        xG=("shot_statsbomb_xg", lambda x: x.sum()),
        big_chances=("shot_statsbomb_xg", lambda x: sum(
            is_big_chance(xG, start, end) for xG, start, end in zip(x, df.loc[x.index, "location"], df.loc[x.index, "shot_end_location"])
        )),
        big_chances_led_to_goal=("shot_statsbomb_xg", lambda x: sum(
            is_big_chance(xG, start, end) for xG, start, end, idx in zip(x, df.loc[x.index, "location"], df.loc[x.index, "shot_end_location"], x.index)
            if df.loc[idx, "shot_outcome"] == "Goal"
        ))
    )
    end = time.time()
    log_step(f"Aggregation Time: {end - start:.2f}s")

    # add more stats
    log_step("Add additional stats")
    player_under_pressure_grouping["shots_inside_the_box"] = player_under_pressure_grouping["shots_in_penalty_area"] + player_under_pressure_grouping["shots_in_goal_area"]
    player_under_pressure_grouping["shots_outside_the_box"] = player_under_pressure_grouping["shots_total"] - player_under_pressure_grouping["shots_inside_the_box"]
    player_under_pressure_grouping["goals_inside_the_box"] = player_under_pressure_grouping["goals_in_penalty_area"] + player_under_pressure_grouping["goals_in_goal_area"]
    player_under_pressure_grouping["goals_outside_the_box"] = player_under_pressure_grouping["goals"] - player_under_pressure_grouping["goals_inside_the_box"]

    # get total numbers
    log_step("Aggregate total steps")
    total_stats = player_under_pressure_grouping.groupby('player').sum()

    # prepare under_pressure stats
    player_under_pressure_grouping = player_under_pressure_grouping.add_prefix('up_')
    player_under_pressure_grouping = player_under_pressure_grouping.reset_index()
    player_under_pressure_grouping = player_under_pressure_grouping[player_under_pressure_grouping["under_pressure"] == True]
    player_under_pressure_grouping = player_under_pressure_grouping.drop("under_pressure", axis=1)

    log_step("Merge total and under_pressure stats")
    player_stats = pd.merge(left=total_stats, right=player_under_pressure_grouping, on="player")

    log_step("Calculate proportional stats")
    # calculate % stats
    calculation_pairs = [
        ('shots_on_target', 'shots_total', "shots_on_target_%"),
        ("goals", "shots_total", "shot_conversion_%"),
        ("goals", "shots_on_target", "shot_conversion_on_target_%"),
        ("up_goals", "up_shots_total", "up_shot_conversion_%"),
        ("up_shots_on_target", "up_shots_total", "up_shots_on_target_%"),
        ("shots_inside_the_box", "shots_total", "shots_inside_the_box_%"),
        ("shots_outside_the_box", "shots_total", "shots_outside_the_box_%"),
        ("goals_from_right_foot", "goals", "goals_from_right_foot_%"),
        ("goals_from_left_foot", "goals", "goals_from_left_foot_%"),
        ("goals_after_first_touch", "goals", "goals_after_first_touch_%"),
        ("big_chances_led_to_goal", "big_chances","big_chance_conversion_%"),
        ("goals_outside_the_box", "goals","goals_outside_the_box_%"),
        ("goals_inside_the_box", "goals","goals_inside_the_box_%"),
        ("goals_from_head", "goals", "goals_from_head_%")
    ]
    for a, b, c in calculation_pairs:
        player_stats[c] = (player_stats[f'{a}'] / player_stats[f'{b}']).round(2)

    columns_to_keep = ["player", "goals", "goals_penalty", "shots_total",'shots_on_target',"shots_avg_distance", 'shots_in_penalty_area',
                    'shots_in_goal_area', 'shots_in_attacking_third', 'shots_long_distance', 'shots_mid_distance',
                    'shots_short_distance','shots_volley',"goals_avg_distance", 
                    'shots_half_volley', 'shots_normal',"goals_from_head" ,"goals_from_head_%",'shots_on_target_%',
                    'shot_conversion_%', 'shot_conversion_on_target_%',
                    'up_shot_conversion_%', 'up_shots_on_target_%',
                    'shots_inside_the_box_%', 'shots_outside_the_box_%',
                    'goals_from_right_foot_%', 'goals_from_left_foot_%',
                    'goals_after_first_touch_%',"big_chance_conversion_%", "goals_outside_the_box_%","goals_inside_the_box_%"
        ]

    return player_stats[columns_to_keep]


def main():
    log_step("Loading Data")
    all_columns = ["player", "position", "type", "location", "minute", "shot_aerial_won", "play_pattern", "shot_body_part", "shot_end_location" ,
           "shot_first_time","shot_outcome","shot_statsbomb_xg", "shot_technique", "shot_type", "shot_follows_dribble", 
           "under_pressure"]
    df = load_data(frac=1)
    df = df.loc[df["type"] == "Shot", all_columns]
    log_step(f"Data shape: {df.shape}")

    log_step("Starting analysis")
    df = analyze_player_shots(df)
    print(df.shape)


if __name__ == "__main__":
    main()