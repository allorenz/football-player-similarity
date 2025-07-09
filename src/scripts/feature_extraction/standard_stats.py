"""
TODO: Make this cleaner and place it somewhere!!!

"""

import pandas as pd
import numpy as np
import json 
import sys
import os
from tqdm import tqdm
from datetime import datetime
from statsbombpy import sb
from pathlib import Path
from collections import Counter

# Run the notebook from inside the notebooks folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), '..')))

from dataloader import Dataloader

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
with open(f"{PROJECT_ROOT_DIR}/config/position_mapping.json","r") as f:
    position_mapping = json.load(f)

def log_step(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

def get_most_frequent_pos(input_list):
    """
    Return most occupied position and maps it to global position (i.e. GK,DF,MF,FW)
    """
    input_list = extract_positions(input_list, unique=False)
    if not input_list:
        return None
    
    counter = Counter(input_list)
    max_count = max(counter.values())
    most_frequent_pos = list({item for item, count in counter.items() if count == max_count})
    
    # Map positions to roles, then remove duplicates using a set
    mapped_positions = {position_mapping[pos] for pos in most_frequent_pos}
    
    # Join the unique mapped positions into a single string
    return ", ".join(mapped_positions)

def extract_positions(positions,unique=False):
    # Flatten the array of lists into a single list
    flattened_positions = np.concatenate(positions.values).tolist()
    # Check if the entire flattened list is empty
    if all(len(pos_list) == 0 for pos_list in flattened_positions):
        return []
    # Extract positions from the dictionaries

    if unique:
        positions = [pos["position"] for pos in flattened_positions if "position" in pos]
        return list(set(positions))
        
    else:
        return [pos["position"] for pos in flattened_positions if "position" in pos]
    
def extract_global_position(positions):
    positions_list = get_most_frequent_pos(positions)
    if len(positions_list)==1:
        return position_mapping[positions_list[0]]
    elif len(positions_list)==0:
        return positions
    else:
        try:
            output = position_mapping[positions_list[0]]
            for pos in positions_list[1:]:
                output += f",{position_mapping[pos]}"
            return output
        except:
            print(positions_list)

def get_positions_played(df):
    positions_played_df = df.groupby("player_id").agg(
        position=("positions", lambda x: get_most_frequent_pos(x)),
        positions_played=("positions", lambda x: extract_positions(x)),
        unique_positions_played=("positions", lambda x: extract_positions(x,unique=True))
        #global_position=("positions", lambda x: extract_global_position(x)) 
    )
    return positions_played_df

def get_minutes_played(df):
    df_with_flags = df.copy()

    all_players = set()
    match_duration = df_with_flags["minute"].max()

    for player in df_with_flags.loc[df_with_flags["player"].notna(),"player"].values:
        all_players.add(player)

    for player in df_with_flags.loc[df_with_flags["substitution_replacement"].notna(), "substitution_replacement"].values:
        all_players.add(player)

    # flags 
    df_with_flags["is_substituted"] = df_with_flags["substitution_outcome"].notna()
    df_with_flags["match_duration"] = match_duration
    df_with_flags["minutes_played_subbed_in"] = match_duration - df_with_flags["minute"]

    # filter
    df_player_subbed_out = df_with_flags.loc[df_with_flags["is_substituted"]==True,["player", "player_id", "minute"]]
    df_player_subbed_in = df_with_flags.loc[df_with_flags["is_substituted"]==True, ["substitution_replacement", "player_id", "minutes_played_subbed_in"]]

    # rename
    df_player_subbed_in = df_player_subbed_in.rename({"substitution_replacement":"player",
                                                      "minutes_played_subbed_in" :"minutes_played"}, axis=1)
    df_player_subbed_out = df_player_subbed_out.rename({"minute":"minutes_played"}, axis=1)

    # add subbed_in subbed_of mask
    df_player_subbed_in["subbed_in"] = 1
    df_player_subbed_out["subbed_out"] = 1

    df_subbed_player = pd.concat([df_player_subbed_in, df_player_subbed_out],axis=0)

    # add remaining players that were not subbed off
    players_not_subbed_off = [player for player in all_players if player not in df_subbed_player["player"].values]
    # get player_id for players not subbed off
    player_id_map = df_with_flags.set_index("player")["player_id"].to_dict()
    df_player_not_subbed_off = pd.DataFrame({
        "player": players_not_subbed_off,
        "player_id": [player_id_map.get(player, np.nan) for player in players_not_subbed_off],
        "minutes_played" : 90
    })

    df_result = pd.concat([df_subbed_player, df_player_not_subbed_off],axis=0).reset_index(drop=True)
    df_result["match_played"] = 1

    return df_result

def analyze_standard_stats(df):
    match_ids = df["match_id"].unique()
    columns = ["player", "player_id", "match_id", "minute", "substitution_replacement", "substitution_outcome"]
    standard_stats = pd.DataFrame()
    
    log_step("Calculating match_played and minutes_played")
    concated_matches_df = pd.DataFrame()

    for match_id in tqdm(match_ids, desc="Concatenating matches", unit="match"):
        current = get_minutes_played(df.loc[df["match_id"]==match_id,columns])
        concated_matches_df = pd.concat([concated_matches_df, current])
    
    df_match_minutes_played = concated_matches_df.groupby("player_id").agg(
            match_played=("match_played","sum"),
            minutes_played=("minutes_played","sum"),
            subbed_in=("subbed_in","sum"),
            subbed_out=("subbed_out","sum")
        )

    
    log_step("Retrieve Lineups from API to map team, country, and positions_played")
    
    df_team_country_concated = pd.DataFrame()
    # load and stack matches 
    for match_id in tqdm(match_ids, desc="Retrieving Lineups per game from API", unit="match"):
        lineups = sb.lineups(match_id=match_id) # ["Hertha Berlin"]
        
        for team in lineups.keys():
            player_information = lineups[team].loc[:, ["player_name","player_id","country","positions", "jersey_number"]].copy()
            player_information["team"] = team
            df_team_country_concated = pd.concat([df_team_country_concated, player_information], axis=0)
    
            
    df_team_country = df_team_country_concated.drop_duplicates(subset=['player_id'])
    df_team_country = df_team_country.rename({"player_name" : "player"},axis=1)

    log_step("Process positions")
    positions_played_df = get_positions_played(df_team_country_concated)
    positions_played_df = positions_played_df.reset_index(drop=False).rename({"player_name" : "player"},axis=1)

    log_step("Merge to final dataframe")
    # merge player position to standard stats
    standard_stats = pd.merge(left=df_match_minutes_played, right=positions_played_df, on="player_id", how="right")
    standard_stats = standard_stats.fillna(0) # player with nan did not play
    
    # merge country and team to standard stats
    standard_stats = pd.merge(left=standard_stats, right=df_team_country,on="player_id", how="left")
    
    # reorder columns and keep only relevant
    standard_stats = standard_stats[["player", "player_id","country","team","position","match_played","minutes_played","subbed_in","subbed_out","unique_positions_played","positions_played"]]
    
    return standard_stats#, df_team_country_concated
  
"""
Automation
"""

if __name__ == "__main__":
    dataloader = Dataloader(file_path="../../../data/new_approach/new_all_leagues.parquet") # /data/new_approach/all_leagues.parquet
    dataloader.load_data()
    df = dataloader.get_dimension("standard_stats", row_filter=False)
    print(df.shape)
    result_df= analyze_standard_stats(df)
    result_df["full_match_equivalents"] = result_df["minutes_played"] / 90

    # clean column position
    result_df["position"] = result_df["position"].replace(0, "nan")
    result_df['position'] = result_df['position'].str.replace("Forward, Defender", "Forward", case=False, regex=False)
    result_df['position'] = result_df['position'].str.replace("Forward, Midfielder, Defender", "Forward", case=False, regex=False)
    result_df['position'] = result_df['position'].str.replace("Midfielder, Defender", "Midfielder", case=False, regex=False)
    result_df['position'] = result_df['position'].str.replace("Midfielder, Forward", "Midfielder", case=False, regex=False)
    result_df['position'] = result_df['position'].str.replace("Forward, Midfielder", "Forward", case=False, regex=False)
    result_df['position'] = result_df['position'].str.replace("Defender, Forward", "Defender", case=False, regex=False)
    result_df['position'] = result_df['position'].str.replace("Defender, Midfielder", "Defender", case=False, regex=False)
    
    # store standard stats
    folder_dir = f"{PROJECT_ROOT_DIR}/data/new_approach"
    os.makedirs(folder_dir, exist_ok=True)
    file_path = f"{folder_dir}/standard_stats_all.csv"
    result_df.to_csv(file_path,index=False)
    print(result_df)