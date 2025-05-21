"""
Run this script to download raw event data for entire raw event data for top 5 leagues + ucl for the season of 2015/16.
Class Dataloader can be used to access for a dimension (e.g passing) rows and columns for feature engeneering.
"""

import os
import json
import pandas as pd
from statsbombpy import sb
from tqdm import tqdm
from pathlib import Path
from datetime import datetime


COMPETITION_ID = 9
SEASON_ID = 27
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent

with open(f"{PROJECT_ROOT_DIR}/config/competition_config.json", "r") as f:
    league_mapping = json.load(f)

def log_step(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")


class Dataloader:
    def __init__(self, league):
        self.league = league
        # set player dimensions
        self.columns = self._load_column_configs()
        self.dtypes = self._load_dtype_configs()
        # load data
        self.df = pd.DataFrame()

    def load_data(self):
        """
        Loads raw event data into memory
        """
        file_name = f"{self.league}.csv"
        if(os.path.exists((PROJECT_ROOT_DIR / "data" / "event_data" /file_name))):
            log_step("Loading data form local file system")
            file_path = f"{PROJECT_ROOT_DIR}/data/event_data/{file_name}" # PROJECT_ROOT_DIR / "data"
            self.df = pd.read_csv(file_path,dtype=self.dtypes)
            return self.df
        else:
            log_step("File does not exist, downloading data...")
            self.fetch_data()
            file_path = f"{PROJECT_ROOT_DIR}/data/event_data/{file_name}"
            self.df = pd.read_csv(file_path,dtype=self.dtypes)
            return self.df
            

    def fetch_data(self):
        file_name = f"{self.league}.csv"
        if(os.path.exists((PROJECT_ROOT_DIR / "data" / "event_data" /file_name))):
            print(f"{self} is already fetched")
        else:
            df = pd.DataFrame()
            df_matches = sb.matches(
                competition_id=league_mapping[self.league]["competition_id"], 
                season_id=league_mapping[self.league]["season_id"]
            )
            match_ids = df_matches["match_id"].to_list()

            for match_id in tqdm(match_ids, desc=f"Concatenating matches of {self.league}", unit="match"):
                events = sb.events(match_id=match_id)
                df = pd.concat([df, events], ignore_index=True)

            log_step(f"Data concatenated, resulting shape: {df.shape}")
            os.makedirs(f"{PROJECT_ROOT_DIR}/data/event_data/", exist_ok=True)
            file_path = f"{PROJECT_ROOT_DIR}/data/event_data/{self.league}.csv"
            log_step(f"Writing to {file_path}")
            df.to_csv(file_path, index=False)

    def _load_column_configs(self):
        """Load column configuration from an external JSON file."""
        with open(f"{PROJECT_ROOT_DIR}/config/columns_config.json", 'r') as f:
            return json.load(f)
        
    def _load_dtype_configs(self):
        """Load column configuration from an external JSON file."""
        with open(f"{PROJECT_ROOT_DIR}/config/dtypes_config.json", 'r') as f:
            return json.load(f)
    
    def get_dimension(self,dimension, row_filter=True):
        """
        dimensions: [defending, possession, passing, shooting, goal_keeping, standard_stats]
        """
        if row_filter:
            return self.df.loc[self.df["type"].isin(self.columns[dimension]["row_filter"]), self.columns[dimension]["columns"]].copy()
        return self.df[self.columns[dimension]["columns"]].copy()
    
    def get_data(self):
        """
        Returns the whole raw event data
        """
        return self.df.copy()
    
    def __repr__(self):
        return f"dataloader_{self.league}"

    
if __name__=="__main__":
    # download data for all leagues
    # for league in league_mapping.keys():
    #     dataloader = Dataloader(league)
    #     dataloader.fetch_data()

    # access raw event data given a league
    dataloader = Dataloader("bundesliga")
    dataloader.load_data()
    dimension = dataloader.get_dimension("standard_stats")