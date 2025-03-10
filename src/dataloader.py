import os
import pandas as pd
from statsbombpy import sb
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import json


COMPETITION_ID = 9
SEASON_ID = 27
PROJECT_ROOT_DIR = Path(__file__).parent.parent


"""
Old function:

def load_data(frac=0.001):
    if(os.path.exists("./data/bundesliga.csv")):
        file_path = "./data/bundesliga.csv"
    elif (os.path.exists("../data/bundesliga.csv")):
        file_path = "../data/bundesliga.csv"
    
    else:
        print("File does not exist, downloading data...")
        df = pd.DataFrame()
        df_matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
        match_ids = df_matches["match_id"].to_list()

        for match_id in tqdm(match_ids, desc="Concatenating matches", unit="match"):
            events = sb.events(match_id=match_id)
            df = pd.concat([df, events], ignore_index=True)

        print(f"Data concatenated, resulting shape: {df.shape}")
        print("Writing to ../data/bundesliga.csv")
        df.to_csv(file_path, index=False)
        return df
    
    print("Loading data form local file system")
    try:
        df = pd.read_csv(file_path).sample(frac=frac, random_state=42)
        print(f"File loaded with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error reading the file: {e}") 

"""

def log_step(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")


def load_data(frac=0.001):
    if(os.path.exists((PROJECT_ROOT_DIR / "data" / "bundesliga.csv"))):
        file_path = f"{PROJECT_ROOT_DIR}/data/bundesliga.csv" # PROJECT_ROOT_DIR / "data"
        log_step("File exists")
    else:
        log_step("File does not exist, downloading data...")
        df = pd.DataFrame()
        df_matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
        match_ids = df_matches["match_id"].to_list()

        for match_id in tqdm(match_ids, desc="Concatenating matches", unit="match"):
            events = sb.events(match_id=match_id)
            df = pd.concat([df, events], ignore_index=True)

        log_step(f"Data concatenated, resulting shape: {df.shape}")

        file_path = f"{PROJECT_ROOT_DIR}/data/bundesliga.csv"
        log_step(f"Writing to {file_path}")
        df.to_csv(file_path, index=False)
        return df
    
    log_step("Loading data form local file system")
    try:
        df = pd.read_csv(file_path).sample(frac=frac, random_state=42)
        log_step(f"File loaded with shape: {df.shape}")
        return df
    except Exception as e:
        log_step(f"Error reading the file: {e}") 

    log_step("Done")


class Dataloader:
    def __init__(self):
        # set player dimensions
        self.columns = self._load_column_configs()
        self.dtypes = self._load_dtype_configs()
        # load data
        self.df = self._load_data()

    def _load_data(self):
        if(os.path.exists((PROJECT_ROOT_DIR / "data" / "bundesliga.csv"))):
            log_step("Loading data form local file system")
            file_path = f"{PROJECT_ROOT_DIR}/data/bundesliga.csv" # PROJECT_ROOT_DIR / "data"
            df = pd.read_csv(file_path,dtype=self.dtypes)
            return df
        else:
            log_step("File does not exist, downloading data...")
            df = pd.DataFrame()
            df_matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
            match_ids = df_matches["match_id"].to_list()

            for match_id in tqdm(match_ids, desc="Concatenating matches", unit="match"):
                events = sb.events(match_id=match_id)
                df = pd.concat([df, events], ignore_index=True)

            log_step(f"Data concatenated, resulting shape: {df.shape}")
            file_path = f"{PROJECT_ROOT_DIR}/data/bundesliga.csv"
            log_step(f"Writing to {file_path}")
            df.to_csv(file_path, index=False)

    def _load_column_configs(self):
        """Load column configuration from an external JSON file."""
        with open(f"{PROJECT_ROOT_DIR}/src/columns_config.json", 'r') as f:
            return json.load(f)
        
    def _load_dtype_configs(self):
        """Load column configuration from an external JSON file."""
        with open(f"{PROJECT_ROOT_DIR}/src/dtypes_config.json", 'r') as f:
            return json.load(f)
    
    def get_dimension(self,dimension, row_filter=None):
        if row_filter is None:
            return self.df.loc[self.df["type"].isin(self.columns[dimension]["row_filter"]), self.columns[dimension]["columns"]].copy()
        return self.df[self.columns[dimension]["columns"]].copy()
    
    def get_data(self):
        return self.df.copy()
    
    
 

if __name__=="__main__":
    dataloader = Dataloader()
     
    print(dataloader.get_data().shape)