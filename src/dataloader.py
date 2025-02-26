import os
import pandas as pd
from statsbombpy import sb
from tqdm import tqdm


COMPETITION_ID = 9
SEASON_ID = 27


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



def main():   
    df = load_data(frac=0.0001)
    print(df.shape)
    

if __name__=="__main__":
    main()