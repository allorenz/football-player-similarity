"""Execute after standards stats were extracted for all leagues."""
import pandas as pd



def main():
    leagues = ["bundesliga", "premier_league", "la_liga", "serie_a", "ligue_1", "ucl"]
    dfs = []
    for league in leagues:
        print(league)
        temp_df = pd.read_csv(f"{league}.csv")
        temp_df["competition"] = league
        dfs.append(temp_df)

    df_concat = pd.concat(dfs, ignore_index=True)
    df_concat.to_csv("../new_approach/all_leagues.csv", index=False)
    df_concat.to_parquet("../new_approach/all_leagues.parquet", index=False)



if __name__ == "__main__":
    main()