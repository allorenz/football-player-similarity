"""
Fix:
- goalkeeping has similar columns to shooting. fix this
"""
import os
import json
import pandas as pd
from feature_selection import load_dimension, load_standard_stats, filter_df, train_evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_data(match_played=2, minutes_played=90):
    """Merges all dimensions and applies filters"""
    # vars
    dimensions = ["defending","possession", "passing", "shooting"] # "goal_keeping"]
    df_standard_stats = pd.read_csv("../../data/new_approach/standard_stats_all.csv",dtype={"player_id":"int32"}) # load_standard_stats(unique_index=True)

    # Merge all dimensions
    df = df_standard_stats[["player_id","position", "match_played", "minutes_played"]].copy()
    for dim in dimensions:
        # load
        df_dimension = pd.read_csv(f"../../data/new_approach/{dim}_ex.csv",dtype={"player_id":"int32"})
        print(f"Dim {dim} shape{df_dimension.shape}")

        # merge and update base df
        df = pd.merge(
            left=df,
            right=df_dimension.loc[:, df_dimension.columns != "player"],
            left_on="player_id", 
            right_on="player_id",
            how="left"
        )
    print(f"Merge shape: {df.shape}")

    # filter rows
    print(f"Apply filters: match_played={match_played} , minutes_player={minutes_played}")
    df_filtered = filter_df(df, match_played=match_played, minutes_played=minutes_played)

    
    
    # filter columns
    config_1_columns = ["player_id","position"]
    config_2_columns = ["player_id","position"]

    for dim in dimensions:
        path = f"../../experiment_results/feature_selection/{dim}.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config_1_columns.extend(data["conf_1"]["selected_columns"])
        config_2_columns.extend(data["conf_2"]["selected_columns"])

    # Optionally remove duplicates if needed
    config_1_columns = list(set(config_1_columns))
    config_2_columns = list(set(config_2_columns))

    print("Config 1 columns:", len(config_1_columns))
    print("Config 2 columns:", len(config_2_columns))


    return (df_filtered.loc[:,config_1_columns].copy(), df_filtered.loc[:,config_2_columns].copy())


if __name__ == "__main__":
    print("Get data for experiments")
    experiments_tuple = get_data()
    df_1, df_2 = experiments_tuple

    for experiment in zip([df_1, df_2], ["absolute_values", "relative_values"]):
        df_ex, ex_name = experiment
        print(f"Experiment: {ex_name}, shape: {df_ex.shape}")

        print("Prepare experiment")
        df_ex = df_ex.set_index("player_id")
        X = df_ex.drop(columns=["position"])
        y = df_ex["position"]
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        
        models = {
            "logistic_regression": LogisticRegression(penalty="l1", solver="liblinear", C=1),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }
        dir_ex_results = "../../experiment_results/modeling"
        os.makedirs(dir_ex_results, exist_ok=True)

        for model_name, model in models.items():
            print(f"Train and evaluate model for {model_name.replace('_', ' ').title()}")
            ex1_results = train_evaluate_model(X, y, model, scale=True)
            print(f"Experiment 1 results for {model_name}:", ex1_results)
            result_path = f"{dir_ex_results}/expt_{ex_name}_{model_name}.json"
            with open(result_path, "w") as f:
                json.dump(ex1_results, f, indent=2)
            print(f"Results saved to {result_path}")
