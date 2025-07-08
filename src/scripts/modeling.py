"""
Fix:
- goalkeeping has similar columns to shooting. fix this
"""

import json
import pandas as pd
from feature_selection import load_dimension, load_standard_stats, filter_df, train_evaluate_model


def get_data(match_played=2, minutes_played=90):
    """Merges all dimensions and applies filters"""
    # vars
    dimensions = ["defending","possession", "passing", "shooting"] # "goal_keeping"]
    df_standard_stats = load_standard_stats(unique_index=True)

    # Merge all dimensions
    df = df_standard_stats[["position", "match_played", "minutes_played"]].copy()
    for dim in dimensions:
        # load
        df_dimension = load_dimension(dim,unique_index=True)
        print(f"Dim shape{df_dimension.shape}")
        # merge and update base df
        df = pd.merge(
            left=df,
            right=df_dimension,
            left_index=True, 
            right_index=True,
            how="inner"  # optional, prevents dropping unmatched rows
        )
    print(f"Merge shape: {df.shape}")
    
    # filter rows
    print(f"Apply filters: match_played={match_played} , minutes_player={minutes_played}")
    df_filtered = filter_df(df, match_played=match_played, minutes_played=minutes_played)
    
    # filter columns
    config_1_columns = ["position"]
    config_2_columns = ["position"]

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
    experiments_tuple = get_data()
    df_1, df_2 = experiments_tuple

    print(df_2["position"])
