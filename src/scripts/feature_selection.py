import os 
import json
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report


# === Constants ===
PROJECT_ROOT_DIR = Path.cwd().parent.parent


# === Helper Functions
def load_standard_stats(unique_index=False):
    leagues = ["bundesliga", "la_liga", "ligue_1", "premier_league", "serie_a"]
    file_paths = [f"../../data/standard_stats_{l}.csv" for l in leagues]
    df = pd.DataFrame()
    
    for file in file_paths:
        temp_df = pd.read_csv(file)  # Load DataFrame from CSV file
        df = pd.concat([df, temp_df], ignore_index=True)
    df = df.set_index("player")

    if unique_index:
        df = df.loc[~df.index.duplicated(keep='first')]


    return df

def load_dimension(dim, unique_index=False):
    leagues = ["bundesliga", "la_liga", "ligue_1", "premier_league", "serie_a"]
    file_paths = [f"../../data/processed/{l}/{dim}.csv" for l in leagues]
    df = pd.DataFrame()
    
    for file in file_paths:
        temp_df = pd.read_csv(file)  # Load DataFrame from CSV file
        df = pd.concat([df, temp_df], ignore_index=True)
    df = df.set_index("player")

    if unique_index:
        df = df.loc[~df.index.duplicated(keep='first')]

    return df

def filter_df(df_input, match_played=2, minutes_played=90) -> pd.DataFrame:
    df_output = df_input.copy()
    df_output =  df_output.loc[(df_output["match_played"]>match_played) &  (df_output["minutes_played"]>minutes_played) ,:]
    return df_output

def feature_selection(X, y, model, scale_data=True):
    feature_names = X.columns
    if scale_data:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)  

    model.fit(X, y)

    selector = SelectFromModel(model, max_features=20, threshold="mean", importance_getter="coef_")

    selected_columns = feature_names[selector.get_support()]
    
    return selected_columns

def train_evaluate_model(X, y, model, scale=True, test_size=0.2, random_state=42):
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Optionale Skalierung
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Report
    """
    zero_division=0 := set zero if no predictions made for a class (surpresses warning)
    - Reason: for goal_keeping model model never predicts for midfielder and forward
    """
    return classification_report(y_test, y_pred, output_dict=True, zero_division=0)




if __name__ == "__main__":
    # vars
    dimensions = ["defending","possession", "passing", "shooting", "goal_keeping"]
    model = LogisticRegression(penalty="l1", solver="liblinear", C=1, class_weight='balanced')
    results = dict()

    for dim in dimensions:
        print(f"Current dimension: {dim}")
        # load
        df_dimension = load_dimension(dim)
        df_standard_stats = load_standard_stats()

        # merge and filter
        df_full = pd.merge(left=df_standard_stats[["position","match_played","minutes_played"]],
            right=df_dimension,
            left_index=True, 
            right_index=True)
        df_filtered = filter_df(df_full, match_played=2, minutes_played=90)

        # prepare columns
        target_column = "position"
        columns_absolute_values = [col for col in df_filtered.columns if not col.endswith("_%") and not col.endswith("_per_match") and col != target_column and col != "match_played" and col != "minutes_played"]
        columns_relative_values = [col for col in df_filtered.columns if col.endswith("_%") and col != target_column]
        config = {
            "conf_1" : {
                "columns_value_type" : "absolute_values",
                "columns" : columns_absolute_values,
            },
            "conf_2": {
                "columns_value_type" : "relative_values",
                "columns" : columns_relative_values,
            }
        }

        # do feature selection
        model = LogisticRegression(penalty="l1", solver="liblinear", C=1)
        results = dict()
        

        for c in config:
            total_selected_features = 0
            print(f"Processing feature representation: {config[c]['columns_value_type']}")
            X = df_filtered[config[c]["columns"]]
            y = df_filtered["position"]


            selected_columns = feature_selection(X, y, model, scale_data=True)
            X_selected = df_filtered[selected_columns]
            total_selected_features += len(selected_columns)
            

            prediction_results = train_evaluate_model(X_selected, y, model, scale=True)
            results[c] = {
                "scores": prediction_results,
                "selected_columns": list(selected_columns),
                "n_features" : int(len(selected_columns))
            }
            print(f"selected features {dim}-{config[c]['columns_value_type']}: {total_selected_features}")

        # Create output directory if it doesn't exist
        dir_ex_results = "../../experiment_results/feature_selection"
        os.makedirs(dir_ex_results, exist_ok=True)    
        with open(f"{dir_ex_results}/{dim}.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {dir_ex_results}/{dim}.json")

    print("### Feature Selection Done. ###")