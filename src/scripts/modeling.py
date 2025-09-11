

import os
import json
import numpy as np
import pandas as pd


from pathlib import Path
from feature_selection import train_evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split


# === Constants ===
PROJECT_ROOT_DIR = Path.cwd().parent.parent

# === Helper Functions ===
def get_data(target:str = "position_level_0", match_played=2, minutes_played=90):
    """Merges all dimensions and applies filters"""
    # vars
    dimensions = ["defending","possession", "passing", "shooting", "goal_keeping"]
    df_standard_stats = pd.read_csv("../../data/new_approach/standard_stats_all_final.csv",dtype={"player_id":"int32"}) # load_standard_stats(unique_index=True)

    # Merge all dimensions
    df = df_standard_stats[["player_id", "position_level_0", "position_level_1","position_level_2", "match_played", "minutes_played"]].copy()
    for dim in dimensions:
        # load
        df_dimension = pd.read_csv(f"../../data/new_approach/{dim}_ex.csv",dtype={"player_id":"int32"})
        print(f"Dim {dim} shape{df_dimension.shape}")
        print("Columns:", df_dimension.columns.tolist())
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
    df_filtered = df.loc[(df["match_played"]>=match_played) & (df["minutes_played"]>=minutes_played), : ].copy()#filter_df(df, match_played=match_played, minutes_played=minutes_played)

    # filter columns
    config_1_columns = ["player_id", "position_level_0", "position_level_1","position_level_2"]
    config_2_columns = ["player_id", "position_level_0", "position_level_1","position_level_2"]
    config_3_columns = ["player_id", "position_level_0", "position_level_1","position_level_2"]

    # load and merge selected features
    for dim in dimensions:
        path = f"../../experiment_results/feature_selection_{target}/automated_{dim}.json"
        print(f"Load features from: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("---")
        print(dim)
        config_1_columns.extend(data["conf_1"]["selected_columns"])
        #print(data["conf_1"]["selected_columns"])
        config_2_columns.extend(data["conf_2"]["selected_columns"])
        config_3_columns.extend(data["conf_3"]["selected_columns"])
    
    # Optionally remove duplicates if needed
    config_1_columns = list(set(config_1_columns))
    config_2_columns = list(set(config_2_columns))
    config_3_columns = list(set(config_3_columns))

    print("Config 1 columns:", len(config_1_columns))
    print("Config 2 columns:", len(config_2_columns))
    print("Config 3 columns:", len(config_3_columns))

    return (df_filtered.loc[:,config_1_columns].copy(), df_filtered.loc[:,config_2_columns].copy(), df_filtered.loc[:,config_3_columns].copy())

# === Modeling ===
def run_modeling(target: str = "position_level_0", include_heatmap=False, match_played=2, minutes_played=90):
    print("Get data for experiments")
    experiments_tuple = get_data(target=target, match_played=match_played, minutes_played=minutes_played)
    df_1, df_2 = experiments_tuple
    print(f"Absolute values shape: {df_1.shape}")


    # load heatmap data
    df_heatmap = pd.read_csv("../../data/new_approach/feature_multichannel_heatmap.csv")
    df_heatmap = df_heatmap.set_index("player_id")

    if include_heatmap:
        print(f"Heatmap data shape: {df_heatmap.shape}")
        df_1 = df_1.merge(
            df_heatmap[[c for c in df_heatmap.columns if "comp" in c]],
            left_on="player_id",
            right_index=True,
            how="left"
        )
        df_2 = df_2.merge(
            df_heatmap[[c for c in df_heatmap.columns if "comp" in c]],
            left_on="player_id",
            right_index=True,
            how="left"
        )
        print(f"Fused inputdata: {df_1.shape}, {df_2.shape}")


    for experiment in zip([df_1, df_2], ["absolute_values", "relative_values"]):
        df_ex, ex_name = experiment
        print(f"Experiment: {ex_name}, shape: {df_ex.shape}")

        print("Prepare experiment")
        df_ex = df_ex.set_index("player_id")
        X = df_ex.drop(columns=["position_level_0", "position_level_1", "position_level_2"])
        y = df_ex[target]
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        
        models = {
            # "logistic_regression": LogisticRegression(penalty="l1", solver="liblinear", C=1),
            "LGMB" : LGBMClassifier(verbose=-1),
            # "xgboost" : XGBClassifier(use_label_encoder=True, eval_metric='logloss'), 
            # "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            # "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
        dir_ex_results = f"../../experiment_results/modeling_{target}"
        os.makedirs(dir_ex_results, exist_ok=True)

        for model_name, model in models.items():
            print(f"Train and evaluate model for {model_name.replace('_', ' ').title()}")
            ex1_results = train_evaluate_model(X, y, model, scale=True, test_size=0.4)
            print(f"Experiment 1 results for {model_name}:", ex1_results)
            result_path = f"{dir_ex_results}/automated_{ex_name}_{model_name}_heatmap_{include_heatmap}.json"
            with open(result_path, "w") as f:
                json.dump(ex1_results, f, indent=2)
            print(f"Results saved to {result_path}")


def run_modeling_v2(df: pd.DataFrame, target: str = "position_level_0", include_heatmap=False, data_flag="automated", df_name="manual_selected"):
    # load heatmap data
    df_heatmap = pd.read_csv("../../data/new_approach/feature_multichannel_heatmap.csv")
    df_heatmap = df_heatmap.set_index("player_id")

    # -----------------
    # Cross-validation strategy
    # -----------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # -----------------
    # Define pipelines
    # -----------------
    pipelines = {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA()),  # param grid can control n_components or disable
            ("clf", LogisticRegression(
                penalty="l1", solver="saga", max_iter=2000, random_state=42
            ))
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA()),
            ("clf", RandomForestClassifier(
                        n_estimators=200,    # more stable than 100
                        max_depth=None,      # let it grow, but monitor overfitting
                        min_samples_leaf=2,  # prevents tiny leaves, helps generalization
                        random_state=42,
                        n_jobs=-1
                    ))
        ]),
        "LightGBM": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA()),
            ("clf", LGBMClassifier(
                        objective="multiclass",
                        n_estimators=200,    # baseline boosting rounds
                        num_leaves=31,       # safe default, balances depth & variance
                        learning_rate=0.1,   # standard start
                        class_weight="balanced",  # useful if your classes are uneven
                        random_state=42,
                        n_jobs=-1
                    ))
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA()),
            ("clf", XGBClassifier(
                objective="multi:softprob",   # probabilities for multiclass
                eval_metric="mlogloss",       # stable multiclass metric
                n_estimators=200,             # 200 boosting rounds is plenty here
                max_depth=6,                  # balanced trees (not too deep for 2.5k samples)
                learning_rate=0.1,            # standard starting point
                subsample=0.8,                # random row sampling (prevents overfit)
                colsample_bytree=0.8,         # random feature sampling (helps with correlated features)
                reg_lambda=1.0,               # L2 regularization
                reg_alpha=0.5,                # L1 regularization (sparsity on weights)
                random_state=42,
                n_jobs=-1
            ))
        ])
    }

    # -----------------
    # Run evaluation with PCA yes/no and Heatmap yes/no
    # -----------------
    results = []

    # keep untouched base data
    df_base = df.copy()

    for heatmap_flag, heatmap_label in [(False, "noHeatmap"), (True, "Heatmap")]:
        
        if heatmap_flag:
            df = df_base.merge(
                df_heatmap[[c for c in df_heatmap.columns if "comp" in c]],
                left_on="player_id",
                right_index=True,
                how="left"
            )
        else:
            df = df_base.copy()

        # prepare X and y
        df = df.fillna(0)
        X = df.drop(columns=["position_level_0", "position_level_1", "position_level_2"])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        print(f"X_train: {X_train.shape}, y_train: {y_train_encoded.shape}")

        for name, pipe in pipelines.items():

            # Choose y depending on the model
            if name in ["XGBoost", "LightGBM"]:
                y_use = y_train_encoded
            else:
                y_use = y_train

            # Test both PCA ON and PCA OFF
            for pca_setting, pca_label in [(PCA(n_components=.95, random_state=42), "PCA"), ("passthrough", "noPCA")]:
                pipe_clone = clone(pipe)
                pipe_clone.set_params(pca=pca_setting)

                print(f"=== Model: {name}, Heatmap: {heatmap_label}, PCA: {pca_label} ===")
                scores = cross_validate(
                    pipe_clone, X_train, y_use,
                    cv=cv,
                    scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
                    return_train_score=False,
                    n_jobs=-1
                )

                results.append({
                    "Model": f"{name}_{pca_label}",
                    "Heatmap": heatmap_label,  # <-- NEW COLUMN
                    "Accuracy (mean)": np.mean(scores["test_accuracy"]),
                    "Accuracy (std)": np.std(scores["test_accuracy"]),
                    "Precision_macro (mean)": np.mean(scores["test_precision_macro"]),
                    "Precision_macro (std)": np.std(scores["test_precision_macro"]),
                    "Recall_macro (mean)": np.mean(scores["test_recall_macro"]),
                    "Recall_macro (std)": np.std(scores["test_recall_macro"]),
                    "F1_macro (mean)": np.mean(scores["test_f1_macro"]),
                    "F1_macro (std)": np.std(scores["test_f1_macro"]),
                })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Accuracy (mean)", ascending=False)
    print(results_df)

    # Save results to CSV
    out_path = PROJECT_ROOT_DIR / "experiment_results" / f"modeling_{target}"
    out_path.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path / f"{data_flag}_results_{df_name}.csv", index=False)


def get_manual_selected_features_data(match_played=2, minutes_played=90) -> pd.DataFrame:
    # define dimensions
    dimensions = ["defending","possession", "passing", "shooting", "goal_keeping"]
    
    # load standard stats
    df_standard_stats = pd.read_csv(f"{PROJECT_ROOT_DIR}/data/new_approach/standard_stats_all_final.csv",dtype={"player_id":"int32"}) # load_standard_stats(unique_index=True)
    df = df_standard_stats[["player_id", "position_level_0", "position_level_1", "position_level_2", "match_played", "minutes_played"]].copy()
    
    # load and merge all dimensions
    for dim in dimensions:
        # load
        df_dimension = pd.read_csv(f"{PROJECT_ROOT_DIR}/data/new_approach/{dim}_ex.csv",dtype={"player_id":"int32"})
        print(f"Full dimension '{dim}' shape {df_dimension.shape}")

        # merge and update base df
        df = pd.merge(
            left=df,
            right=df_dimension.loc[:, df_dimension.columns != "player"],
            left_on="player_id", 
            right_on="player_id",
            how="left"
        )
    # filter rows
    print(f"Apply filters: match_played={match_played} , minutes_player={minutes_played}")
    df_filtered = df.loc[(df["match_played"]>=match_played) & (df["minutes_played"]>=minutes_played), : ].copy()#filter_df(df, match_played=match_played, minutes_played=minutes_played)

    # keep only manually selected columns
    columns_to_keep = ["player_id", "position_level_0", "position_level_1","position_level_2"]
    for dim in dimensions:
        path = f"{PROJECT_ROOT_DIR}/experiment_results/feature_selection_manual/manual_{dim}.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        columns_to_keep.extend(data["selected_columns"])
    print(f"Number of selected manual selected features: {len(columns_to_keep)}")
    df_filtered = df_filtered.loc[:, columns_to_keep]
    
    return df_filtered

if __name__ == "__main__":
    # configs
    match_played = 2
    minutes_played = 90
    levels = ["position_level_0", "position_level_1", "position_level_2"]
    
    # === Experiment Automated Selected Feautures ===
    for level in levels:
        # load data
        experiments_tuple = get_data(target=level, match_played=match_played, minutes_played=minutes_played)
        df_1, df_2, df_3 = experiments_tuple

        for df_ex, df_name in zip([df_1, df_2, df_3], ["absolute_values", "relative_values", "all_values"]):
            # run experiment
            run_modeling_v2(
                df=df_ex,
                target=level, 
                include_heatmap=True,
                data_flag="automated",
                df_name=df_name
            )

    # === Experiment Manual Selected Features ===
    for level in levels:
        # load data
        df = get_manual_selected_features_data(match_played=match_played, minutes_played=minutes_played)
        # run experiment
        run_modeling_v2(
            df=df,
            target=level, 
            include_heatmap=True,
            data_flag="manual",
            df_name="manual_selected"
        )