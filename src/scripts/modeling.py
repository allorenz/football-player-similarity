"""
Fix:
- goalkeeping has similar columns to shooting. fix this
"""
from operator import le
import os
import json
from xml.etree.ElementInclude import include
import pandas as pd
from feature_selection import filter_df, train_evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier


def get_data(target:str = "position_level_0", match_played=2, minutes_played=90):
    """Merges all dimensions and applies filters"""
    # vars
    dimensions = ["defending","possession", "passing", "shooting", "goal_keeping"]
    df_standard_stats = pd.read_csv("../../data/new_approach/standard_stats_all_test.csv",dtype={"player_id":"int32"}) # load_standard_stats(unique_index=True)

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

    # load and merge selected features
    for dim in dimensions:
        path = f"../../experiment_results/feature_selection_{target}/automated_{dim}.json"
        print(f"Load features from: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("---")
        print(dim)
        config_1_columns.extend(data["conf_1"]["selected_columns"])
        print(data["conf_1"]["selected_columns"])
        config_2_columns.extend(data["conf_2"]["selected_columns"])
    
    # Optionally remove duplicates if needed
    config_1_columns = list(set(config_1_columns))
    config_2_columns = list(set(config_2_columns))

    print("Config 1 columns:", len(config_1_columns))
    print("Config 2 columns:", len(config_2_columns))


    return (df_filtered.loc[:,config_1_columns].copy(), df_filtered.loc[:,config_2_columns].copy())


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


def run_modeling_v2(target: str = "position_level_0", include_heatmap=False, match_played=2, minutes_played=90):
    # load data
    print("Get experiment data.")
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

    # prepare X and y
    X = df_1.drop(columns=["position_level_0", "position_level_1", "position_level_2"])
    y = df_1[target]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

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
                multi_class="multinomial", solver="saga", max_iter=5000, random_state=42
            ))
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA()),
            ("clf", RandomForestClassifier(random_state=42))
        ]),
        "LightGBM": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA()),
            ("clf", LGBMClassifier(
                objective="multiclass", class_weight="balanced", random_state=42
            ))
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA()),
            ("clf", XGBClassifier(
                objective="multi:softmax", eval_metric="mlogloss", use_label_encoder=False,
                random_state=42
                ))
        ]),
        "SVC": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42))
        ]),
    }

    # -----------------
    # Define parameter grids (for optional tuning)
    # -----------------
    param_grids = {
        "LogReg": {
            "pca": [PCA(n_components=20), "passthrough"],  # test with & without PCA
            "clf__C": [0.1, 1, 10]
        },
        "RandomForest": {
            "pca": [PCA(n_components=50), "passthrough"],
            "clf__n_estimators": [200],
            "clf__max_depth": [None, 10, 20]
        },
        "LightGBM": {
            "pca": [PCA(n_components=50), "passthrough"],
            "clf__n_estimators": [200],
            "clf__num_leaves": [31, 64]
        }
    }

    # -----------------
    # Run evaluation
    # -----------------
    results = []

    for name, pipe in pipelines.items():
        print(f"Evaluating {name}...")
        
        # choose y depending on the model
        if name in ["XGBoost", "LightGBM"]:
            y_use = y_encoded
        else:
            y_use = y
        
        scores = cross_validate(
            pipe, X, y_use,
            cv=cv,
            scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"],
            return_train_score=False,
            n_jobs=-1
        )
        results.append({
            "Model": name,
            "Accuracy (mean)": np.mean(scores["test_accuracy"]),
            "Accuracy (std)": np.std(scores["test_accuracy"]),
            "F1_macro (mean)": np.mean(scores["test_f1_macro"]),
            "Precision_macro (mean)": np.mean(scores["test_precision_macro"]),
            "Recall_macro (mean)": np.mean(scores["test_recall_macro"]),
        })

    results_df = pd.DataFrame(results)
    print(results_df.sort_values("Accuracy (mean)", ascending=False))



if __name__ == "__main__":
    # === Experiment ===
    # levels = ["position_level_0", "position_level_1", "position_level_2"]
    # for level in levels[1:]:
    #     run_modeling(target=level, include_heatmap=True)
    #     run_modeling(target=level, include_heatmap=False)

    run_modeling_v2(
        target="position_level_0", 
        include_heatmap=True,
        match_played=4,
        minutes_played=360
    )