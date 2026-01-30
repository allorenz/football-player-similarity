import os
import json
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split, cross_validate
from sklearn.decomposition import PCA
from tqdm import tqdm
from lightgbm import LGBMClassifier

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns




# === Constants ===
PROJECT_ROOT_DIR = Path.cwd().parent.parent


# === Helper Functions ===
def get_manual_selected_features_data(match_played=2, minutes_played=90) -> pd.DataFrame:
    # define dimensions
    dimensions = ["defending","possession", "passing", "shooting", "goal_keeping"]
    
    # load standard stats
    df_standard_stats = pd.read_csv(f"{PROJECT_ROOT_DIR}/data/new_approach/standard_stats_all_test.csv",dtype={"player_id":"int32"}) # load_standard_stats(unique_index=True)
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
        path = f"{PROJECT_ROOT_DIR}/experiment_results/feature_selection_position_level_0/automated_{dim}.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        columns_to_keep.extend(data["selected_columns"])
    print(f"Number of selected manual selected features: {len(columns_to_keep)}")
    df_filtered = df_filtered.loc[:, columns_to_keep]
    
    return df_filtered

def get_data(target:str = "position_level_0", match_played=2, minutes_played=90):
    """Merges all dimensions and applies filters"""
    # vars
    dimensions = ["defending","possession", "passing", "shooting", "goal_keeping"]
    df_standard_stats = pd.read_csv(f"{PROJECT_ROOT_DIR}/data/new_approach/standard_stats_all_final.csv",dtype={"player_id":"int32"}) # load_standard_stats(unique_index=True)

    # Merge all dimensions
    df = df_standard_stats[["player_id", "position_level_0", "position_level_1","position_level_2", "match_played", "minutes_played"]].copy()
    for dim in dimensions:
        # load
        df_dimension = pd.read_csv(f"{PROJECT_ROOT_DIR}/data/new_approach/{dim}_ex.csv",dtype={"player_id":"int32"})
        # print(f"Dim {dim} shape{df_dimension.shape}")
        # print("Columns:", df_dimension.columns.tolist())
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
        path = f"{PROJECT_ROOT_DIR}/experiment_results/feature_selection_{target}/automated_{dim}.json"
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

    return df_filtered.loc[:,config_3_columns].copy()


def grid_search(X_train, y_train) -> pd.DataFrame:
    # Define models + grids
    """
    param_grid = [
        {
            "clf": [LogisticRegression(
                penalty="l1",             # L1 regularization (sparse, feature selection effect)
                solver="saga",            # required for multinomial + L1 or liblinear (OvR)
                multi_class="multinomial",# true softmax, not one-vs-rest
                max_iter=5000,            # high enough to ensure convergence
                random_state=42,
                n_jobs=-1
            )],
            "clf__C": [0.001, 0.01, 0.1, 1, 5, 10],     # inverse regularization strength
            "clf__tol": [1e-3, 1e-4],                   # convergence tolerance
            "clf__fit_intercept": [True, False],        # test whether intercept helps
            "clf__class_weight": [None, "balanced"],    # handle class imbalance
            "clf__max_iter": [2000, 5000]               # stability vs speed
        }
    ]
    """
    param_grid = [
    {
        "clf": [LGBMClassifier(
            objective="multiclass",
            num_class=4,
            boosting_type="gbdt",   # standard gradient boosting, dart
            random_state=42,
            n_jobs=-1
        )],

        # learning rate / number of estimators trade-off
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__n_estimators": [300, 500, 1000],

        # tree complexity
        "clf__max_depth": [-1, 8, 12, 16],   # -1 = no limit
        "clf__num_leaves": [31, 63, 127],    # leaves per tree

        # regularization
        "clf__reg_alpha": [0.0, 0.1, 1.0],          # L1 regularization

        # class imbalance handling
        "clf__class_weight": [None, "balanced"],
    }
]

    # Pipeline (scaler + classifier)
    pipe = Pipeline([
        ("clf", LGBMClassifier(random_state=42))
    ])

    # Stratified KFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = ParameterGrid(param_grid)

    # Grid Search
    results = []
    for params in tqdm(grid, desc="Grid Search Progress"):
        pipe.set_params(**params)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"], n_jobs=-1, return_train_score=False)

        results.append({
                    "Model": params,
                    "Accuracy (mean)": np.mean(scores["test_accuracy"]),
                    "Accuracy (std)": np.std(scores["test_accuracy"]),
                    "Precision_macro (mean)": np.mean(scores["test_precision_macro"]),
                    "Precision_macro (std)": np.std(scores["test_precision_macro"]),
                    "Recall_macro (mean)": np.mean(scores["test_recall_macro"]),
                    "Recall_macro (std)": np.std(scores["test_recall_macro"]),
                    "F1_macro (mean)": np.mean(scores["test_f1_macro"]),
                    "F1_macro (std)": np.std(scores["test_f1_macro"]),
        })

    # Collect results
    results_df = pd.DataFrame(results).sort_values(by="Accuracy (mean)", ascending=False)
    os.makedirs(f"{PROJECT_ROOT_DIR}/experiment_results/tuning", exist_ok=True)
    results_df.to_csv(f"{PROJECT_ROOT_DIR}/experiment_results/tuning/tuning_results.csv", index=False)
    # store model
    joblib.dump(results_df["Model"].iloc[0]["clf"], f"{PROJECT_ROOT_DIR}/experiment_results/tuning/best_model.pkl")

    # Print
    print("=== Head ===")
    print(results_df.head(3))
    print("=== Tail ===")
    print(results_df.tail(3))
    print("=== Lowest Standard Deviation ===")
    print(results_df.sort_values(by="Accuracy (std)", ascending=True).head(3))

    return results_df


if __name__ == "__main__":
    # init 
    label_encoder = LabelEncoder()
    target = "position_level_0"
    best_params = {}
    is_grid_search = False
    
    # Get data
    df = get_data(target=target, match_played=2, minutes_played=90)
    df = df.set_index("player_id")
    df_heatmap = pd.read_csv(f"{PROJECT_ROOT_DIR}/data/new_approach/feature_multichannel_heatmap_final_test.csv")
    df_heatmap = df_heatmap.set_index("player_id")

    # split
    X_df = df.drop(columns=["position_level_0", "position_level_1", "position_level_2"])
    y: np.ndarray = df[target].values
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

    # scale player vectors
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA on player vectors
    pca = PCA(n_components=.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Create PCA DataFrames
    X_train_pca_df = pd.DataFrame(X_train_pca, columns=[f"pca_{i+1}" for i in range(X_train_pca.shape[1])], index=X_train.index)
    X_test_pca_df = pd.DataFrame(X_test_pca, columns=[f"pca_{i+1}" for i in range(X_test_pca.shape[1])], index=X_test.index)

    # Merge PCA player vectors with heatmap features
    X_train_combined = pd.merge(
        left=X_train_pca_df,
        right=df_heatmap[[c for c in df_heatmap.columns if "comp" in c]],
        left_index=True,
        right_index=True,
        how="left"
    )
    X_test_combined = pd.merge(
        left=X_test_pca_df,
        right=df_heatmap[[c for c in df_heatmap.columns if "comp" in c]],
        left_index=True,
        right_index=True,
        how="left"
    )

    # Grid Search
    if is_grid_search:
        grid_search_results_df = grid_search(X_train_combined.values, y_train)
        best_model = joblib.load(f"{PROJECT_ROOT_DIR}/experiment_results/tuning/best_model.pkl")
        best_params = best_model.get_params()
    else:
        best_model = joblib.load(f"{PROJECT_ROOT_DIR}/experiment_results/tuning/best_model.pkl")
        best_params = best_model.get_params()

    # Fit on train predict on test
    model = LGBMClassifier(**best_params)
    model.fit(X_train_combined.values, y_train)
    y_pred = model.predict(X_test_combined.values)

    # classification report
    classification_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    with open(f"{PROJECT_ROOT_DIR}/experiment_results/tuning/classification_report.txt", "w") as f:
        f.write(classification_report)
    print(classification_report)

    # confusion matrix of model
    class_order = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
    class_indices = [list(label_encoder.classes_).index(c) for c in class_order]

    # Compute confusion matrix in this order
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=class_indices
    )

    # Plot with correct tick labels
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues',
                xticklabels=class_order,
                yticklabels=class_order)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{PROJECT_ROOT_DIR}/experiment_results/tuning/confusion_matrix.svg", format="svg")
    plt.show()