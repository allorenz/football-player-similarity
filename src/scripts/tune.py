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
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split, cross_validate
from tqdm import tqdm
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
        path = f"{PROJECT_ROOT_DIR}/experiment_results/feature_selection_manual/manual_{dim}.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        columns_to_keep.extend(data["selected_columns"])
    print(f"Number of selected manual selected features: {len(columns_to_keep)}")
    df_filtered = df_filtered.loc[:, columns_to_keep]
    
    return df_filtered



def grid_search(X_train, y_train) -> pd.DataFrame:
    # Define models + grids
    param_grid = [
        {
            "clf": [XGBClassifier(
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
            )],
            "clf__n_estimators": [100, 200, 400],          # boosting rounds
            # "clf__learning_rate": [0.01, 0.05, 0.1, 0.2], # smaller rate needs more trees
            "clf__max_depth": [3, 5, 7, 9],               # tree depth tradeoff (bias vs variance)
            "clf__min_child_weight": [1, 3, 5],           # min sum of instance weight per leaf
            # "clf__subsample": [0.6, 0.8, 1.0],            # row sampling
            # "clf__colsample_bytree": [0.6, 0.8, 1.0],     # feature sampling
            "clf__gamma": [0, 0.1, 0.5, 1],               # min loss reduction for split
            "clf__reg_lambda": [0.5, 1.0, 2.0],           # L2 regularization
            "clf__reg_alpha": [0, 0.1, 0.5, 1.0]          # L1 regularization
           
        }
    ]

    # Pipeline (scaler + classifier)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier())  # placeholder, replaced by param_grid
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
    df = get_manual_selected_features_data(match_played=2, minutes_played=90)
    X: np.ndarray = df.drop(columns=["position_level_0", "position_level_1", "position_level_2"]).values
    y: np.ndarray = df[target].values
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y, random_state=42)

    # Grid Search
    if is_grid_search:
        grid_search_results_df = grid_search(X_train, y_train)
        best_model = joblib.load(f"{PROJECT_ROOT_DIR}/experiment_results/tuning/best_model.pkl")
        best_params = best_model.get_params()
    else:
        best_model = joblib.load(f"{PROJECT_ROOT_DIR}/experiment_results/tuning/best_model.pkl")
        best_params = best_model.get_params()

    # Fit on train predict on test
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # classification report
    classification_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    with open(f"{PROJECT_ROOT_DIR}/experiment_results/tuning/classification_report.txt", "w") as f:
        f.write(classification_report)
    print(classification_report)

    # confusion matrix of model
    cm = confusion_matrix(y_test, y_pred, labels=range(len(label_encoder.classes_)))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{PROJECT_ROOT_DIR}/experiment_results/tuning/confusion_matrix.png")
    plt.show()