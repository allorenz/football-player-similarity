from tqdm import tqdm
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import pandas as pd
from ast import literal_eval
from pathlib import Path
import os 
import sys 
from mplsoccer import Pitch, VerticalPitch, Standardizer
import numpy as np


# Run the notebook from inside the notebooks folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), '..')))

from scripts.dataloader import Dataloader


# === Constants ===
PROJECT_ROOT_DIR = Path.cwd().parent.parent

# === Helper Functions ===
def convert_to_list(input_data):
    if(isinstance(input_data, str)):
        try:
            return literal_eval(input_data)
        except (ValueError, SyntaxError):
            print(f"Error: The string {input_data} could not be converted to a list.")
            return None
    
    return input_data



def train_evaluate_model(X, y, model, scale=True, test_size=0.2, random_state=42):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Optional scaling
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation
    return classification_report(y_test, y_pred, output_dict=True, zero_division=0)


def evaluate_multiple_models(X, y_encoded, label_encoder, scale=True):
    models = {
        # "Logistic Regression": LogisticRegression(penalty="l1", solver="liblinear", C=1),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        # "LGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    }

    results_all = {}

    for name, model in models.items():
        print(f"Training and evaluating: {name}")
        raw_results = train_evaluate_model(X, y_encoded, model, scale=scale)

        # Convert digit keys back to original class names
        label_map = dict(zip(map(str, range(len(label_encoder.classes_))), label_encoder.classes_))
        readable_results = {
            label_map[k] if k.isdigit() else k: v
            for k, v in raw_results.items()
        }

        results_all[name] = readable_results

    return results_all


def main():
    # init
    dimension_event_types = {
        "shooting": ["Shot"],
        "defending": ["Pressure", "Interception", "Block", "Clearance", "Ball Recovery"],
        "possession": ["Carry" ,"Dribble"],
        "passing": ["Pass"],
        "goal_keeping": ["Goal Keeper"]
    }
    dataloader = Dataloader()
    dataloader.load_data()
    
    
    print("Loading standard stats")
    standard_stats = pd.read_csv("../../data/new_approach/standard_stats_all_final.csv")
    #standard_stats = standard_stats.loc[(standard_stats["match_played"]>=2) & (standard_stats["minutes_played"]>=90), : ].copy()
    standard_stats = standard_stats.dropna(subset=["position_level_0"])
    standard_stats = standard_stats.set_index("player_id")

    print("Create pitch")
    pitch = VerticalPitch(positional=True, positional_color='#eadddd', shade_color='#f2f2f2')
    bin_x = np.linspace(pitch.dim.left, pitch.dim.right, num=7)
    bin_y = np.sort(np.array([pitch.dim.bottom, pitch.dim.six_yard_bottom,
                            pitch.dim.six_yard_top, pitch.dim.top]))

    df_heatmap_pca = standard_stats[["player", "position_level_0","position_level_1", "position_level_2"]].copy()


    for k, v in dimension_event_types.items():
        print(f"loading dimension {k}")
        df = dataloader.get_dimension(dimension=k)
            
        # only selected types
        if "type" in df.columns:
            df = df.loc[df["type"].isin(v), ["player", "player_id", "location"]]
        df = df.set_index("player_id")


        print("Converting x,y")
        df["location"] = df["location"].apply(convert_to_list)
        df[["x", "y"]] = df["location"].apply(pd.Series)
        
        # merge with positions
        df_dimension_stats = pd.merge(
            df, 
            standard_stats[["position_level_0"]], 
            left_index=True, 
            right_index=True, 
            how='left'
        )

        # calculate heatmap for each player
        list_of_dfs = []
        # player_list = df_dimension_stats["player"].unique()
        player_list = df_dimension_stats.index.unique()

        for player in tqdm(player_list):
            player_df = df_dimension_stats.loc[df_dimension_stats.index == player, ["x", "y"]]
            player_id = player_df.reset_index()["player_id"].iloc[0]
            bin_statistic = pitch.bin_statistic(player_df["x"], player_df["y"], statistic='count', bins=(8, 8), normalize=True) # 8,8, 0.025
            bin_statistic["statistic"] = np.where(bin_statistic["statistic"] < 0.025, 0, bin_statistic["statistic"])
            bin_stastistic_1d = bin_statistic['statistic'].flatten()
            list_of_dfs.append(
                pd.DataFrame([bin_stastistic_1d], columns=[f"bin {i+1}" for i in range(len(bin_stastistic_1d))], index=[player_id])
            )

        # join bins with postion and role
        df_heatmap = pd.concat(list_of_dfs)

        df_heatmap = pd.merge(
            left=standard_stats[["position_level_0", "position_level_1", "position_level_2"]], 
            right=df_heatmap,
            left_index=True, 
            right_index=True, 
            how='left'
        )    
        df_heatmap = df_heatmap.fillna(0)


        # split in X and y
        X = df_heatmap.drop(columns=["position_level_0", "position_level_1", "position_level_2"])
        y = df_heatmap["position_level_2"]
        
        # do PCA
        pca = PCA(n_components=5)
        X_pca = pca.fit_transform(X)
        df_heatmap_pca = pd.merge(
            left=df_heatmap_pca,
            right=pd.DataFrame(X_pca, index=df_heatmap.index, columns=[f"{k}_comp_{i+1}" for i in range(X_pca.shape[1])]),
            left_index=True,
            right_index=True,
            how='left'
        )

        print(df_heatmap_pca)

        # evaluate heatmap dimension   
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Run and collect results (original features)
        all_results = evaluate_multiple_models(X, y_encoded, le, scale=True)

        # Print out accuracy of each model
        for model_name, metrics in all_results.items():
            print(f"\nModel: {model_name}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            for label in le.classes_:
                if label in metrics:
                    print(f"  {label} - Precision: {metrics[label]['precision']:.2f}, "
                        f"Recall: {metrics[label]['recall']:.2f}, F1: {metrics[label]['f1-score']:.2f}")
        

    
    # store
    df_heatmap_pca.to_csv(f"{PROJECT_ROOT_DIR}/data/new_approach/feature_multichannel_heatmap_final_test.csv")


if __name__ == "__main__":  
    main()