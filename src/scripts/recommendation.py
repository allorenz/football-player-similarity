import os
import json
import math
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from feature_selection import filter_df
from sklearn.preprocessing import normalize, StandardScaler
from pathlib import Path



# === Constants ===
PROJECT_ROOT_DIR = Path.cwd().parent.parent



class Recommender:
    def __init__(self, match_played=2, minutes_played=90, eval_mode=False):
        self.match_played = match_played
        self.minutes_played = minutes_played
        self.dimensions = ["defending", "possession", "passing", "shooting"]
        self.df = None
        self.df_standard_stats = None
        self.query_player_id = None
        self.df_cosine = None
        self.df_result = None
        self.eval_mode = eval_mode
        self.train = None
        self.test = None
        self._load_data()

    def _load_data(self):
        # load merge all dimensinos and apply filters
        self.df = get_data(match_played=self.match_played, minutes_played=self.minutes_played)
        self.df = self.df.set_index("player_id", drop=True)
        
        # load standard stats
        self.df_standard_stats = pd.read_csv(
            f"{PROJECT_ROOT_DIR}/data/new_approach/standard_stats_all_final.csv", dtype={"player_id": "int32"}
        ).set_index("player_id", drop=True)
        self.df_standard_stats = self.df_standard_stats.loc[(self.df_standard_stats["match_played"]>=self.match_played) & (self.df_standard_stats["minutes_played"]>=self.minutes_played), : ].copy()#filter_df(df, match_played=match_played, minutes_played=minutes_played)
        


    def compute_cosine_similarities(self, query_player_id=None):
        # init 
        df_cosine_list = []
        
       
        
        for dim in self.dimensions:
            # Load selected features
            path = f"{PROJECT_ROOT_DIR}/experiment_results/feature_selection_position_level_0/automated_{dim}.json"
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            selected_columns = data["conf_3"]["selected_columns"]

            # Extract dimension-specific data
            df_dim = self.df.loc[:, selected_columns].copy()

            if not self.eval_mode:
                # Separate query player
                # get query player id
                self.query_player_id = self.df_standard_stats.loc[
                    self.df_standard_stats["player"] == self.query_player_name
                ].index[0]
                query_vector: np.ndarray = df_dim.loc[self.query_player_id].values.reshape(1, -1)
                df_dim_wo_query: np.ndarray = df_dim.drop(self.query_player_id).to_numpy()
            else:
                # query from whole df
                query_vector: np.ndarray = df_dim.loc[query_player_id].values.reshape(1, -1)
                # matrix of training data
                df_dim_wo_query: np.ndarray = df_dim.loc[self.train.index].to_numpy()

            # Standardize
            scaler = StandardScaler()
            scaler.fit(df_dim_wo_query)
            df_dim_scaled = pd.DataFrame(scaler.transform(df_dim.to_numpy()), index=df_dim.index, columns=df_dim.columns)
            query_vector_scaled = scaler.transform(query_vector)

            # Compute similarity
            similarities = cosine_similarity(df_dim_scaled.values, query_vector_scaled).flatten() # normalization not neccessary: cosine_similarity(normalize(df_dim_scaled.values), normalize(query_vector_scaled)).flatten()
            df_dim_eval = pd.DataFrame({f"sim_{dim}": similarities}, index=df_dim.index)
            df_cosine_list.append(df_dim_eval)

        # Combine all similarities
        self.df_cosine = pd.concat(df_cosine_list, axis=1)

        # Aggregate scores
        self.df_cosine['avg_sim'] = self.df_cosine.mean(axis=1)
        self.df_cosine['weighted_avg'] = self.df_cosine.apply(weighted_avg, axis=1, args=(self.dimensions,self.weights))

    def build_result_dataframe(self):
        """built output dataframe with player information and cosine similarities"""
        self.df_result = pd.merge(
            self.df_standard_stats[["player","position_level_0", "position_level_1","position_level_2", "match_played", "minutes_played", "team", "country"]],
            self.df_cosine,
            left_index=True,
            right_index=True,
            how="left"
        ).sort_values(by="avg_sim", ascending=False)

    def plot_results(self):
        plot_boxplots(self.df_result, self.dimensions)
        plot_distribution_plot(self.df_result, self.dimensions)

    def recommend(self, query_player_name="Thomas Müller", query_player_id = None, dimensions=["goal_keeping", "defending", "passing","possession","shooting"], weights = None) -> pd.DataFrame:
        self.dimensions = dimensions
        self.weights = weights
        # init query player
        self.query_player_name = query_player_name

        # compute cosine
        self.compute_cosine_similarities(query_player_id=query_player_id)
        
        # build result dataframe
        self.build_result_dataframe()

        # create plots
        if not self.eval_mode:
            self.plot_results()  

        return self.df_result

def get_data(target:str = "position_level_0", match_played=2, minutes_played=90):
    """Merges all dimensions and applies filters"""
    # vars
    dimensions = ["goal_keeping", "defending","possession", "passing", "shooting"]
    df_standard_stats = pd.read_csv(f"{PROJECT_ROOT_DIR}/data/new_approach/standard_stats_all_final.csv",dtype={"player_id":"int32"}) # load_standard_stats(unique_index=True)

    # Merge all dimensions
    df = df_standard_stats[["player_id", "position_level_0", "position_level_1","position_level_2", "match_played", "minutes_played"]].copy()
    for dim in dimensions:
        # load
        df_dimension = pd.read_csv(f"{PROJECT_ROOT_DIR}/data/new_approach/{dim}_ex.csv",dtype={"player_id":"int32"})
        print(f"Dim {dim} shape{df_dimension.shape}")
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
    ex_config_all_values_columns = ["player_id", "position_level_0", "position_level_1","position_level_2"]

    # load and merge selected features
    for dim in dimensions:
        path = f"{PROJECT_ROOT_DIR}/experiment_results/feature_selection_{target}/automated_{dim}.json"
        # print(f"Load features from: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # print("---")
        # print(dim)
        ex_config_all_values_columns.extend(data["conf_3"]["selected_columns"])
    
    # remove duplicates 
    ex_config_all_values_columns = list(set(ex_config_all_values_columns))

    print("Config 3 columns:", len(ex_config_all_values_columns))

    return df_filtered.loc[:,ex_config_all_values_columns].copy()

def weighted_avg(row, dimensions, weights):
    if weights is None:
        weights = [1 for _ in range(len(dimensions))]
    col = [f"sim_{d}"for d in dimensions]
    values = row[col]
    return (values * weights).sum() / sum(weights)


def plot_boxplots(df, dimensions, save_path=f"{PROJECT_ROOT_DIR}/out/plots/boxplot.png"):
    import matplotlib.pyplot as plt

    columns_to_plot = [f"sim_{d}"for d in dimensions]
    data_to_plot = df[columns_to_plot]

    # Plotting
    plt.figure(figsize=(10, 6))
    data_to_plot.boxplot()
    plt.title('Box Plot of Simulation Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_distribution_plot(df, dimensions, save_path=f"{PROJECT_ROOT_DIR}/out/plots/distribution_plot.png"):
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Set up plot style
    sns.set(style="whitegrid")

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    columns = [f"sim_{d}"for d in dimensions]
    n = len(columns)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten()

    # Plot each distribution
    for i, col in enumerate(columns):
        sns.histplot(data=df, x=col, kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
    plt.tight_layout()
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


"""
def dev():
    scaler = StandardScaler()
    # Get data
    df, _ = get_data(match_played=2, minutes_played=90)
    df = df.set_index("player_id", drop=True)
    df_standard_stats = pd.read_csv(f"{PROJECT_ROOT_DIR}/data/new_approach/standard_stats_all.csv", dtype={"player_id": "int32"})
    df_standard_stats = df_standard_stats.set_index("player_id", drop=True)
    
    # Set query player
    query_player_name = "Thomas Müller"
    query_player_id = df_standard_stats.loc[df_standard_stats["player"]==query_player_name].index[0]
    print(f"Query player name: {query_player_name}, Query player ID: {query_player_id}")
    
    # calculate cosine similarity for each dimension
    dimensions = ["defending","possession", "passing", "shooting"] # , "goal_keeping"]
    df_cosine = pd.DataFrame(index=df.index)

    for dim in dimensions:
        # load dimension selected features
        dimension_selected_features_path = f"{PROJECT_ROOT_DIR}/experiment_results/feature_selection/{dim}.json"
        with open(dimension_selected_features_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # filter df for selected columns
        selected_columns = data["conf_1"]["selected_columns"]
        df_dim = df.loc[:, selected_columns].copy()
        print(df_dim.shape)

        # get query vector
        query_vector = df_dim.loc[query_player_id].values.reshape(1, -1)

        # scale the dimension df, and player
        df_dim = pd.DataFrame(scaler.fit_transform(df_dim), index=df_dim.index, columns=df_dim.columns)
        query_vector = scaler.transform(query_vector)

        # Calculate cosine similarity
        query_vector = df_dim.loc[query_player_id].values.reshape(1, -1)
        similarities = cosine_similarity(normalize(df_dim.values, norm='l2'), normalize(query_vector, norm='l2')).flatten()
        df_dim_eval = pd.DataFrame({
            "player": df_dim.index,
            f"sim_{dim}": similarities.reshape(-1)
        })
        df_dim_eval = df_dim_eval.set_index("player", drop=True)

        # merge cosine similarity results
        df_cosine = pd.merge(
            left=df_cosine,
            right=df_dim_eval,
            left_index=True,
            right_index=True,
            how="left"
        )
        
    df_cosine['avg_sim'] = df_cosine.mean(axis=1)
    df_cosine['weighted_avg'] = df_cosine.apply(weighted_avg, axis=1)
    
    df_result = pd.merge(
        left=df_standard_stats[["player","position","team","country", "match_played"]],
        right=df_cosine,
        left_index=True,
        right_index=True,
        how="left"
    ).sort_values(by="avg_sim", ascending=False)
    print(df_result)

    plot_boxplots(df_result)

    plot_distribution_plot(df_result)
"""


def main():
    rec = Recommender(match_played=2, minutes_played=90, eval_mode=False)
    output_df = rec.recommend(query_player_name="Eden Hazard")
    print(output_df)


if __name__ == "__main__":
    main()
    