import ast
import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import List
from recommendation import Recommender
from pathlib import Path
from typing import List, Union, Set



# === Constants ===
PROJECT_ROOT_DIR = Path.cwd().parent.parent


# === Helper Functions ===
def convert_to_list(input_str):
    """
    Convert a string representation of a list to an actual list.
    
    :param input_str: String representation of a list
    :return: List object
    """
    return len(ast.literal_eval(input_str))
    

def eval_unique_pos():
    # Initialize the recommender
    recommender = Recommender()
    
    # keep only player with 1 position played
    recommender.df_standard_stats["n_unique_positions_played"] = recommender.df_standard_stats["unique_positions_played"].apply(convert_to_list)
    recommender.df_standard_stats = recommender.df_standard_stats[recommender.df_standard_stats["n_unique_positions_played"] == 1]
    
    print(recommender.df_standard_stats)

def reciprocal_rank(y_true:str , y_pred:List[str], k=10):
    """
    Purpose: Users look at results from the top; gets annoyed pretty fast; stops once
    they found the first relevant; doesn’t care about the rest

    Args:
        y_true:
        y_pred:
        k:

    Returns:

    """
    y_pred_k = y_pred[:k]
    relevance_masking = [1 if label == y_true else 0 for label in y_pred_k]
    try:
        rank = relevance_masking.index(1) + 1
        return 1 / rank
    except ValueError:
        return 0


def average_precision_k(y_true:str , y_pred:List[str], k=10):
    """
    Purpose:
     Users look at results closely, every time they find a new relevant
    document, they look at the full picture of what has been before
    Args:
        y_true:
        y_pred:
        k:

    Returns:

    """
    y_pred_k = y_pred[:k]
    relevance_masking = [1 if label == y_true else 0 for label in y_pred_k]

    precisions = []
    relevant_count = 0

    for i, rel in enumerate(relevance_masking):
        if rel == 1:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precisions.append(precision_at_i)

    if not precisions:
        return 0.0  # No relevant items found

    return sum(precisions) / len(precisions)


def precision_at_k(y_true: str, y_pred: List[str], k: int = 10) -> float:
    """
    Precision@k

    Args:
        y_true: The single relevant item (ground truth).
        y_pred: Ranked list of predicted items.
        k: Cutoff rank.

    Returns:
        Precision at k (float).
    """
    y_pred_k = y_pred[:k]
    relevant_hits = sum(1 for item in y_pred_k if item == y_true)
    return relevant_hits / k

def recall_at_k(y_true: Union[List[str], Set[str]], y_pred: List[str], k: int = 10) -> float:
    """
    Recall@k

    Args:
        y_true: Set or list of relevant items (ground truth).
        y_pred: Ranked list of predicted items.
        k: Cutoff rank.

    Returns:
        Recall at k (float).
    """
    y_true = set(y_true)  # ensure it's a set
    if not y_true:
        return 0.0

    y_pred_k = set(y_pred[:k])
    retrieved_relevant = len(y_true.intersection(y_pred_k))

    return retrieved_relevant / len(y_true)

# === Main Method ===
def main():
    # setup
    recommender = Recommender(match_played=2, minutes_played=90, eval_mode=True)
    K = 10
    final_results = {}
    ex_results = {}
    

    # train test split
    train, test = train_test_split(recommender.df_standard_stats, test_size=0.2, random_state=42, stratify=recommender.df_standard_stats["position_level_0"])
    # test = test.head(1)
    recommender.train = recommender.df_standard_stats.loc[train.index].copy()
    recommender.test = recommender.df_standard_stats.loc[test.index].copy()
    experiment_setup = (
        test.groupby("position_level_0")["player"]
        .apply(list)           # collect all players per position
        .to_dict()             # turn into dictionary
    )


    # experiment_setup = {
    #     # "goalkeeper" : ["Manuel Neuer", "Gianluigi Buffon", "Jan Oblak", "Alphonse Areola"],
    #     # "defender" : ["Diego Roberto Godín Leal"], "Daniel Carvajal Ramos", "Raphaël Varane", "Gerard Piqué Bernabéu",
    #     #               "Mats Hummels"],
    #     "midfielder" : ["Thomas Müller"] # , "Mesut Özil", "Kevin De Bruyne", "Luka Modrić", "N''Golo Kanté"],
    #     # "forward" : ["Cristiano Ronaldo dos Santos Aveiro", "Luis Alberto Suárez Díaz", "Riyad Mahrez", "Kingsley Coman",
    #     #              "Timo Werner"]
    # }
    

    # start evaluation
    for position, query_players in experiment_setup.items():
        average_precision_scores = []
        reciprocal_ranks = []

        for query_player in tqdm(query_players, desc="Evaluating players"):
            try:
                # get player id
                query_player_id = test[test["player"] == query_player].index[0]
                # query_vector = test.loc[query_player_id].values.reshape(1, -1)
                # recommend
                recommended_df = recommender.recommend(
                    query_player_name=query_player, 
                    query_player_id=query_player_id
                )
                recommended_df = recommended_df[recommended_df["player"]!=query_player]

                # evaluate
                y_true: str = recommender.df_standard_stats.loc[recommender.df_standard_stats["player"] == query_player, "position_level_0"].iloc[0]
                y_pred = recommended_df["position_level_0"].values
                ap = average_precision_k(y_true, y_pred)
                rr = reciprocal_rank(y_true, y_pred)
                precision_k = precision_at_k(y_true, y_pred, k=K)
                recall_k = recall_at_k([y_true], y_pred, k=K)

                # handle results
                average_precision_scores.append(ap)
                reciprocal_ranks.append(rr)
                ex_results[query_player] = {
                    f"AP@{K}" : ap,
                    f"RR@{K}" : rr,
                    f"P@{K}" : precision_k,
                    f"R@{K}" : recall_k,
                    "y_true": y_true,
                    "y_pred": list(y_pred[:K]),
                }

            except Exception as e:
                print(f"Failed for player {query_player}.")
                print(e)

        # MAP@k per position
        final_results[position] = {
            f"MAP@{K}" : np.mean(average_precision_scores),
            f"MRR@{K}" : np.mean(reciprocal_ranks),
            f"AP@{K}" : np.mean(precision_k),
            f"R@{K}" : np.mean(recall_k),
        }

    # Output
    print(final_results)
    print(ex_results)

    # save json results to path
    os.makedirs("{PROJECT_ROOT_DIR}/experiment_results/recommendation/", exist_ok=True)
    with open(f"{PROJECT_ROOT_DIR}/experiment_results/recommendation/results_k={K}.json", "w") as f:
        json.dump(final_results, f, indent=4)
    with open(f"{PROJECT_ROOT_DIR}/experiment_results/recommendation/individual_results_k={K}.json", "w") as f:
        json.dump(ex_results, f, indent=4)


if __name__ == "__main__":
    main()
    # recommender = Recommender(match_played=2, minutes_played=90)
    # print(recommender.df_standard_stats.head())
    
    
    # df_standard_stats = pd.read_csv(
    #         f"{PROJECT_ROOT_DIR}/data/new_approach/standard_stats_all_final.csv", dtype={"player_id": "int32"}
    #     ).set_index("player_id", drop=True)
    # df_standard_stats = df_standard_stats[(df_standard_stats["minutes_played"] >= 90) & (df_standard_stats["match_played"] >= 2)]
    # train, test = train_test_split(df_standard_stats, test_size=0.2, random_state=42, stratify=df_standard_stats["position_level_0"])
    # experiment_setup = (
    #     test.groupby("position_level_0")["player"]
    #     .apply(list)           # collect all players per position
    #     .to_dict()             # turn into dictionary
    # )
    # print()