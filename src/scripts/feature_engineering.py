"""
TODO:
Load and execute all feature engineering that were designed inside the notebooks.
"""
import os
import json
import pandas as pd
from pathlib import Path
from feature_extraction.base_extractor import BaseDimensionFeatureExtractor
from feature_extraction.goalkeeping_extractor import GoalKeepingFeatureExtractor
from dataloader import Dataloader


PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent


with open(f"{PROJECT_ROOT_DIR}/config/competition_config.json", "r") as f:
    league_mapping = json.load(f)

class FeatureEngineeringPipeline:
    def __init__(self, dimensions=None):
        """
        Initialize pipeline with dimension extractors
        
        :param dimensions: List of dimension extractors
        """
        self.dimensions = dimensions or []
    
    def add_dimension(self, dimension_extractor):
        """Add a new dimension extractor to the pipeline"""
        self.dimensions.append(dimension_extractor)
    
    def process(self, df, standard_stats, league, dim):
        """
        Process raw data through all dimension extractors
        
        :param raw_data: Raw event data
        :return: Processed feature dataframe
        """
        
        for extractor in self.dimensions:
            extractor(df, standard_stats, league, dim).run()

        
    

if __name__ == "__main__":
    #league = "bundesliga"
    dim = "goal_keeping"

    for league in league_mapping.keys():
        # load standard stats
        standard_stats = pd.read_csv(f"../../data/standard_stats_{league}.csv").loc[:,["player","full_match_equivalents"]]

        # load event data
        dataloader = Dataloader(league)
        dataloader.load_data()
        df = dataloader.get_dimension("goal_keeping")

        # execute pipeline
        pipeline = FeatureEngineeringPipeline()
        pipeline.add_dimension(GoalKeepingFeatureExtractor)
        pipeline.process(df, standard_stats, league, dim)

    # print(GoalKeepingFeatureExtractor(df, standard_stats).convert_columns())