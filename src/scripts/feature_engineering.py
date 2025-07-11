"""
Execute this file to load and apply all feature engineering steps that were 
originally designed inside the notebooks.
"""
import json
import time
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from dataloader import Dataloader
from datetime import datetime

from feature_extraction.goalkeeping_extractor import GoalKeepingFeatureExtractor
from feature_extraction.defending_extractor import DefendingFeatureExtractor
from feature_extraction.passing_extractor import PassingFeatureExtractor
from feature_extraction.possession_extractor import PossessionFeatureExtractor
from feature_extraction.shooting_extractor import ShootingFeatureExtractor


PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent


def log_step(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

class OldFeatureEngineeringPipeline:
    def __init__(self, dimensions=None):
        """
        Initialize pipeline with dimension extractors
        
        :param dimensions: List of dimension extractors
        """
        self.dimensions = dimensions or []
    
    def add_dimension(self, dimension_extractor):
        """Add a new dimension extractor to the pipeline"""
        self.dimensions.append(dimension_extractor)
    
    def process(self, df, standard_stats, league):
        """
        Process raw data through all dimension extractors
        
        :param raw_data: Raw event data
        :return: Processed feature dataframe
        """
        
        for extractor_class in self.dimensions:
            extractor = extractor_class(df, standard_stats, league)
            log_step(f"{extractor.__class__.__name__}: {league}")
            extractor.run()


def old_pipeline():
    # get league names
    with open(f"{PROJECT_ROOT_DIR}/config/competition_config.json", "r") as f:
        league_mapping = json.load(f)
    league_names = [c for c in league_mapping.keys() if c != "ucl"] # dont use ucl, it misses multiple columns

    for league in league_names: # league_names:
        log_step(f"Feature Extraction for {league}")
        start = time.time()
        # load standard stats for players at league
        standard_stats = pd.read_csv(f"{PROJECT_ROOT_DIR}/data/standard_stats_{league}.csv").loc[:,["player","full_match_equivalents"]]

        # load event data
        dataloader = Dataloader(league)
        df = dataloader.load_data()

        # create pipeline
        pipeline = OldFeatureEngineeringPipeline()
        pipeline.add_dimension(GoalKeepingFeatureExtractor)
        pipeline.add_dimension(DefendingFeatureExtractor)
        pipeline.add_dimension(PassingFeatureExtractor)
        pipeline.add_dimension(PossessionFeatureExtractor)
        pipeline.add_dimension(ShootingFeatureExtractor)

        # execute pipeline
        pipeline_start_time = time.time()
        pipeline.process(df, standard_stats, league)
        pipeline_end_time = time.time()
        end = time.time()

        # execution time
        pipeline_execution_time = pipeline_end_time - pipeline_start_time
        league_execution_time = end - start

        log_step(f"Pipeline Execution time: {pipeline_execution_time:.2f} seconds")
        log_step(f"League Execution time: {league_execution_time:.2f} seconds")

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
    
    def process(self, df, standard_stats, league):
        """
        Process raw data through all dimension extractors
        
        :param raw_data: Raw event data
        :return: Processed feature dataframe
        """
        for extractor_class in tqdm(self.dimensions, desc="Processing dimensions"):
            log_step(f"{extractor_class.__name__}: {league}")
            extractor = extractor_class(df, standard_stats, league)
            extractor.run()


def run_feature_engineering(standard_stats_path:str , raw_event_data_path: str):
    log_step("Start Feature Extraction Pipeline")
    start = time.time()
    # load standard stats for players at league
    standard_stats = pd.read_csv(standard_stats_path).loc[:,["player","player_id","full_match_equivalents"]]

    # load event data
    dataloader = Dataloader(raw_event_data_path)
    dataloader.load_data()
    df = dataloader.get_data()

    # create pipeline
    pipeline = FeatureEngineeringPipeline()
    pipeline.add_dimension(GoalKeepingFeatureExtractor)
    pipeline.add_dimension(DefendingFeatureExtractor)
    pipeline.add_dimension(PassingFeatureExtractor)
    pipeline.add_dimension(PossessionFeatureExtractor)
    pipeline.add_dimension(ShootingFeatureExtractor)

    # execute pipeline
    pipeline_start_time = time.time()
    pipeline.process(df, standard_stats, None)
    pipeline_end_time = time.time()
    end = time.time()

    # execution time
    pipeline_execution_time = pipeline_end_time - pipeline_start_time
    league_execution_time = end - start

    log_step(f"Pipeline Execution time: {pipeline_execution_time:.2f} seconds")
    log_step(f"League Execution time: {league_execution_time:.2f} seconds")


if __name__ == "__main__":
    try:
        log_step("Starting feature engineering process...")
        run_feature_engineering(
            standard_stats_path="../../data/new_approach/standard_stats_all.csv",
            raw_event_data_path="../../data/new_approach/new_all_leagues.parquet"
        )
        log_step("Feature engineering process completed successfully.")
    except Exception as e:
        log_step(f"An error occurred: {e}")