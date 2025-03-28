from abc import ABC, abstractmethod
import pandas as pd


class BaseDimensionFeatureExtractor(ABC):
    """Abstract base class for dimension feature extractors"""
    def __init__(self, df: pd.DataFrame, standard_stats: pd.DataFrame, league:str, dim:str):
        self.df = df
    @abstractmethod
    def convert_columns(self, raw_data):
        """Main method to extract features for a specific dimension"""
        pass

    @abstractmethod
    def extract(self):
        pass

    @abstractmethod
    def store_data(self):
        pass

    @abstractmethod
    def run(self):
        pass