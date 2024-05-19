import pandas as pd
import utils

PATH_TO_FILES = "../data/fbref/cleaned/"

dataframes = utils.get_all_attributes(PATH_TO_FILES)
standard_stats = dataframes["standard_stats"]
shooting = dataframes["shooting"]
possession = dataframes["possession"]
playing_time = dataframes["playing_time"]
passing = dataframes["passing"]
miscellaneous_stats = dataframes["miscellaneous_stats"]
goalkeeping = dataframes["goalkeeping"]
goal_and_shot_creation = dataframes["goal_and_shot_creation"]
defensive_actions = dataframes["defensive_actions"]
advanced_goalkeeping = dataframes["advanced_goalkeeping"]


df = pd.merge(standard_stats, shooting, on="Player",how="inner",suffixes=[None,"_joined_on"])
df = df[df.columns.drop(list(df.filter(regex='_joined_on')))]
df

# performs cross join (?) as possession only has 2400+ rows
pd.merge(df, possession, on="Player",how="right",suffixes=[None,"_joined_on"] )


df = df[df.columns.drop(list(df.filter(regex='_joined_on')))]
df