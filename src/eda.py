import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../data/fbref/standard_stats.csv')
data = data[data["Season"] == "2022-2023"]

# drop columns as they contain 'Machtes' or 'nan' values
data = data.drop(columns=['Unnamed: 24_level_0_Matches', 'Unnamed: 33_level_0_Matches'], inplace=False)

standard_stats = data.rename(columns={'Unnamed: 0_level_0_Player': 'Player',
                                      'Unnamed: 1_level_0_Nation': 'Nation', 
                                      'Unnamed: 2_level_0_Pos': 'Pos',
                                      'Unnamed: 3_level_0_Age' : 'Age', 
                                      'Unnamed: 4_level_0_MP': 'MP'})

# MP (Mathches Played) seems odd for some player including Ederson


# inspect uniqueness of player names
standard_stats["Player"].value_counts()
standard_stats[standard_stats["Player"] == "Loïc Bade"]
# conclusion
# Some players occur multiple times
# - seems to be due to transfer during the season
#   - player becomes new record when playing for new club



# inspect xGoals per 90 minutes
standard_stats["Per 90 Minutes_xG"].hist(bins=10)
standard_stats[standard_stats["Per 90 Minutes_xG"] > 2]

# conclusion: 
# returns at first players who played one match
# and scored more or equal than 2 goals in game or
# shows it for a whole squad entry 
# --> check fbref as they faced the same problem and considered
# only players with a certain amount of matches played for any calculation
# to avoid this disturbance 


# plotting histograms exclude GK
standard_stats.columns
features = [ 'Age', 'MP',
       'Playing Time_Starts', 'Playing Time_Min', 'Playing Time_90s',
       'Performance_Gls', 'Performance_Ast', 'Performance_G+A',
       'Performance_G-PK', 'Performance_PK', 'Performance_PKatt',
       'Performance_CrdY', 'Performance_CrdR', 'Progression_PrgC',
       'Progression_PrgP', 'Progression_PrgR', 'Per 90 Minutes_Gls',
       'Per 90 Minutes_Ast', 'Per 90 Minutes_G+A', 'Per 90 Minutes_G-PK',
       'Per 90 Minutes_G+A-PK', 'Playing Time_MP', 'Expected_xG',
       'Expected_npxG', 'Expected_xAG', 'Expected_npxG+xAG',
       'Per 90 Minutes_xG', 'Per 90 Minutes_xAG', 'Per 90 Minutes_xG+xAG',
       'Per 90 Minutes_npxG', 'Per 90 Minutes_npxG+xAG']


X = standard_stats[standard_stats["Pos"] != "GK"]
X = X[features]
X.hist(bins=10,figsize=(20,20));

# conclusion
# the x-axis has to be adapted for some attributes e.g. xGoals which
data = pd.read_csv('../data/fbref/shooting.csv')
data = data[data["Season"] == "2022-2023"]



# CONCLUSIONS
# - standard_stats contains for each club the accumulated overall stats like xGoals
#   - that can be use to analyze clubs and their playing styles
# - create df only for gk and field players
# - create player ids (solve multiple player records problem)
# check all csv files as they might contain redudant attributes
# check variables aggregated/combined variables (e.g per 90) as they contain information
# - keep per 90 or keep original attributes