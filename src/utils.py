import re
import os
import pandas as pd


# add mio and k to values 
def adjust_money_appearance(x):
    x = int(x)
    if(x >= 1000000):
        x = x / 1000000
        x = "€" + str(x) + " Mio"
        return x
    if(x < 1000000 and x >= 10000):
        x = x / 1000
        x = "€" + str(x) + "k"
        return x
    return "€" + str(x)


# Function to clean the column
def clean_nation(nation):
    pattern = r'\s([A-Z]+)$'
    match = re.search(pattern, str(nation))
    if match:
        return match.group(1)
    else:
        return nation
    
# Import all files all at once
def get_all_attributes(path_to_files):
    filenames = [
        'advanced_goalkeeping.csv',
        'defensive_actions.csv',
        'goalkeeping.csv',
        'goal_and_shot_creation.csv',
        'miscellaneous_stats.csv',
        'passing.csv',
        'playing_time.csv',
        'possession.csv',
        'shooting.csv',
        'standard_stats.csv'
    ]

    dataframes = {}

    for filename in filenames:
        filepath = os.path.join(path_to_files, filename)
        df_name = filename.replace('.csv', '') 
        dataframes[df_name] = pd.read_csv(filepath)

    return dataframes