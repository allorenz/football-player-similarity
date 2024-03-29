import pandas as pd

## Purpose of this file is to create a general player information data set 
## that contains unique idea for every player 

df_std_stats = pd.read_csv('../data/fbref/cleaned/standard_stats.csv')

# filter by "Squad Total" and "Opponent Total"
teams = df_std_stats[(df_std_stats['Player'] == 'Opponent Total') | (df_std_stats['Player'] == 'Squad Total')]
team_lst = list(teams['Team'].unique())

# filter by players
df_std_stats = df_std_stats[(df_std_stats['Player'] != 'Opponent Total') & (df_std_stats['Player'] != 'Squad Total')]

t = df_std_stats['Player'].value_counts()
t

# goal is to create universal id's
## If they have 3 counts than these are most likely different players
## with the same name (e.g. there are two differents vitinhas with the same age and heritage)
## If thre are 2 counts than its potenitally due to a transfer. Easy solution is to group them
## by name
## if more than 3 then do it by hands there just 7 cases

s = df_std_stats.groupby(['Player']).sum()
s = s.reset_index()
s = s.value_counts()
s.head() 
s[4000]