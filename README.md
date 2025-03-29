## Instructions
1. Execute dataloader.py script. 
    It downloads raw event data of top 5 european leauge of the season 2015/16.
2. Execute standard_stats.py script. 
    It fetches and generates numerous standard stats e.g. minutes_played, full_match_equivalents. __full_match_equivalents__ is currently required for the following step.
3. Execute feature_engineering.py script to extract feature from raw event data.
    The generated features are stored locally.