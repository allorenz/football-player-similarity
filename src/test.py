import pandas as pd
import numpy as np
import time

# Create a dummy dataset
n = 10**7
df = pd.DataFrame({
    'player': np.random.choice(['player1', 'player2', 'player3'], size=n),
    'passing': np.random.randint(0, 100, size=n),
    'shooting': np.random.randint(0, 100, size=n),
    'dribbling': np.random.randint(0, 100, size=n)
})
print(df.shape)
start = time.time()

# Option 1: One combined groupby operation
result_combined = df.groupby('player').agg({
    'passing': ['sum', 'mean'],
    'shooting': ['sum', 'mean'],
    'dribbling': ['sum', 'mean']
})
end = time.time()
print(f"one oepration: {end-start}")

start = time.time()
# Option 2: Separate groupby operations for each dimension
result_passing = df.groupby('player')['passing'].agg(['sum', 'mean'])
result_shooting = df.groupby('player')['shooting'].agg(['sum', 'mean'])
result_dribbling = df.groupby('player')['dribbling'].agg(['sum', 'mean'])
result_passing = df.groupby('player')['passing'].agg(['sum', 'mean'])
result_shooting = df.groupby('player')['shooting'].agg(['sum', 'mean'])
result_dribbling = df.groupby('player')['dribbling'].agg(['sum', 'mean'])
end = time.time()
print(f"multiple oeprations: {end-start}")