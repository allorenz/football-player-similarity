from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np 

# load data
df = pd.read_csv('../data/preprocessed_data.csv', sep=',')
df_player = pd.read_csv('../data/player_data.csv', sep=',')

# prepare player attributes
X = df.drop(['Unnamed: 0'], axis=1, inplace=False)
X = X.set_index("ID")
idx = pd.DataFrame()
idx["ID"] = df["ID"]

# prepare the target
df_player = df_player.drop(['Unnamed: 0'], axis=1, inplace=False).set_index('ID')
target = X.loc[[212198]].values.tolist()
print(target)
print("Queried Player")
name_of_target = df_player.loc[[212198], 'Name'].values[0]
print(name_of_target)

# prepare the labels
label_numbers = [i for i in range(len(X.columns))]
labels = [label for label in X.columns ]
label_dict = dict(zip(label_numbers, labels))

# create cluster len(X.columns)
kmeans = KMeans(n_clusters=10, random_state=42, n_init="auto").fit(X)
target_arr  = np.array(target)
pred = kmeans.predict(target_arr)

# predict target
pred_label = label_dict[int(pred)]
print(f"Output of the prediction: {pred_label}")

# Conclusion
"""
When working with k mean clustering, I created clusters that correspond to the number
of columns in X. Meaning, for each variable I create one cluster. 
When predicting, the function returned the variable (label) that is the closest
to one of of the values that are inthe target array/vector. 
This not what I wanted, as I am hoping to get a whole row of values in return.
"""