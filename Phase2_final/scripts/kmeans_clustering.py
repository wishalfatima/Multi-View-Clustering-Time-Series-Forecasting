"""
K‑Means clustering on interpolated + Z‑normalized data (5000 customers)
Concatenates all 6 views into one feature matrix.
"""

import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans

N_CUSTOMERS = 5000
N_CLUSTERS = 5
RANDOM_STATE = 42

data_path = 'data/processed/'

print("Loading views...")
smart = pd.read_csv(data_path + 'sample_smartmeter_5k_interp_z.csv').iloc[:N_CUSTOMERS, :]
avgtemp = pd.read_csv(data_path + 'sample_avgtemp_5k_interp_z.csv').iloc[:N_CUSTOMERS, :]
maxtemp = pd.read_csv(data_path + 'sample_maxtemp_5k_interp_z.csv').iloc[:N_CUSTOMERS, :]
mintemp = pd.read_csv(data_path + 'sample_mintemp_5k_interp_z.csv').iloc[:N_CUSTOMERS, :]
maxgsun = pd.read_csv(data_path + 'sample_maxgsun_5k_interp_z.csv').iloc[:N_CUSTOMERS, :]
sumgsun = pd.read_csv(data_path + 'sample_sumgsun_5k_interp_z.csv').iloc[:N_CUSTOMERS, :]

def prepare_view(df):
    return df.drop(columns=['ID']).values.astype(float)

# Concatenate all six views
data_concatenated = np.hstack([prepare_view(df) for df in [smart, avgtemp, maxtemp, mintemp, maxgsun, sumgsun]])
print(f"Concatenated data shape: {data_concatenated.shape}")

print(f"\nRunning K‑Means with {N_CLUSTERS} clusters...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10, max_iter=300, verbose=1)
cluster_labels = kmeans.fit_predict(data_concatenated)

os.makedirs('results/clusters', exist_ok=True)
result_df = pd.DataFrame({'customer_id': smart['ID'].values, 'cluster': cluster_labels})
result_df.to_csv('results/clusters/kmeans_clusters_final.csv', index=False)

print("\n K‑Means clustering completed.")
print(f"Cluster distribution: {np.bincount(cluster_labels)}")