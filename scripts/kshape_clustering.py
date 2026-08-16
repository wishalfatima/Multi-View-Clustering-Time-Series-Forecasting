"""
kShape clustering on interpolated + Z‑normalized data (5000 customers)
Uses only smartmeter data (1 view).
"""

import pandas as pd
import numpy as np
import os
from tslearn.clustering import KShape

N_CUSTOMERS = 5000
N_CLUSTERS = 5
MAX_ITER = 30
RANDOM_STATE = 42

data_path = 'data/processed/'

print("Loading smartmeter data...")
smart = pd.read_csv(data_path + 'sample_smartmeter_5k_interp_z.csv').iloc[:N_CUSTOMERS, :]
customer_ids = smart['ID'].values

# Remove ID column
data = smart.drop(columns=['ID']).values.astype(float)
print(f"Data shape: {data.shape}")

# Reshape for kShape: (n_samples, n_timestamps, n_features)
data_reshaped = data.reshape(data.shape[0], data.shape[1], 1)
print(f"Reshaped data: {data_reshaped.shape}")

print(f"\nRunning kShape with {N_CLUSTERS} clusters...")
kshape = KShape(n_clusters=N_CLUSTERS, max_iter=MAX_ITER, random_state=RANDOM_STATE, verbose=True)
cluster_labels = kshape.fit_predict(data_reshaped)

os.makedirs('results/clusters', exist_ok=True)
result_df = pd.DataFrame({'customer_id': customer_ids, 'cluster': cluster_labels})
result_df.to_csv('results/clusters/kshape_clusters_final.csv', index=False)

print("\n kShape clustering completed.")
print(f"Cluster distribution: {np.bincount(cluster_labels)}")