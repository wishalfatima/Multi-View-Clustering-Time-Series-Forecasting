"""
kShape Clustering for Electricity Consumption Data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from tslearn.clustering import KShape

# parameters
n_customers = 5000
n_days = 365
n_clusters = 5
max_iter = 30
random_state = 42

print(f"using sample: {n_customers} customers, {n_days} days")
print(f"number of clusters: {n_clusters}")
print(f"max iterations: {max_iter}")

# load smartmeter data
print("\nloading smartmeter data...")
smart = pd.read_csv('data/sample_23_20percent.csv')

# take subset
smart_subset = smart.iloc[:n_customers, :n_days+1]

# save customer ids
customer_ids = smart_subset['ID'].values

# remove ID column
data = smart_subset.drop(columns=['ID']).values

print(f"data shape: {data.shape}")

# normalize data
print("\nnormalizing data...")
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data.T).T

# verify no NaN
print(f"contains NaN: {np.isnan(data_normalized).any()}")

# replace NaN if any
data_normalized = np.nan_to_num(data_normalized, nan=0.0)

# reshape for kShape: (n_samples, n_timestamps, n_features)
# for univariate time series, n_features = 1
data_reshaped = data_normalized.reshape(data_normalized.shape[0], data_normalized.shape[1], 1)
print(f"reshaped data: {data_reshaped.shape}")

# run kShape clustering
print(f"\nrunning kShape clustering...")
kshape = KShape(
    n_clusters=n_clusters, 
    max_iter=max_iter, 
    random_state=random_state, 
    verbose=True
)
cluster_labels = kshape.fit_predict(data_reshaped)

print("\nclustering completed successfully")

# save results
os.makedirs('results/clusters', exist_ok=True)

np.save('results/clusters/kshape_clusters.npy', cluster_labels)

result_df = pd.DataFrame({
    'customer_id': customer_ids,
    'cluster': cluster_labels
})
result_df.to_csv('results/clusters/kshape_clusters.csv', index=False)

print('\ncluster distribution:')
unique, counts = np.unique(cluster_labels, return_counts=True)
for c, cnt in zip(unique, counts):
    print(f'  cluster {c}: {cnt} customers')

print('\nresults saved to:')
print('  results/clusters/kshape_clusters.csv')
print('  results/clusters/kshape_clusters.npy')