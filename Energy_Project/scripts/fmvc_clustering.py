"""
FMVC Clustering for Electricity Consumption Data - Vectorized Version
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

print("loading data...")

# parameters
n_customers = 5000
n_days = 365
n_clusters = 5
fuzzy_exponent = 2
max_iter = 30
epsilon = 1e-6

# load data
smart = pd.read_csv('data/sample_23_20percent.csv')
avgtemp = pd.read_csv('data/sample_avgtemp_23_20percent.csv')
maxtemp = pd.read_csv('data/sample_maxtemp_23_20percent.csv')
mintemp = pd.read_csv('data/sample_mintemp_23_20percent.csv')
maxgsun = pd.read_csv('data/sample_maxGSun_23_20percent.csv')
sumgsun = pd.read_csv('data/sample_sumGSun_23_20percent.csv')

# clean weather files
weather_files = [avgtemp, maxtemp, mintemp, maxgsun, sumgsun]
for df in weather_files:
    if '2022-12-31 23:59:59' in df.columns:
        df.drop(columns=['2022-12-31 23:59:59'], inplace=True)
    new_cols = ['ID'] + list(smart.columns[1:])
    df.columns = new_cols

# take subset
def take_subset(df, n_cust, n_days):
    cols = df.columns[:n_days+1]
    df_subset = df[cols]
    df_subset = df_subset.iloc[:n_cust]
    return df_subset

smart = take_subset(smart, n_customers, n_days)
avgtemp = take_subset(avgtemp, n_customers, n_days)
maxtemp = take_subset(maxtemp, n_customers, n_days)
mintemp = take_subset(mintemp, n_customers, n_days)
maxgsun = take_subset(maxgsun, n_customers, n_days)
sumgsun = take_subset(sumgsun, n_customers, n_days)

# prepare view
def prepare_view(df):
    data = df.drop(columns=['ID']).values.astype(float)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data

views = []
view_names = ['smartmeter', 'avgtemp', 'maxtemp', 'mintemp', 'maxgsun', 'sumgsun']

for name, df in zip(view_names, [smart, avgtemp, maxtemp, mintemp, maxgsun, sumgsun]):
    data = prepare_view(df)
    views.append(data)
    print(f'{name}: {data.shape}')

# normalize and concatenate
print('\nnormalizing and concatenating views...')
scaler = MinMaxScaler()
normalized_views = []

for view in views:
    view_norm = scaler.fit_transform(view)
    view_norm = np.nan_to_num(view_norm, nan=0.0, posinf=0.0, neginf=0.0)
    normalized_views.append(view_norm)

data = np.hstack(normalized_views)
print(f'concatenated data shape: {data.shape}')

n_samples, n_features = data.shape

# initialize membership matrix randomly
print('\ninitializing membership...')
membership = np.random.rand(n_samples, n_clusters)
membership = membership / membership.sum(axis=1, keepdims=True)

# initialize centers
centers = np.zeros((n_clusters, n_features))
for j in range(n_clusters):
    centers[j] = np.average(data, axis=0, weights=membership[:, j])

# vectorized fmvc
print(f'\nrunning FMVC clustering with {n_clusters} clusters...')
m = fuzzy_exponent

for iteration in range(max_iter):
    # save old membership for convergence check
    old_membership = membership.copy()
    
    # update centers (vectorized)
    for j in range(n_clusters):
        weights = membership[:, j] ** m
        if weights.sum() > 0:
            centers[j] = np.average(data, axis=0, weights=weights)
    
    # update membership (vectorized)
    # compute distances from each sample to each center
    dist = np.zeros((n_samples, n_clusters))
    for j in range(n_clusters):
        diff = data - centers[j]
        dist[:, j] = np.sqrt(np.sum(diff ** 2, axis=1))
    
    # handle zero distances
    min_dist = np.min(dist, axis=1, keepdims=True)
    zero_mask = (min_dist == 0)
    
    # compute new membership
    for j in range(n_clusters):
        ratio = dist / np.maximum(dist[:, j:j+1], 1e-10)
        ratio = ratio ** (2 / (m - 1))
        denominator = np.sum(ratio, axis=1)
        membership[:, j] = 1 / denominator
    
    # fix zero distance cases
    for i in range(n_samples):
        if zero_mask[i, 0]:
            membership[i, :] = 0
            membership[i, np.argmin(dist[i])] = 1
    
    # check convergence
    diff = np.max(np.abs(membership - old_membership))
    print(f'  iteration {iteration+1}: change = {diff:.6f}')
    
    if diff < epsilon:
        print(f'converged after {iteration+1} iterations')
        break

# get hard cluster labels
cluster_labels = np.argmax(membership, axis=1)

print('\nclustering completed successfully')

# save results
os.makedirs('results/clusters', exist_ok=True)

np.save('results/clusters/fmvc_clusters.npy', cluster_labels)

customer_ids = smart['ID'].values
result_df = pd.DataFrame({
    'customer_id': customer_ids,
    'cluster': cluster_labels
})
result_df.to_csv('results/clusters/fmvc_clusters.csv', index=False)

print('\ncluster distribution:')
unique, counts = np.unique(cluster_labels, return_counts=True)
for c, cnt in zip(unique, counts):
    print(f'  cluster {c}: {cnt} customers')

print('\nresults saved to results/clusters/fmvc_clusters.csv')
print('results saved to results/clusters/fmvc_clusters.npy')