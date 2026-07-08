"""
K-Means Clustering for Electricity Consumption Data
Uses all 6 views (smartmeter + weather) for fair comparison with OFMVC and FMVC
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import os

# parameters
n_customers = 5000
n_days = 365
n_clusters = 5
random_state = 42

print(f"using sample: {n_customers} customers, {n_days} days")
print(f"number of clusters: {n_clusters}")
print(f"random state: {random_state}")

# load all 6 views
print("\nloading data...")
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

# take subset of customers and days
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

# prepare view (remove ID column and handle NaN)
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

# normalize each view and concatenate
print('\nnormalizing and concatenating views...')
scaler = MinMaxScaler()
normalized_views = []

for view in views:
    view_norm = scaler.fit_transform(view)
    view_norm = np.nan_to_num(view_norm, nan=0.0, posinf=0.0, neginf=0.0)
    normalized_views.append(view_norm)

# concatenate all views into one matrix
data = np.hstack(normalized_views)
print(f'concatenated data shape: {data.shape}')

# run K-Means clustering
print(f"\nrunning K-Means clustering with {n_clusters} clusters...")
kmeans = KMeans(
    n_clusters=n_clusters, 
    random_state=random_state, 
    n_init=10, 
    verbose=1,
    max_iter=300
)
cluster_labels = kmeans.fit_predict(data)

print("\nclustering completed successfully")

# save results
os.makedirs('results/clusters', exist_ok=True)

np.save('results/clusters/kmeans_clusters.npy', cluster_labels)

customer_ids = smart['ID'].values
result_df = pd.DataFrame({
    'customer_id': customer_ids,
    'cluster': cluster_labels
})
result_df.to_csv('results/clusters/kmeans_clusters.csv', index=False)

print('\ncluster distribution:')
unique, counts = np.unique(cluster_labels, return_counts=True)
for c, cnt in zip(unique, counts):
    print(f'  cluster {c}: {cnt} customers')

print('\ncluster percentages:')
for c, cnt in zip(unique, counts):
    percentage = (cnt / n_customers) * 100
    print(f'  cluster {c}: {percentage:.1f}%')

print('\nresults saved to:')
print('  results/clusters/kmeans_clusters.csv')
print('  results/clusters/kmeans_clusters.npy')

print(f'\ninertia (within-cluster sum of squares): {kmeans.inertia_:.2f}')