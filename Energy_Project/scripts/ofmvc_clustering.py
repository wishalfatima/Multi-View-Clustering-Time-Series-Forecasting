"""
OFMVC Clustering for Electricity Consumption Data - Fixed for NaN values
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import sys

# add path to ofmvc code
sys.path.insert(0, '../MVC-Time-series-Forecasting')

from FuzzyClustering.main_clustering import main_clustering
from FuzzyClustering.Find_Neighbors import Find_Neighbors
from FuzzyClustering.Center_Points import center_Points
from scipy.spatial.distance import cdist

# parameters for smaller sample
n_customers = 5000
n_days = 365

print(f"using sample: {n_customers} customers, {n_days} days")

# load all views
print("loading data...")
smart = pd.read_csv('data/sample_23_20percent.csv')
avgtemp = pd.read_csv('data/sample_avgtemp_23_20percent.csv')
maxtemp = pd.read_csv('data/sample_maxtemp_23_20percent.csv')
mintemp = pd.read_csv('data/sample_mintemp_23_20percent.csv')
maxgsun = pd.read_csv('data/sample_maxGSun_23_20percent.csv')
sumgsun = pd.read_csv('data/sample_sumGSun_23_20percent.csv')

# clean weather files (remove extra column and rename)
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
    # replace NaN and infinite values with 0
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data

views = []
view_names = ['smartmeter', 'avgtemp', 'maxtemp', 'mintemp', 'maxgsun', 'sumgsun']

for name, df in zip(view_names, [smart, avgtemp, maxtemp, mintemp, maxgsun, sumgsun]):
    data = prepare_view(df)
    views.append(data)
    print(f'{name}: {data.shape}')

# normalize each view
print('\nnormalizing data...')
normalized_views = []

for i, view in enumerate(views):
    scaler = MinMaxScaler()
    view_normalized = scaler.fit_transform(view)
    view_normalized = np.nan_to_num(view_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    normalized_views.append(view_normalized)
    print(f'view {i+1} normalized: {view_normalized.shape}')
    print(f'  has NaN: {np.isnan(view_normalized).any()}')

# ofmvc parameters
n_clusters = 5
n_views = len(normalized_views)
n_samples = normalized_views[0].shape[0]
n_features = normalized_views[0].shape[1]

print(f'\nclustering parameters:')
print(f'  number of clusters: {n_clusters}')
print(f'  number of views: {n_views}')
print(f'  number of samples: {n_samples}')
print(f'  features per view: {n_features}')

# ofmvc hyperparameters
t_max = 10
NR = 5
alpha1 = 0.5
q = 2
gama = 1
landa = np.array([0.5] * n_views)
viewpoint = np.ones(n_views)

# find neighbors
print('\nfinding neighbors...')
Neig, dm = Find_Neighbors(NR, normalized_views, landa, n_views)

# initialize centers
print('initializing centers...')
center_points = []
for i in range(n_views):
    dm_i = cdist(normalized_views[i], normalized_views[i])
    centers, _ = center_Points(normalized_views[i], dm_i, n_clusters, n_features, n_samples, landa[i])
    center_points.append(centers)
    print(f'  view {i+1} centers: {centers.shape}')

# run ofmvc
print('\nrunning OFMVC clustering...')
try:
    final = main_clustering(
        viewpoint=viewpoint,
        data=normalized_views,
        center_points=center_points,
        k=n_clusters,
        t_max=t_max,
        NR=NR,
        alpha1=alpha1,
        landa=landa,
        q=q,
        gama=gama,
        Neig=Neig,
        number_viwe=n_views,
        row=n_samples,
        sample_weight=np.ones(n_samples)
    )
    
    print('\nclustering completed successfully')
    
    cluster_labels = np.argmax(final, axis=0)
    np.save('results/clusters/ofmvc_clusters_small.npy', cluster_labels)
    
    os.makedirs('results/clusters', exist_ok=True)
    
    customer_ids = smart['ID'].values
    result_df = pd.DataFrame({
        'customer_id': customer_ids,
        'cluster': cluster_labels
    })
    result_df.to_csv('results/clusters/ofmvc_clusters_small.csv', index=False)
    
    print('\ncluster distribution:')
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for c, cnt in zip(unique, counts):
        print(f'  cluster {c}: {cnt} customers')
    
    print('\nresults saved to results/clusters/ofmvc_clusters_small.csv')
    
except Exception as e:
    print(f'error: {e}')
    import traceback
    traceback.print_exc()