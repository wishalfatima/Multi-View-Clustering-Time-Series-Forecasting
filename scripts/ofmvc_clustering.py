"""
OFMVC clustering on interpolated + Z-normalized data (5000 customers)
Now with view-weight LEARNING enabled (learn_weights=True) — this is what
distinguishes true OFMVC (Optimal FMVC) from plain FMVC.
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, '/Users/wishal/Documents/P1_Project/MVC-Time-series-Forecasting')

from scipy.spatial.distance import cdist
from FuzzyClustering.main_clustering import main_clustering
from FuzzyClustering.Find_Neighbors import Find_Neighbors
from FuzzyClustering.Center_Points import center_Points

# Parameters
N_CUSTOMERS = 5000   
N_DAYS = 365
N_CLUSTERS = 5
T_MAX = 20
NR = 5
ALPHA1 = 0.5
Q = 2
GAMA = 1
LANDA = 0.5

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

view_names = ['smartmeter', 'avgtemp', 'maxtemp', 'mintemp', 'maxgsun', 'sumgsun']
views = [prepare_view(df) for df in [smart, avgtemp, maxtemp, mintemp, maxgsun, sumgsun]]
print(f"Data shape: {views[0].shape}")

# No extra scaling – data is already Z-normalized
normalized_views = views

n_views = len(normalized_views)
n_samples = normalized_views[0].shape[0]
n_features = normalized_views[0].shape[1]
landa_array = np.array([LANDA] * n_views)
viewpoint = np.ones(n_views)

print(f"\nClustering {n_samples} customers with {n_views} views, {N_CLUSTERS} clusters")

print("\nFinding neighbors...")
Neig, dm = Find_Neighbors(NR, normalized_views, landa_array, n_views)

print("Initializing centers...")
center_points = []
for i in range(n_views):
    dm_i = cdist(normalized_views[i], normalized_views[i])
    centers, _ = center_Points(normalized_views[i], dm_i, N_CLUSTERS,
                               n_features, n_samples, landa_array[i])
    center_points.append(centers)

print("\nRunning OFMVC (with view-weight learning enabled)...")
Final, learned_weights = main_clustering(
    viewpoint=viewpoint,
    data=normalized_views,
    center_points=center_points,
    k=N_CLUSTERS,
    t_max=T_MAX,
    NR=NR,
    alpha1=ALPHA1,
    landa=landa_array,
    q=Q,
    gama=GAMA,
    Neig=Neig,
    number_viwe=n_views,
    row=n_samples,
    sample_weight=np.ones(n_samples),
    learn_weights=True   # <-- this is what makes it OFMVC, not FMVC
)

print("\n" + "="*60)
print("LEARNED VIEW WEIGHTS")
print("="*60)
for name, weight in zip(view_names, learned_weights):
    print(f"  {name:12s}: {weight:.4f}")
uniform = 1.0 / n_views
print(f"\n(For reference, uniform/FMVC weight would be: {uniform:.4f} for each view)")

cluster_labels = np.argmax(Final, axis=0)
os.makedirs('results/clusters', exist_ok=True)
result_df = pd.DataFrame({'customer_id': smart['ID'].values, 'cluster': cluster_labels})
result_df.to_csv('results/clusters/ofmvc_clusters_final.csv', index=False)

print("\n OFMVC clustering completed.")
print(f"Cluster distribution: {np.bincount(cluster_labels)}")