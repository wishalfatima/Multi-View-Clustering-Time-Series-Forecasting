"""
Test OFMVC forecasting only
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, '../MVC-Time-series-Forecasting')
sys.path.insert(0, '../MVC-Time-series-Forecasting/Forecasting')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_model_modified import run_local_models
from parameters import get_electricity_dataset_params

def clean_forecasting_data(df):
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)
    return df.astype(float)

# replace get_m3_dataset_params
import get_m3_dataset_params
get_m3_dataset_params.get_m3_dataset_params = get_electricity_dataset_params

import run_model_modified
run_model_modified.get_m3_dataset_params = get_electricity_dataset_params

# load ofmvc clusters
clusters_df = pd.read_csv('results/clusters/ofmvc_clusters.csv')
cluster_labels = clusters_df['cluster'].values
n_clusters = len(np.unique(cluster_labels))

print(f"OFMVC - Number of clusters: {n_clusters}")

# load and clean data
data_2023 = pd.read_csv('data/sample_23_20percent.csv')
data_2024 = pd.read_csv('data/sample_24_20percent.csv')

data_2023 = clean_forecasting_data(data_2023)
data_2024 = clean_forecasting_data(data_2024)

# save cleaned data
data_2023.to_csv('data/sample_23_clean.csv', index=False)
data_2024.to_csv('data/sample_24_clean.csv', index=False)

print(f"Data shape: 2023={data_2023.shape}, 2024={data_2024.shape}")
print(f"Data types: {data_2023.dtypes.unique()}")

try:
    results, init_data, preds, test_y, train_y, pred_df = run_local_models(
        dataset_name='electricity',
        number_of_clusters=n_clusters,
        AEName='TCN',
        Dim=16,
        epochs=10,
        batch=20,
        use_saved_model=False,
        save_trained_model=False,
        run=1,
        external_clusters=cluster_labels
    )
    
    print(f"\nSUCCESS!")
    print(f"Validation SMAPE: {np.mean(results['val_SMAPE']):.2f}%")
    print(f"Test SMAPE: {np.mean(results['test_SMAPE']):.2f}%")
    
except Exception as e:
    print(f"Error: {e}")