"""
Drop-in replacement for get_m3_dataset_params
Returns electricity data in M3-compatible format
"""

import pandas as pd
import numpy as np

def get_electricity_dataset_params(dataset_name='electricity'):
    """
    Mimics get_m3_dataset_params but returns electricity data.
    """
    
    # load 2023 smartmeter data
    data_2023 = pd.read_csv('data/sample_23_20percent.csv')
    
    # use 5000 customers
    n_customers = 5000
    data_2023 = data_2023.iloc[:n_customers, 1:].values
    
    # convert to list of time series
    dataset = [data_2023[i, :].astype(float) for i in range(n_customers)]
    
    # remove NaN values
    dataset = [ts[~np.isnan(ts)] for ts in dataset]
    
    # dummy labels
    labels = np.zeros(n_customers, dtype=int)
    
    # forecasting parameters
    lag = 28
    look_forward = 18
    sample_overlap = look_forward - 1
    learning_rate = 0.0001
    dataset_path = './data'
    suilin_smape = False
    frequency = 7
    
    print(f"loaded {len(dataset)} customers")
    
    return dataset, labels, lag, look_forward, sample_overlap, learning_rate, dataset_path, suilin_smape, frequency