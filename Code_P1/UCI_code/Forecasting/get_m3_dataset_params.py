import os
import pandas as pd
import numpy as np

def get_m3_dataset_params(dataset_name, base_path=None):
    """
    Retrieve dataset parameters for different M3 dataset categories.

    Args:
        dataset_name (str): The specific M3 dataset category (e.g., 'm3-demo', 'm3-finance').
        base_path (str): The base path where datasets are stored (not used in CSV version).

    Returns:
        tuple: A tuple containing (dataset, labels, lag, look_forward, sample_overlap, 
               learning_rate, dataset_path, suilin_smape, frequency)
    """
    
    # Define dataset-specific configurations
    dataset_configs = {
        'm3-demo':     {'category': 'DEMOGRAPHIC', 'lag': 28, 'look_forward': 18, 'batch_size': 7,  'epochs': 20},
        'm3-finance':  {'category': 'FINANCE',     'lag': 28, 'look_forward': 18, 'batch_size': 9,  'epochs': 25},
        'm3-industry': {'category': 'INDUSTRY',    'lag': 36, 'look_forward': 18, 'batch_size': 10, 'epochs': 25},
        'm3-macro':    {'category': 'MACRO',       'lag': 28, 'look_forward': 18, 'batch_size': 15, 'epochs': 25},
        'm3-micro':    {'category': 'MICRO',       'lag': 28, 'look_forward': 18, 'batch_size': 18, 'epochs': 25},
        'm3-other':    {'category': 'OTHER',       'lag': 28, 'look_forward': 18, 'batch_size': 2,  'epochs': 25},
    }

    # Ensure dataset name is valid
    if dataset_name not in dataset_configs:
        raise ValueError(f"Invalid dataset name '{dataset_name}'. Choose from {list(dataset_configs.keys())}.")

    # Extract dataset settings
    config = dataset_configs[dataset_name]
    category = config['category']
    
    # Load from local CSV file
    csv_path = './data/M3Month.csv'
    
    try:
        # Read the CSV file
        all_data = pd.read_csv(csv_path)
        
        # Clean category strings (remove extra spaces)
        all_data['Category'] = all_data['Category'].str.strip()
        
        # Filter data for this category
        category_data = all_data[all_data['Category'] == category]
        
        if len(category_data) == 0:
            raise ValueError(f"No data found for category '{category}' in {csv_path}")
        
        print(f"Found {len(category_data)} series for category '{category}'")
        
        # Extract time series values (columns from index 6 onward, where the numbers start)
        raw_data = category_data.iloc[:, 6:].to_numpy().astype('float64')
        
        # Remove any columns that are all NaN (empty cells)
        raw_data = raw_data[:, ~np.isnan(raw_data).all(axis=0)]
        
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at {csv_path}. Please ensure the file exists.")
    except Exception as e:
        raise Exception(f"Error loading data from CSV: {str(e)}")

    # Load labels (assuming you have these files locally)
    # You may need to adjust this path based on where your labels are stored
    labels_file = f"{dataset_name}_fcm_raw_16_noFeWe_NR2.npy"
    labels_path = os.path.join("./data", labels_file)
    
    try:
        labels = np.load(labels_path)
    except FileNotFoundError:
        print(f"Warning: Labels file not found at {labels_path}. Using default labels.")
        # Create default labels (all zeros) as fallback
        labels = np.zeros(len(raw_data))

    # Define training parameters
    lag = config['lag']
    look_forward = config['look_forward']
    batch_size = config['batch_size']
    epochs = config['epochs']
    learning_rate = 0.0001  # Fixed learning rate for M3 datasets
    frequency = 12  # M3 datasets typically have monthly frequency

    # Compute sample overlap for sliding window approach
    sample_overlap = look_forward - 1

    # Preprocess dataset: Remove NaNs and store time series lengths
    dataset = [ts[~np.isnan(ts)] for ts in raw_data]
    seri_len = [len(ts) for ts in dataset]

    print(f"Dataset: {dataset_name} | Min Length: {np.min(seri_len)} | Max Length: {np.max(seri_len)}")

    return dataset, labels, lag, look_forward, sample_overlap, learning_rate, "./data", False, frequency