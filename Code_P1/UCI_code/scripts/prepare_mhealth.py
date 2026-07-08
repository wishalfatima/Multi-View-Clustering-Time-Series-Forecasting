"""
Prepare MHEALTH dataset for OFMVC - Creates proper time series windows
"""

import pandas as pd
import numpy as np
import os
from glob import glob

def create_time_series_windows(data, window_size=100, step=50):
    """
    Convert long sequence into overlapping windows
    data: 2D array (total_samples, features)
    Returns: 3D array (n_windows, window_size, features)
    """
    n_samples = data.shape[0]
    n_features = data.shape[1]
    
    windows = []
    for start in range(0, n_samples - window_size + 1, step):
        window = data[start:start + window_size, :]
        windows.append(window)
    
    return np.array(windows)

def prepare_mhealth():
    """Convert raw MHEALTH log files to time series windows"""
    
    print("📥 Loading MHEALTH raw data from log files...")
    
    # Get all log files
    log_files = sorted(glob('data/mhealth/mHealth_subject*.log'))
    print(f"Found {len(log_files)} subject files")
    
    # Define sensor columns (23 sensor readings)
    # We'll read all columns except the last one (label)
    all_data = []
    all_labels = []
    
    for file in log_files:
        print(f"  Reading {os.path.basename(file)}...")
        # Log files are space-separated, no header
        df = pd.read_csv(file, sep='\s+', header=None)
        
        # Columns 0-22 are sensor readings, column 23 is label
        sensor_data = df.iloc[:, 0:23].values
        labels = df.iloc[:, 23].values
        
        all_data.append(sensor_data)
        all_labels.append(labels)
    
    # Combine all subjects
    combined_data = np.vstack(all_data)
    combined_labels = np.concatenate(all_labels)
    
    print(f"\nTotal raw samples: {len(combined_data)}")
    print(f"Total sensors (features): {combined_data.shape[1]}")
    
    # Create time series windows
    WINDOW_SIZE = 100  # 100 time points per sample (2 seconds at 50Hz)
    STEP = 50          # 50% overlap
    
    print(f"\n📊 Creating time series windows...")
    print(f"   Window size: {WINDOW_SIZE}")
    print(f"   Step size: {STEP}")
    
    # Create windows for each sensor separately (each becomes a view)
    n_sensors = combined_data.shape[1]
    sensor_names = [
        'chest_acc_x', 'chest_acc_y', 'chest_acc_z',
        'chest_ecg_1', 'chest_ecg_2',
        'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z',
        'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
        'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z',
        'arm_acc_x', 'arm_acc_y', 'arm_acc_z',
        'arm_gyro_x', 'arm_gyro_y', 'arm_gyro_z',
        'arm_mag_x', 'arm_mag_y', 'arm_mag_z',
    ]
    
    # Create processed directory
    os.makedirs('data/mhealth_processed', exist_ok=True)
    
    all_windows = []
    labels_for_windows = []
    
    for sensor_idx in range(n_sensors):
        sensor_data = combined_data[:, sensor_idx:sensor_idx+1]  # Keep as 2D
        windows = create_time_series_windows(sensor_data, WINDOW_SIZE, STEP)
        all_windows.append(windows)
        
        # Get labels for each window (majority vote)
        window_labels = []
        for start in range(0, len(combined_labels) - WINDOW_SIZE + 1, STEP):
            window_labels_batch = combined_labels[start:start + WINDOW_SIZE]
            # Take most common label in window
            from scipy.stats import mode
            majority_label = mode(window_labels_batch, keepdims=True)[0][0]
            window_labels.append(majority_label)
        
        if sensor_idx == 0:
            labels_for_windows = window_labels
        
        # Save as CSV (samples, time points)
        # Each row is a sample, each column is a time point
        windows_2d = windows.reshape(windows.shape[0], -1)
        output_path = f'data/mhealth_processed/{sensor_names[sensor_idx]}.csv'
        np.savetxt(output_path, windows_2d, delimiter=',', fmt='%.6f')
        print(f"  Saved {sensor_names[sensor_idx]}: shape {windows_2d.shape}")
    
    # Save labels
    np.savetxt('data/mhealth_processed/labels.txt', labels_for_windows, fmt='%d')
    
    # Save series names
    with open('data/mhealth_processed/series_names.txt', 'w') as f:
        for i in range(len(labels_for_windows)):
            f.write(f"window_{i}\n")
    
    print("\n" + "="*50)
    print("✅ MHEALTH time series data preparation complete!")
    print(f"   Total windows (samples): {len(labels_for_windows)}")
    print(f"   Window size: {WINDOW_SIZE} time points")
    print(f"   Total sensors (views): {n_sensors}")
    print("="*50)

if __name__ == "__main__":
    prepare_mhealth()