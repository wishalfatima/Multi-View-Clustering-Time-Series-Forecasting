"""
Prepare PAMAP2 dataset as TIME SERIES windows (like MHEALTH)
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

def prepare_pamap2():
    """Convert PAMAP2 raw data to TIME SERIES windows"""
    
    print("📥 Loading PAMAP2 data from .dat files...")
    
    # Path to protocol folder
    protocol_path = 'data/pamap2/PAMAP2_Dataset/Protocol'
    
    # Find all .dat files
    dat_files = glob(f'{protocol_path}/subject*.dat')
    
    if not dat_files:
        print(f"No .dat files found in {protocol_path}")
        return
    
    print(f"Found {len(dat_files)} .dat files")
    
    all_data = []
    all_activities = []
    
    for file in dat_files:
        print(f"  Reading {os.path.basename(file)}...")
        
        # Read .dat file
        df = pd.read_csv(file, sep='\s+', header=None)
        
        # Replace NaN with 0
        df = df.fillna(0)
        
        # Filter out activity 0 (null class)
        df = df[df[1] != 0]
        
        # Extract sensor data (columns 3 onward)
        sensor_data = df.iloc[:, 3:].values
        
        # Extract activities
        activities = df[1].values
        
        all_data.append(sensor_data)
        all_activities.extend(activities)
    
    # Combine all data
    combined_data = np.vstack(all_data)
    combined_activities = np.array(all_activities)
    
    print(f"\nTotal raw samples: {len(combined_data)}")
    print(f"Total features: {combined_data.shape[1]}")
    
    # Create time series windows
    WINDOW_SIZE = 100  # 100 consecutive readings (1 second at 100Hz)
    STEP = 50          # 50% overlap
    
    print(f"\n📊 Creating time series windows...")
    print(f"   Window size: {WINDOW_SIZE}")
    print(f"   Step size: {STEP}")
    
    # Create windows for each sensor group separately
    # Hand sensor views (columns 0-16)
    hand_data = combined_data[:, 0:17]
    # Chest sensor views (columns 17-33)
    chest_data = combined_data[:, 17:34]
    # Ankle sensor views (columns 34-50)
    ankle_data = combined_data[:, 34:51]
    
    # Create windows for each
    hand_windows = create_time_series_windows(hand_data, WINDOW_SIZE, STEP)
    chest_windows = create_time_series_windows(chest_data, WINDOW_SIZE, STEP)
    ankle_windows = create_time_series_windows(ankle_data, WINDOW_SIZE, STEP)
    
    # Create labels for each window (majority vote)
    window_labels = []
    for start in range(0, len(combined_activities) - WINDOW_SIZE + 1, STEP):
        window_activities = combined_activities[start:start + WINDOW_SIZE]
        # Take most common activity in window
        from scipy.stats import mode
        majority_label = mode(window_activities, keepdims=True)[0][0]
        window_labels.append(majority_label)
    
    window_labels = np.array(window_labels)
    
    print(f"\nTotal windows created: {len(window_labels)}")
    
    # Create directory
    os.makedirs('data/pamap2_processed', exist_ok=True)
    
    # Save each view (reshaped to samples × time_points)
    # Hand accelerometer, gyro, magnetometer (first 9 columns)
    hand_accel = hand_windows[:, :, 0:3].reshape(hand_windows.shape[0], -1)
    hand_gyro = hand_windows[:, :, 3:6].reshape(hand_windows.shape[0], -1)
    hand_mag = hand_windows[:, :, 6:9].reshape(hand_windows.shape[0], -1)
    
    # Chest accelerometer, gyro, magnetometer
    chest_accel = chest_windows[:, :, 0:3].reshape(chest_windows.shape[0], -1)
    chest_gyro = chest_windows[:, :, 3:6].reshape(chest_windows.shape[0], -1)
    chest_mag = chest_windows[:, :, 6:9].reshape(chest_windows.shape[0], -1)
    
    # Ankle accelerometer, gyro, magnetometer
    ankle_accel = ankle_windows[:, :, 0:3].reshape(ankle_windows.shape[0], -1)
    ankle_gyro = ankle_windows[:, :, 3:6].reshape(ankle_windows.shape[0], -1)
    ankle_mag = ankle_windows[:, :, 6:9].reshape(ankle_windows.shape[0], -1)
    
    views = {
        'hand_accel': hand_accel,
        'hand_gyro': hand_gyro,
        'hand_mag': hand_mag,
        'chest_accel': chest_accel,
        'chest_gyro': chest_gyro,
        'chest_mag': chest_mag,
        'ankle_accel': ankle_accel,
        'ankle_gyro': ankle_gyro,
        'ankle_mag': ankle_mag,
    }
    
    print(f"\n📊 Creating {len(views)} time series view files...")
    
    for view_name, view_data in views.items():
        output_path = f'data/pamap2_processed/{view_name}.csv'
        np.savetxt(output_path, view_data, delimiter=',', fmt='%.6f')
        print(f"  Saved {view_name}: shape {view_data.shape}")
    
    # Map activities to 0..n-1
    unique_activities = np.unique(window_labels)
    activity_map = {act: i for i, act in enumerate(unique_activities)}
    labels_numeric = np.array([activity_map[l] for l in window_labels])
    
    # Save labels
    np.savetxt('data/pamap2_processed/labels.txt', labels_numeric, fmt='%d')
    
    # Save series names
    with open('data/pamap2_processed/series_names.txt', 'w') as f:
        for i, lbl in enumerate(window_labels):
            f.write(f"window_{i}_activity_{lbl}\n")
    
    print("\n" + "="*50)
    print("✅ PAMAP2 TIME SERIES data preparation complete!")
    print(f"   Total windows (samples): {len(labels_numeric)}")
    print(f"   Window size: {WINDOW_SIZE} time points")
    print(f"   Total activities: {len(unique_activities)}")
    print(f"   Total views: {len(views)}")
    print("="*50)

if __name__ == "__main__":
    prepare_pamap2()