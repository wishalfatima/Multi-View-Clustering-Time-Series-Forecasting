"""
Prepare HAR dataset for OFMVC clustering
Handles the actual HAR folder structure with train/test
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

def load_har_data(har_path='data/har/raw/UCI HAR Dataset'):
    """
    Load HAR dataset from train/test folders
    """
    print("📂 Loading HAR dataset...")
    
    # Load training data
    train_X = pd.read_csv(f'{har_path}/train/X_train.txt', sep='\s+', header=None)
    train_y = pd.read_csv(f'{har_path}/train/y_train.txt', sep='\s+', header=None)
    train_subject = pd.read_csv(f'{har_path}/train/subject_train.txt', sep='\s+', header=None)
    
    # Load test data
    test_X = pd.read_csv(f'{har_path}/test/X_test.txt', sep='\s+', header=None)
    test_y = pd.read_csv(f'{har_path}/test/y_test.txt', sep='\s+', header=None)
    test_subject = pd.read_csv(f'{har_path}/test/subject_test.txt', sep='\s+', header=None)
    
    # Combine train and test
    X = pd.concat([train_X, test_X], ignore_index=True)
    y = pd.concat([train_y, test_y], ignore_index=True)
    subject = pd.concat([train_subject, test_subject], ignore_index=True)
    
    # Activity labels mapping
    activity_labels = {
        1: 'WALKING',
        2: 'WALKING_UPSTAIRS',
        3: 'WALKING_DOWNSTAIRS',
        4: 'SITTING',
        5: 'STANDING',
        6: 'LAYING'
    }
    
    # Map activity numbers to names
    y['activity'] = y[0].map(activity_labels)
    
    print(f"   Total samples: {X.shape[0]}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Subjects: {subject[0].nunique()}")
    print(f"   Activities: {y['activity'].unique()}")
    
    return X, y, subject

def create_views_from_sensors(har_path='data/har/raw/UCI HAR Dataset'):
    """
    Load raw sensor data as separate views
    Each sensor signal becomes a view
    """
    print("\n🔧 Loading raw sensor signals as views...")
    
    views = {}
    sensor_types = ['body_acc', 'body_gyro', 'total_acc']
    axes = ['x', 'y', 'z']
    
    for sensor in sensor_types:
        for axis in axes:
            view_name = f"{sensor}_{axis}"
            
            # Load train data
            train_file = f'{har_path}/train/Inertial Signals/{sensor}_{axis}_train.txt'
            test_file = f'{har_path}/test/Inertial Signals/{sensor}_{axis}_test.txt'
            
            train_data = pd.read_csv(train_file, sep='\s+', header=None)
            test_data = pd.read_csv(test_file, sep='\s+', header=None)
            
            # Combine train and test
            combined = pd.concat([train_data, test_data], ignore_index=True)
            
            views[view_name] = combined.values
            print(f"  Loaded {view_name}: shape {combined.shape}")
    
    return views

def load_inertial_signals(har_path='data/har/raw/UCI HAR Dataset', max_samples=500):
    """
    Load inertial signals and group by subject and activity
    """
    print("\n📊 Loading inertial signals...")
    
    # Load labels and subjects
    train_y = pd.read_csv(f'{har_path}/train/y_train.txt', sep='\s+', header=None)
    train_subject = pd.read_csv(f'{har_path}/train/subject_train.txt', sep='\s+', header=None)
    test_y = pd.read_csv(f'{har_path}/test/y_test.txt', sep='\s+', header=None)
    test_subject = pd.read_csv(f'{har_path}/test/subject_test.txt', sep='\s+', header=None)
    
    # Combine
    all_y = pd.concat([train_y, test_y], ignore_index=True)
    all_subject = pd.concat([train_subject, test_subject], ignore_index=True)
    
    # Activity mapping
    activity_map = {
        1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS',
        4: 'SITTING', 5: 'STANDING', 6: 'LAYING'
    }
    
    # Load inertial signals
    sensor_files = {
        'body_acc_x': 'body_acc_x',
        'body_acc_y': 'body_acc_y', 
        'body_acc_z': 'body_acc_z',
        'body_gyro_x': 'body_gyro_x',
        'body_gyro_y': 'body_gyro_y',
        'body_gyro_z': 'body_gyro_z',
        'total_acc_x': 'total_acc_x',
        'total_acc_y': 'total_acc_y',
        'total_acc_z': 'total_acc_z'
    }
    
    # Store time series grouped by subject + activity
    all_series = {}
    
    for subject_id in all_subject[0].unique()[:5]:  # Limit to 5 subjects
        for activity_id in all_y[0].unique():
            activity_name = activity_map[activity_id]
            
            # Find indices for this subject and activity
            subject_idx = all_subject[all_subject[0] == subject_id].index
            activity_idx = all_y[all_y[0] == activity_id].index
            idx = subject_idx.intersection(activity_idx)
            
            if len(idx) > 0:
                key = f"subject_{subject_id}_{activity_name}"
                all_series[key] = {}
                
                # Load each sensor as a separate view
                for sensor_name, file_prefix in sensor_files.items():
                    # Load train and test data for this sensor
                    train_file = f'{har_path}/train/Inertial Signals/{file_prefix}_train.txt'
                    test_file = f'{har_path}/test/Inertial Signals/{file_prefix}_test.txt'
                    
                    train_data = pd.read_csv(train_file, sep='\s+', header=None)
                    test_data = pd.read_csv(test_file, sep='\s+', header=None)
                    all_data = pd.concat([train_data, test_data], ignore_index=True)
                    
                    # Get the time series for this subject/activity
                    series = all_data.iloc[idx].values.flatten()
                    
                    # Limit length
                    if len(series) > max_samples:
                        series = series[:max_samples]
                    
                    all_series[key][sensor_name] = series
    
    print(f"  Created {len(all_series)} time series")
    print(f"  Each has {len(list(all_series.values())[0]) if all_series else 0} views")
    
    return all_series

def save_for_ofmvc(all_series, output_path):
    """Save prepared data for OFMVC"""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not all_series:
        print("No data to save!")
        return
    
    # Get all view names from first series
    first_key = list(all_series.keys())[0]
    view_names = list(all_series[first_key].keys())
    print(f"\n📋 View names: {view_names}")
    
    # Create a dictionary to store data for each view
    view_data_dict = {view: [] for view in view_names}
    series_names = []
    
    # Collect data for each series
    for series_name, views in all_series.items():
        series_names.append(series_name)
        for view in view_names:
            if view in views:
                view_data_dict[view].append(views[view])
            else:
                view_data_dict[view].append([])
    
    # Pad and save each view
    for view, data_list in view_data_dict.items():
        max_len = max(len(d) for d in data_list) if data_list else 0
        
        padded_data = []
        for d in data_list:
            if len(d) < max_len:
                padded = d + [np.nan] * (max_len - len(d))
            else:
                padded = d
            padded_data.append(padded)
        
        output_file = output_path / f"{view}.csv"
        np.savetxt(output_file, padded_data, delimiter=',', fmt='%.6f')
        print(f"Saved {view}: {len(data_list)} series, max length {max_len}")
    
    # Save series names
    with open(output_path / "series_names.txt", 'w') as f:
        for name in series_names:
            f.write(f"{name}\n")
    
    print(f"\n✅ Data saved to {output_path}")

def main():
    print("="*60)
    print("HAR Dataset Preparation for OFMVC")
    print("="*60)
    
    # Path to HAR dataset
    har_path = 'data/har/raw/UCI HAR Dataset'
    
    # Check if dataset exists
    if not os.path.exists(har_path):
        print(f"❌ HAR dataset not found at {har_path}")
        print("Please place the HAR dataset in data/har/raw/")
        return
    
    # Load and prepare data
    all_series = load_inertial_signals(har_path, max_samples=500)
    
    if all_series:
        # Save prepared data
        output_path = 'data/har/processed'
        save_for_ofmvc(all_series, output_path)
        
        print("\n" + "="*60)
        print("✅ HAR data preparation complete!")
        print(f"📊 Total series: {len(all_series)}")
        print(f"📁 Output folder: {output_path}")
        print("="*60)
    else:
        print("❌ No data was prepared. Check your HAR dataset structure.")

if __name__ == "__main__":
    from pathlib import Path
    main()
