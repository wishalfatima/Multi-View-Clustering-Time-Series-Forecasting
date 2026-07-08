"""
Run WISDM experiments with increasing number of views
Fixed version - properly handles NaN and different lengths
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Define the 12 views in order
VIEWS = [
    'phone_accel_x', 'phone_accel_y', 'phone_accel_z',
    'phone_gyro_x', 'phone_gyro_y', 'phone_gyro_z',
    'watch_accel_x', 'watch_accel_y', 'watch_accel_z',
    'watch_gyro_x', 'watch_gyro_y', 'watch_gyro_z'
]

def load_and_preprocess_wisdm_views(view_names):
    """Load, clean, and make all views the same length"""
    data_list = []
    
    for view in view_names:
        file_path = f'data/wisdm_processed/{view}.csv'
        df = pd.read_csv(file_path, header=None)
        
        # Step 1: Replace NaN with 0 (this fixes the NaN error)
        df = df.fillna(0)
        
        # Step 2: Convert to numpy
        data = df.values
        
        data_list.append(data)
        print(f"  Loaded {view}: shape {data.shape}")
    
    # Step 3: Find minimum length across ALL views
    min_len = min(d.shape[1] for d in data_list)
    print(f"\n  📏 Truncating all views to minimum length: {min_len}")
    
    # Step 4: Truncate all to same length
    truncated_data = [d[:, :min_len] for d in data_list]
    
    return truncated_data

def get_true_labels():
    """Get true activity labels"""
    series_names = pd.read_csv('data/wisdm_processed/series_names.txt', 
                                header=None, 
                                names=['series_name'])
    activities = series_names['series_name'].apply(lambda x: x.split('_')[-1])
    unique_activities = activities.unique()
    activity_to_label = {act: i for i, act in enumerate(unique_activities)}
    true_labels = activities.map(activity_to_label).values
    return true_labels, unique_activities

def run_experiment(view_subset, experiment_name):
    """Run clustering on a subset of views"""
    
    print(f"\n{'='*60}")
    print(f"📊 {experiment_name}")
    print(f"   Number of views: {len(view_subset)}")
    print('='*60)
    
    # 1. Load and preprocess data
    print("\n📂 Loading and preprocessing data...")
    data_list = load_and_preprocess_wisdm_views(view_subset)
    
    # 2. Normalize each view
    print("\n🔄 Normalizing data...")
    scaler = MinMaxScaler()
    normalized_data = []
    for i, view_data in enumerate(data_list):
        normalized = scaler.fit_transform(view_data)
        normalized_data.append(normalized)
        print(f"  View {i+1}: normalized shape {normalized.shape}")
    
    # 3. Combine all views into one feature matrix
    print("\n🔗 Combining all views into one feature matrix...")
    combined_features = np.hstack(normalized_data)
    print(f"  Combined shape: {combined_features.shape}")
    
    # 4. Simple K-Means clustering (since MVC code is complex)
    print("\n🔮 Running K-Means clustering...")
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    # Number of clusters = number of unique activities (18)
    n_clusters = len(get_true_labels()[1])
    print(f"  Number of clusters: {n_clusters}")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(combined_features)
    
    # Get true labels for evaluation
    true_labels, _ = get_true_labels()
    
    # Calculate metrics
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    
    print(f"\n  📊 Results:")
    print(f"     Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"     Normalized Mutual Info (NMI): {nmi:.4f}")
    
    return {
        'name': experiment_name,
        'n_views': len(view_subset),
        'ari': ari,
        'nmi': nmi
    }

def main():
    print("="*60)
    print("🎯 WISDM Multi-View Clustering Experiment")
    print("Testing how clustering performance changes with more views")
    print("="*60)
    
    # Get dataset info
    true_labels, activities = get_true_labels()
    print(f"\n📊 Dataset Info:")
    print(f"   Total series: {len(true_labels)}")
    print(f"   Total activities: {len(activities)}")
    print(f"   Activities: {list(activities)[:10]}...")
    print("="*60)
    
    # Define experiments (adding views incrementally)
    experiments = [
        ("1 view (phone accel X)", VIEWS[0:1]),
        ("3 views (phone accel X,Y,Z)", VIEWS[0:3]),
        ("6 views (all phone sensors)", VIEWS[0:6]),
        ("9 views (phone + watch accel)", VIEWS[0:9]),
        ("12 views (all sensors)", VIEWS[0:12]),
    ]
    
    # Run experiments
    results = []
    for name, view_subset in experiments:
        result = run_experiment(view_subset, name)
        results.append(result)
    
    # Print summary table
    print("\n" + "="*60)
    print("📊 SUMMARY OF RESULTS")
    print("="*60)
    print(f"{'Experiment':<35} {'Views':<8} {'ARI':<10} {'NMI':<10}")
    print("-"*60)
    for r in results:
        print(f"{r['name']:<35} {r['n_views']:<8} {r['ari']:.4f}      {r['nmi']:.4f}")
    print("="*60)
    
    # Plot results
    try:
        import matplotlib.pyplot as plt
        n_views = [r['n_views'] for r in results]
        ari_scores = [r['ari'] for r in results]
        nmi_scores = [r['nmi'] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(n_views, ari_scores, 'o-', label='ARI', linewidth=2, markersize=8, color='blue')
        plt.plot(n_views, nmi_scores, 's-', label='NMI', linewidth=2, markersize=8, color='orange')
        plt.xlabel('Number of Views')
        plt.ylabel('Score')
        plt.title('Clustering Performance vs Number of Views (WISDM Dataset)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(n_views)
        plt.savefig('wisdm_clustering_results.png', dpi=150)
        print("\n📈 Plot saved as 'wisdm_clustering_results.png'")
    except:
        print("\n⚠️ Matplotlib not available, skipping plot")
    
    print("\n✅ Experiment complete!")

if __name__ == "__main__":
    main()