"""
WISDM Experiment with OFMVC (Fixed Version)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
import os
import sys

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import OFMVC modules
from FuzzyClustering.main_clustering import main_clustering
from FuzzyClustering.Find_Neighbors import Find_Neighbors
from FuzzyClustering.Center_Points import center_Points
from FuzzyClustering.calculateMetrics import calculateMetrics

# -----------------------------
# VIEWS (12 sensor channels)
# -----------------------------
VIEWS = [
    'phone_accel_x', 'phone_accel_y', 'phone_accel_z',
    'phone_gyro_x', 'phone_gyro_y', 'phone_gyro_z',
    'watch_accel_x', 'watch_accel_y', 'watch_accel_z',
    'watch_gyro_x', 'watch_gyro_y', 'watch_gyro_z'
]


# -----------------------------
# LOAD DATA
# -----------------------------
def load_wisdm_data(view_names, max_series=None):
    data_list = []

    for view in view_names:
        file_path = f'data/wisdm_processed/{view}.csv'

        df = pd.read_csv(file_path, header=None)
        df = df.fillna(0)

        data = df.values.astype(float)

        if max_series:
            data = data[:max_series]

        data_list.append(data)
        print(f"  Loaded {view}: {data.shape}")

    # Ensure equal length
    min_len = min(d.shape[1] for d in data_list)
    data_list = [d[:, :min_len] for d in data_list]

    print(f"  Truncated all views to length: {min_len}")

    return data_list


# -----------------------------
# LABELS
# -----------------------------
def get_true_labels(max_series=None):
    series_names = pd.read_csv(
        'data/wisdm_processed/series_names.txt',
        header=None,
        names=['series_name']
    )

    activities = series_names['series_name'].apply(lambda x: x.split('_')[-1])
    unique = activities.unique()

    label_map = {act: i for i, act in enumerate(unique)}
    labels = activities.map(label_map).values

    if max_series:
        labels = labels[:max_series]

    return labels, unique


# -----------------------------
# MAIN EXPERIMENT
# -----------------------------
def run_ofmvc_experiment(view_subset, name, k=18, max_series=200):

    print("\n" + "="*60)
    print(f"📊 {name}")
    print("="*60)

    # 1. Load
    print("\n📂 Loading data...")
    data_list = load_wisdm_data(view_subset, max_series)

    n_views = len(data_list)
    row = data_list[0].shape[0]

    # 2. Normalize
    print("\n🔄 Normalizing...")
    scaler = MinMaxScaler()

    normalized_data = []
    for i, d in enumerate(data_list):
        norm = scaler.fit_transform(d)
        normalized_data.append(norm)
        print(f"  View {i}: {norm.shape}")

    normalized_data = [np.array(v) for v in normalized_data]

    # 3. Parameters
    t_max = 10
    NR = 5
    alpha1 = 0.5
    q = 2
    gama = 1
    landa = np.array([0.5] * n_views)
    viewpoint = np.ones(n_views)

    # 4. Neighbors
    print("\n🔍 Finding neighbors...")
    Neig, dm = Find_Neighbors(NR, normalized_data, landa, n_views)

    # 5. Initialize centers
    print("\n🎯 Initializing centers...")
    center_points = []
    for i in range(n_views):
        dm_i = cdist(normalized_data[i], normalized_data[i])
        centers, _ = center_Points(
            normalized_data[i],
            dm_i,
            k,
            normalized_data[i].shape[1],
            row,
            landa[i]
        )
        center_points.append(centers)
        print(f"  View {i} centers: {centers.shape}")

    # 6. Run clustering
    print("\n🔮 Running OFMVC...")

    try:
        Final = main_clustering(
            viewpoint=viewpoint,
            data=normalized_data,
            center_points=center_points,
            k=k,
            t_max=t_max,
            NR=NR,
            alpha1=alpha1,
            landa=landa,
            q=q,
            gama=gama,
            Neig=Neig,
            number_viwe=n_views,
            row=row,
            sample_weight=np.ones(row)
        )

        print("\n✅ Converged")
        
        # Get cluster assignments (Final is a weighted sum matrix)
        # Each column represents a series, each row represents a cluster
        # The cluster with highest value is the assigned cluster
        Cluster_elem = np.argmax(Final, axis=0)
        
        # Get true labels
        true_labels, _ = get_true_labels(max_series)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, normalized_mutual_info_score, f1_score, adjusted_rand_score
        
        acc = accuracy_score(true_labels, Cluster_elem)
        nmi = normalized_mutual_info_score(true_labels, Cluster_elem)
        f1 = f1_score(true_labels, Cluster_elem, average='weighted')
        ari = adjusted_rand_score(true_labels, Cluster_elem)
        
        print(f"\n📈 Results:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  NMI: {nmi:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print(f"  ARI: {ari:.4f}")
        
        return {"name": name, "views": n_views, "acc": acc, "nmi": nmi, "f1": f1, "ari": ari, "success": True}
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"name": name, "views": n_views, "success": False}


# -----------------------------
# MAIN
# -----------------------------
def main():

    print("="*60)
    print("🎯 WISDM + OFMVC Experiment")
    print("Testing how clustering performance changes with more views")
    print("="*60)

    # Get dataset info
    true_labels, activities = get_true_labels()
    print(f"\n📊 Dataset: {len(true_labels)} series, {len(activities)} activities")
    print(f"   Activities: {list(activities)[:10]}...")
    print("="*60)

    # Experiments (adding views incrementally)
    # Skip 1-view because it has a bug in the original code (beta variable not defined)
    experiments = [
        ("3 views (phone accel X,Y,Z)", VIEWS[:3]),
        ("6 views (all phone sensors)", VIEWS[:6]),
        ("9 views (phone + watch accel)", VIEWS[:9]),
        ("12 views (all sensors)", VIEWS[:12]),
    ]

    # Use 200 series for faster testing
    MAX_SERIES = 200

    results = []
    for name, views in experiments:
        res = run_ofmvc_experiment(views, name, max_series=MAX_SERIES)
        results.append(res)

    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY OF OFMVC RESULTS")
    print("="*60)
    print(f"{'Experiment':<35} {'Views':<8} {'Accuracy':<12} {'NMI':<12} {'F1':<12} {'ARI':<12}")
    print("-"*85)
    for r in results:
        if r["success"]:
            print(f"{r['name']:<35} {r['views']:<8} {r['acc']:.4f}      {r['nmi']:.4f}      {r['f1']:.4f}      {r['ari']:.4f}")
        else:
            print(f"{r['name']:<35} {r['views']:<8} {'FAILED':<12} {'FAILED':<12} {'FAILED':<12} {'FAILED':<12}")
    print("="*60)

    # Plot results
    successful = [r for r in results if r["success"]]
    if successful:
        try:
            import matplotlib.pyplot as plt
            n_views = [r['views'] for r in successful]
            acc = [r['acc'] for r in successful]
            nmi = [r['nmi'] for r in successful]
            ari = [r['ari'] for r in successful]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].plot(n_views, acc, 'o-', linewidth=2, markersize=8, color='blue')
            axes[0].set_xlabel('Number of Views')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_title('Accuracy vs Number of Views')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_xticks(n_views)
            
            axes[1].plot(n_views, nmi, 's-', linewidth=2, markersize=8, color='orange')
            axes[1].set_xlabel('Number of Views')
            axes[1].set_ylabel('NMI')
            axes[1].set_title('NMI vs Number of Views')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xticks(n_views)
            
            axes[2].plot(n_views, ari, '^-', linewidth=2, markersize=8, color='green')
            axes[2].set_xlabel('Number of Views')
            axes[2].set_ylabel('ARI')
            axes[2].set_title('ARI vs Number of Views')
            axes[2].grid(True, alpha=0.3)
            axes[2].set_xticks(n_views)
            
            plt.tight_layout()
            plt.savefig('wisdm_ofmvc_results.png', dpi=150)
            print("\n📈 Plot saved as 'wisdm_ofmvc_results.png'")
        except Exception as e:
            print(f"\nPlot error: {e}")

    print("\n✅ Complete!")


if __name__ == "__main__":
    main()