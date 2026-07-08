"""
HAR Experiment with OFMVC - Using MHEALTH/WISDM Pattern
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from FuzzyClustering.main_clustering import main_clustering
from FuzzyClustering.Find_Neighbors import Find_Neighbors
from FuzzyClustering.Center_Points import center_Points

# Views (9 sensor channels)
VIEWS = [
    'body_acc_x', 'body_acc_y', 'body_acc_z',
    'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
    'total_acc_x', 'total_acc_y', 'total_acc_z'
]

def load_har_data(view_names, max_samples=None):
    """Load HAR time series data"""
    data_list = []
    
    for view in view_names:
        file_path = f'data/har/processed/{view}.csv'
        data = pd.read_csv(file_path, header=None).values
        
        if max_samples:
            data = data[:max_samples]
        
        data_list.append(data)
        print(f"  Loaded {view}: {data.shape}")
    
    return data_list

def get_true_labels(max_samples=None):
    labels = np.loadtxt('data/har/processed/labels.txt', dtype=int)
    if max_samples:
        labels = labels[:max_samples]
    
    # Remap to 0..n-1
    unique = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique)}
    labels = np.array([label_map[l] for l in labels])
    
    return labels, unique

def run_experiment(view_subset, name, k=6, max_samples=30):
    print("\n" + "="*60)
    print(f"📊 {name}")
    print("="*60)
    
    # Load data
    print("\n📂 Loading data...")
    data_list = load_har_data(view_subset, max_samples)
    n_views = len(data_list)
    row = data_list[0].shape[0]
    col = data_list[0].shape[1]
    
    print(f"  Samples: {row}, Features: {col}, Views: {n_views}")
    
    # Normalize
    print("\n🔄 Normalizing...")
    scaler = MinMaxScaler()
    normalized = []
    for i, d in enumerate(data_list):
        norm = scaler.fit_transform(d)
        normalized.append(norm)
        print(f"  View {i}: {norm.shape}")
    
    # Parameters
    t_max = 20
    NR = 5
    alpha1 = 0.5
    q = 2
    gama = 1
    landa = np.array([0.5] * n_views)
    viewpoint = np.ones(n_views)
    
    # Neighbors
    print("\n🔍 Finding neighbors...")
    Neig, dm = Find_Neighbors(NR, normalized, landa, n_views)
    
    # Initialize centers (SEPARATE CENTERS PER VIEW - KEY!)
    print("\n🎯 Initializing centers...")
    center_points = []
    for i in range(n_views):
        dm_i = cdist(normalized[i], normalized[i])
        centers, _ = center_Points(normalized[i], dm_i, k, col, row, landa[i])
        center_points.append(centers)
        print(f"  View {i} centers: {centers.shape}")
    
    # Run OFMVC
    print("\n🔮 Running OFMVC...")
    
    try:
        Final = main_clustering(
            viewpoint=viewpoint,
            data=normalized,
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
        
        # Get cluster assignments
        Cluster_elem = np.argmax(Final, axis=0)
        
        # Get true labels
        true_labels, _ = get_true_labels(max_samples)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
        
        acc = accuracy_score(true_labels, Cluster_elem)
        nmi = normalized_mutual_info_score(true_labels, Cluster_elem)
        ari = adjusted_rand_score(true_labels, Cluster_elem)
        
        print(f"\n📈 Results:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  NMI: {nmi:.4f}")
        print(f"  ARI: {ari:.4f}")
        
        return {"name": name, "views": n_views, "acc": acc, "nmi": nmi, "ari": ari, "success": True}
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"name": name, "views": n_views, "success": False}

def main():
    print("="*60)
    print("🎯 HAR - OFMVC Experiment")
    print("="*60)
    
    true_labels, activities = get_true_labels()
    print(f"\n📊 Dataset: {len(true_labels)} samples, {len(activities)} activities")
    print(f"   Activities: {activities}")
    
    # Experiments (adding views incrementally)
    experiments = [
        ("1 view (body_acc_x)", VIEWS[:1]),
        ("3 views (body_acc x,y,z)", VIEWS[:3]),
        ("6 views (+ body_gyro)", VIEWS[:6]),
        ("9 views (all sensors)", VIEWS[:9]),
    ]
    
    MAX_SAMPLES = 30
    
    results = []
    for name, views in experiments:
        res = run_experiment(views, name, max_samples=MAX_SAMPLES)
        results.append(res)
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"{'Experiment':<35} {'Views':<8} {'Accuracy':<12} {'NMI':<12} {'ARI':<12}")
    for r in results:
        if r["success"]:
            print(f"{r['name']:<35} {r['views']:<8} {r['acc']:.4f}      {r['nmi']:.4f}      {r['ari']:.4f}")
        else:
            print(f"{r['name']:<35} {r['views']:<8} {'FAILED':<12} {'FAILED':<12} {'FAILED':<12}")
    print("="*60)
    
    # ========== PLOTTING CODE (MOVED INSIDE MAIN) ==========
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        import matplotlib.pyplot as plt
        n_views = [r['views'] for r in successful_results]
        acc = [r['acc'] for r in successful_results]
        nmi = [r['nmi'] for r in successful_results]
        ari = [r['ari'] for r in successful_results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(n_views, acc, 'o-', label='Accuracy', linewidth=2, markersize=8)
        plt.plot(n_views, nmi, 's-', label='NMI', linewidth=2, markersize=8)
        plt.plot(n_views, ari, '^-', label='ARI', linewidth=2, markersize=8)
        plt.xlabel('Number of Views')
        plt.ylabel('Score')
        plt.title('HAR - OFMVC Performance vs Number of Views')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(n_views)
        plt.savefig('har_ofmvc_results.png', dpi=150)
        print("\n📈 Plot saved as 'har_ofmvc_results.png'")
    # =======================================================

if __name__ == "__main__":
    main()