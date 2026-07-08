"""
MHEALTH Experiment with OFMVC - Time Series Version
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

# Views (23 sensor channels)
VIEWS = [
    'chest_acc_x', 'chest_acc_y', 'chest_acc_z',
    'chest_ecg_1', 'chest_ecg_2',
    'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z',
    'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
    'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z',
    'arm_acc_x', 'arm_acc_y', 'arm_acc_z',
    'arm_gyro_x', 'arm_gyro_y', 'arm_gyro_z',
    'arm_mag_x', 'arm_mag_y', 'arm_mag_z',
]

def load_mhealth_data(view_names, max_samples=None):
    """Load MHEALTH time series data"""
    data_list = []
    
    for view in view_names:
        file_path = f'data/mhealth_processed/{view}.csv'
        data = pd.read_csv(file_path, header=None).values
        
        if max_samples:
            data = data[:max_samples]
        
        data_list.append(data)
        print(f"  Loaded {view}: {data.shape}")
    
    return data_list

def get_true_labels(max_samples=None):
    labels = np.loadtxt('data/mhealth_processed/labels.txt', dtype=int)
    if max_samples:
        labels = labels[:max_samples]
    
    # Remap to 0..n-1
    unique = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique)}
    labels = np.array([label_map[l] for l in labels])
    
    return labels, unique

def run_experiment(view_subset, name, k=12, max_samples=2000):
    print("\n" + "="*60)
    print(f"📊 {name}")
    print("="*60)
    
    # Load data
    print("\n📂 Loading data...")
    data_list = load_mhealth_data(view_subset, max_samples)
    n_views = len(data_list)
    row = data_list[0].shape[0]
    col = data_list[0].shape[1]
    
    print(f"  Samples: {row}, Time points: {col}, Views: {n_views}")
    
    # Normalize
    print("\n🔄 Normalizing...")
    scaler = MinMaxScaler()
    normalized = []
    for i, d in enumerate(data_list):
        norm = scaler.fit_transform(d)
        normalized.append(norm)
        print(f"  View {i}: {norm.shape}")
    
    # Parameters
    t_max = 10
    NR = 5
    alpha1 = 0.5
    q = 2
    gama = 1
    landa = np.array([0.5] * n_views)
    viewpoint = np.ones(n_views)
    
    # Neighbors
    print("\n🔍 Finding neighbors...")
    Neig, dm = Find_Neighbors(NR, normalized, landa, n_views)
    
    # Initialize centers
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
        return {"name": name, "views": n_views, "success": False}

def main():
    print("="*60)
    print("🎯 MHEALTH - OFMVC (Time Series Version)")
    print("="*60)
    
    true_labels, activities = get_true_labels()
    print(f"\n📊 Dataset: {len(true_labels)} samples, {len(activities)} activities")
    
    # Experiments
    experiments = [
        ("3 views (chest acc)", VIEWS[:3]),
        ("6 views (chest + ecg)", VIEWS[:6]),
        ("9 views (+ ankle acc)", VIEWS[:9]),
        ("12 views (+ ankle gyro)", VIEWS[:12]),
        ("23 views (all sensors)", VIEWS[:23]),
    ]
    
    MAX_SAMPLES = 2000  # 2000 samples for faster testing
    
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
    print("="*60)

if __name__ == "__main__":
    main()