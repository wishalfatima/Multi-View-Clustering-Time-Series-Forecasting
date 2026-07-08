"""
PAMAP2 Experiment with OFMVC - Fixed for k=12 with Plotting
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from FuzzyClustering.main_clustering import main_clustering
from FuzzyClustering.Find_Neighbors import Find_Neighbors
from FuzzyClustering.Center_Points import center_Points

# Views (9 sensor views)
VIEWS = [
    'hand_accel', 'hand_gyro', 'hand_mag',
    'chest_accel', 'chest_gyro', 'chest_mag',
    'ankle_accel', 'ankle_gyro', 'ankle_mag',
]

def load_pamap2_data(view_names, max_samples=None):
    """Load PAMAP2 data"""
    data_list = []
    
    for view in view_names:
        file_path = f'data/pamap2_processed/{view}.csv'
        data = pd.read_csv(file_path, header=None).values
        
        if max_samples:
            data = data[:max_samples]
        
        data_list.append(data)
        print(f"  Loaded {view}: {data.shape}")
    
    return data_list

def get_true_labels(max_samples=None):
    labels = np.loadtxt('data/pamap2_processed/labels.txt', dtype=int)
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
    data_list = load_pamap2_data(view_subset, max_samples)
    n_views = len(data_list)
    row = data_list[0].shape[0]
    col = data_list[0].shape[1]
    
    print(f"  Samples: {row}, Features: {col}, Views: {n_views}, Clusters: {k}")
    
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
    
    # Initialize centers (SEPARATE CENTERS PER VIEW)
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
        return None

def plot_results(results):
    """Plot clustering results"""
    successful = [r for r in results if r is not None and r["success"]]
    
    if not successful:
        print("No successful results to plot")
        return
    
    views = [r["views"] for r in successful]
    nmi = [r["nmi"] for r in successful]
    acc = [r["acc"] for r in successful]
    ari = [r["ari"] for r in successful]

    plt.figure(figsize=(10, 6))
    
    plt.plot(views, acc, 'o-', label='Accuracy', linewidth=2, markersize=8, color='blue')
    plt.plot(views, nmi, 's-', label='NMI', linewidth=2, markersize=8, color='orange')
    plt.plot(views, ari, '^-', label='ARI', linewidth=2, markersize=8, color='green')

    plt.xlabel("Number of Views", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("PAMAP2 - OFMVC Performance vs Number of Views", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(views)
    
    plt.tight_layout()
    plt.savefig("pamap2_ofmvc_results.png", dpi=150)
    print("\n✅ Graph saved as 'pamap2_ofmvc_results.png'")
    
    # Also save as PDF for report
    plt.savefig("pamap2_ofmvc_results.pdf", dpi=150)
    print("✅ Graph saved as 'pamap2_ofmvc_results.pdf'")
    
    plt.show()

def main():
    print("="*60)
    print("🎯 PAMAP2 - OFMVC Experiment")
    print("="*60)
    
    true_labels, activities = get_true_labels()
    print(f"\n📊 Dataset: {len(true_labels)} samples, {len(activities)} activities")
    print(f"   Activities: {activities}")
    
    # Experiments (adding views incrementally)
    experiments = [
        ("3 views (hand accel,gyro,mag)", VIEWS[:3]),
        ("6 views (+ chest accel,gyro,mag)", VIEWS[:6]),
        ("9 views (all sensors)", VIEWS[:9]),
    ]
    
    MAX_SAMPLES = 2000
    
    results = []
    for name, views in experiments:
        res = run_experiment(views, name, k=12, max_samples=MAX_SAMPLES)
        if res:
            results.append(res)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Experiment':<35} {'Views':<8} {'Accuracy':<12} {'NMI':<12} {'ARI':<12}")
    print("-"*75)
    for r in results:
        print(f"{r['name']:<35} {r['views']:<8} {r['acc']:.4f}      {r['nmi']:.4f}      {r['ari']:.4f}")
    print("="*60)
    
    # Plot results
    plot_results(results)

if __name__ == "__main__":
    main()