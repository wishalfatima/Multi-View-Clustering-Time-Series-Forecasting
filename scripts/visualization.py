"""


Visualizes the relationship between weather (average temperature) and
electricity consumption per cluster — as requested by Peter.

Uses:
- OFMVC cluster labels (best performing algorithm)
- Raw smartmeter data (2023) for consumption
- Raw average temperature data (2023) for weather

Saves plots to: results/figures/

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── paths ──────────────────────────────────────────────────────────────────────
RAW_SMART   = '/Users/wishal/Documents/P1_Project/Energy_Project/data/sample_23_20percent.csv'
RAW_AVGTEMP = '/Users/wishal/Documents/P1_Project/Energy_Project/data/sample_avgtemp_23_20percent.csv'
CLUSTER_FILE = 'results/clusters/ofmvc_clusters_final.csv'
OUT_DIR      = 'results/figures'

os.makedirs(OUT_DIR, exist_ok=True)

N_CUSTOMERS = 5000

# ── load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
smart_raw = pd.read_csv(RAW_SMART)
temp_raw  = pd.read_csv(RAW_AVGTEMP)
clusters  = pd.read_csv(CLUSTER_FILE)

# align to 5000 customers
smart_raw = smart_raw.iloc[:N_CUSTOMERS]
temp_raw  = temp_raw.iloc[:N_CUSTOMERS]

# get numeric columns only (drop ID)
smart_vals = smart_raw.drop(columns=['ID'], errors='ignore').values.astype(float)  # (5000, 365)
temp_vals  = temp_raw.drop(columns=['ID'], errors='ignore').values.astype(float)   # (5000, 366)

# Trim both to the same number of days (365)
n_days = min(smart_vals.shape[1], temp_vals.shape[1])
smart_vals = smart_vals[:, :n_days]
temp_vals  = temp_vals[:, :n_days]

cluster_labels = clusters['cluster'].values.astype(int)

print(f"Smart meter shape : {smart_vals.shape}")
print(f"Temperature shape : {temp_vals.shape}")
print(f"Cluster distribution: {np.bincount(cluster_labels)}")

# ── get unique non-empty clusters ──────────────────────────────────────────────
unique_clusters = sorted([c for c in np.unique(cluster_labels) if np.sum(cluster_labels == c) > 0])
n_clusters = len(unique_clusters)
print(f"Non-empty clusters: {unique_clusters}")

# day axis (1 to 365)
days = np.arange(1, smart_vals.shape[1] + 1)

# ── PLOT 1: Average consumption per cluster over time ─────────────────────────
print("\nGenerating Plot 1: Average consumption per cluster over time...")
fig, ax = plt.subplots(figsize=(14, 6))
colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

for idx, cluster_id in enumerate(unique_clusters):
    mask = cluster_labels == cluster_id
    n_in_cluster = np.sum(mask)
    avg_consumption = np.nanmean(smart_vals[mask], axis=0)
    ax.plot(days, avg_consumption, color=colors[idx], linewidth=1.5,
            label=f'Cluster {cluster_id} (n={n_in_cluster})')

ax.set_title('Average Electricity Consumption per Cluster (2023)', fontsize=14, fontweight='bold')
ax.set_xlabel('Day of Year', fontsize=12)
ax.set_ylabel('Average Consumption (kWh)', fontsize=12)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/consumption_per_cluster.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUT_DIR}/consumption_per_cluster.png")

# ── PLOT 2: Average temperature per cluster over time ─────────────────────────
print("Generating Plot 2: Average temperature per cluster over time...")
fig, ax = plt.subplots(figsize=(14, 6))

for idx, cluster_id in enumerate(unique_clusters):
    mask = cluster_labels == cluster_id
    n_in_cluster = np.sum(mask)
    avg_temp = np.nanmean(temp_vals[mask], axis=0)
    ax.plot(days, avg_temp, color=colors[idx], linewidth=1.5,
            label=f'Cluster {cluster_id} (n={n_in_cluster})')

ax.set_title('Average Temperature per Cluster (2023)', fontsize=14, fontweight='bold')
ax.set_xlabel('Day of Year', fontsize=12)
ax.set_ylabel('Average Temperature (°C)', fontsize=12)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/temperature_per_cluster.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUT_DIR}/temperature_per_cluster.png")

# ── PLOT 3: Combined dual-axis plot (consumption + temperature) per cluster ────
print("Generating Plot 3: Combined weather vs consumption per cluster...")
fig = plt.figure(figsize=(16, 4 * n_clusters))
gs = gridspec.GridSpec(n_clusters, 1, hspace=0.5)

for idx, cluster_id in enumerate(unique_clusters):
    mask = cluster_labels == cluster_id
    n_in_cluster = np.sum(mask)
    avg_consumption = np.nanmean(smart_vals[mask], axis=0)
    avg_temp = np.nanmean(temp_vals[mask], axis=0)

    ax1 = fig.add_subplot(gs[idx])
    ax2 = ax1.twinx()

    line1, = ax1.plot(days, avg_consumption, color='steelblue', linewidth=1.5,
                      label='Avg Consumption (kWh)')
    line2, = ax2.plot(days, avg_temp, color='tomato', linewidth=1.5,
                      linestyle='--', label='Avg Temperature (°C)')

    ax1.set_ylabel('Consumption (kWh)', color='steelblue', fontsize=10)
    ax2.set_ylabel('Temperature (°C)', color='tomato', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='tomato')
    ax1.set_title(f'Cluster {cluster_id} — n={n_in_cluster} customers', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Day of Year', fontsize=10)
    ax1.grid(True, alpha=0.2)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=9)

fig.suptitle('Weather (Temperature) vs. Electricity Consumption per Cluster\n(OFMVC Clustering, 2023)',
             fontsize=14, fontweight='bold', y=1.01)
plt.savefig(f'{OUT_DIR}/weather_vs_consumption_per_cluster.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUT_DIR}/weather_vs_consumption_per_cluster.png")

# ── PLOT 4: Scatter — daily mean temp vs daily mean consumption per cluster ────
print("Generating Plot 4: Scatter plot temperature vs consumption per cluster...")
fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 5), sharey=False)
if n_clusters == 1:
    axes = [axes]

for idx, cluster_id in enumerate(unique_clusters):
    mask = cluster_labels == cluster_id
    avg_consumption = np.nanmean(smart_vals[mask], axis=0)  # (365,)
    avg_temp = np.nanmean(temp_vals[mask], axis=0)           # (365,)

    axes[idx].scatter(avg_temp, avg_consumption, alpha=0.5, s=20,
                      color=colors[idx], edgecolors='none')

    # fit a trend line
    valid = ~(np.isnan(avg_temp) | np.isnan(avg_consumption))
    if valid.sum() > 2:
        z = np.polyfit(avg_temp[valid], avg_consumption[valid], 1)
        p = np.poly1d(z)
        temp_sorted = np.sort(avg_temp[valid])
        axes[idx].plot(temp_sorted, p(temp_sorted), color='black',
                       linewidth=1.5, linestyle='--', label=f'Trend')

    n_in_cluster = np.sum(mask)
    axes[idx].set_title(f'Cluster {cluster_id}\n(n={n_in_cluster})', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Avg Temperature (°C)', fontsize=10)
    axes[idx].set_ylabel('Avg Consumption (kWh)', fontsize=10)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend(fontsize=9)

fig.suptitle('Temperature vs. Electricity Consumption per Cluster (Daily Averages, 2023)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/scatter_temp_vs_consumption.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUT_DIR}/scatter_temp_vs_consumption.png")

print("\n" + "="*60)
print("ALL PLOTS SAVED SUCCESSFULLY")
print("="*60)
print(f"Output folder: {OUT_DIR}/")
print("  1. consumption_per_cluster.png")
print("  2. temperature_per_cluster.png")
print("  3. weather_vs_consumption_per_cluster.png  ← most important for report")
print("  4. scatter_temp_vs_consumption.png          ← shows relationship clearly")