"""
Preprocessing with interpolation (instead of zeros) + Z‑normalization.
Keeps all customers (35k), no rows dropped.
"""

import pandas as pd
import numpy as np
import os

INPUT = 'data/raw/'
OUTPUT = 'data/processed/'
os.makedirs(OUTPUT, exist_ok=True)

# Load smartmeter to get reference date columns (365 days)
smart23 = pd.read_csv(INPUT + 'sample_23_20percent.csv')
date_cols = smart23.columns[1:].tolist()

# Files to process (2023 views)
files = {
    'smartmeter': 'sample_23_20percent.csv',
    'avgtemp': 'sample_avgtemp_23_20percent.csv',
    'maxtemp': 'sample_maxtemp_23_20percent.csv',
    'mintemp': 'sample_mintemp_23_20percent.csv',
    'maxgsun': 'sample_maxGSun_23_20percent.csv',
    'sumgsun': 'sample_sumGSun_23_20percent.csv',
}

def interpolate_and_z_norm(df, is_weather=False):
    """
    Interpolate missing values (row‑wise) then Z‑normalize per customer.
    """
    # Align columns to smartmeter dates
    keep_cols = ['ID'] + [c for c in df.columns[1:] if c in date_cols or c.split()[0] in date_cols]
    df_aligned = df[keep_cols]
    # Separate ID and data
    ids = df_aligned['ID']
    data = df_aligned.drop(columns=['ID'])
    # Interpolate across days (axis=1) – forward, backward, linear
    data = data.interpolate(method='linear', axis=1, limit_direction='both')
    # Any remaining NaN (e.g., at edges) fill with 0 (rare after interpolation)
    data = data.fillna(0)
    # Z‑normalize each row
    def z_norm(row):
        mean = row.mean()
        std = row.std()
        if std == 0:
            return row - mean
        return (row - mean) / std
    data_norm = data.apply(z_norm, axis=1)
    # Reattach IDs
    result = pd.concat([ids, data_norm], axis=1)
    return result

# Process each view
for name, fname in files.items():
    print(f"Processing {name}...")
    df = pd.read_csv(INPUT + fname)
    df_processed = interpolate_and_z_norm(df, is_weather=(name != 'smartmeter'))
    # Take first 5000 customers (or keep all 35k – your choice)
    df_5000 = df_processed.iloc[:5000]
    out_file = OUTPUT + f'sample_{name}_5k_interp_z.csv'
    df_5000.to_csv(out_file, index=False)
    print(f"Saved {out_file} – shape {df_5000.shape}")

# Also process 2024 smartmeter (for forecasting)
smart24 = pd.read_csv(INPUT + 'sample_24_20percent.csv')
df_24 = interpolate_and_z_norm(smart24, is_weather=False)
df_24_5000 = df_24.iloc[:5000]
out_24 = OUTPUT + 'sample_24_20percent_5k_interp_z.csv'
df_24_5000.to_csv(out_24, index=False)
print(f"Saved {out_24} – shape {df_24_5000.shape}")

print("Preprocessing complete. All missing values interpolated, then Z‑normalized.")