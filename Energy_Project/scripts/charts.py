import matplotlib.pyplot as plt
import numpy as np

algorithms = ['K-Means', 'FMVC', 'OFMVC', 'kShape']
smape = [30.84, 30.89, 31.46, 31.57]
colors = ['green', 'blue', 'orange', 'red']

plt.figure(figsize=(10, 6))
bars = plt.bar(algorithms, smape, color=colors)
plt.ylabel('Test SMAPE (%)', fontsize=12)
plt.title('Forecasting Accuracy by Clustering Algorithm', fontsize=14)
plt.ylim(30, 32)

for bar, value in zip(bars, smape):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
             f'{value}%', ha='center', va='bottom', fontsize=11)

plt.grid(True, alpha=0.3, axis='y')
plt.savefig('forecasting_comparison.png', dpi=150)
print("Saved: forecasting_comparison.png")