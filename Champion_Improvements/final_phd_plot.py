import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
ANALYSIS_DIR = Path("C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/GA_Mutation_Study/analysis")
FIGURES_DIR = Path("C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/Champion_Improvements/final_results")
FIGURES_DIR.mkdir(exist_ok=True)

def get_mean_curve(part, mu_val, dataset):
    all_scores = []
    for seed in [42, 43, 44]:
        file_path = ANALYSIS_DIR / f"{part}_mu_{mu_val}_{dataset}_seed_{seed}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            all_scores.append(df.set_index('iteration')['accuracy'])
    
    if not all_scores:
        return None
    
    # Average across seeds
    return pd.concat(all_scores, axis=1).mean(axis=1)

# Plotting
plt.figure(figsize=(12, 7))

dataset_colors = {'ham': 'Blues', 'isic': 'Oranges'}
mu_values = [0.0, 0.04, 0.12]
mu_labels = {0.0: 'No Mutation (Baseline)', 0.04: 'Standard Mutation', 0.12: 'High Mutation (PhD Champion Exploration)'}

# Colors matching the "rainbow" look of the user's image
colors_ham = ['#1f77b4', '#3498db', '#5dade2'] # Different shades of blue
colors_isic = ['#d35400', '#e67e22', '#f39c12'] # Different shades of orange

for i, mu in enumerate(mu_values):
    # HAM Curves
    ham_curve = get_mean_curve('PartB2', mu, 'ham')
    if ham_curve is not None:
        plt.plot(ham_curve.index / 1000, ham_curve.values, label=f"HAM: {mu_labels[mu]}", color=colors_ham[i], lw=2.5, marker='o', markersize=4)

    # ISIC Curves
    isic_curve = get_mean_curve('PartB2', mu, 'isic')
    if isic_curve is not None:
        plt.plot(isic_curve.index / 1000, isic_curve.values, label=f"ISIC: {mu_labels[mu]}", color=colors_isic[i], lw=2.5, marker='s', markersize=4)

# Formatting to match PhD standard
plt.xlabel('Iterations (x 1,000)', fontsize=14, fontweight='bold')
plt.ylabel('Balanced Accuracy', fontsize=14, fontweight='bold')
plt.title('Performance Plateau Analysis: The Effect of Mutation on Learning Stability', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right', frameon=True, fontsize=10, shadow=True)
plt.ylim(0.45, 0.95)
plt.xlim(0, 510)

# Add highlight for the PhD Champion range
plt.axvspan(200, 500, color='gray', alpha=0.1, label='Convergence Zone')

save_path = FIGURES_DIR / "phd_evolution_plot.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ Success: PhD Evolution Plot saved to {save_path}")
