import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

# Add ExSTraCS to path
CHAMPION_PATH = Path("C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/Derived_Features_Champion")
IMPROV_PATH = Path("C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/Champion_Improvements")
sys.path.insert(0, str(CHAMPION_PATH / "scikit-ExSTraCS-master"))

from skExSTraCS.ExSTraCS import ExSTraCS

# Paths
RESULTS_DIR = IMPROV_PATH / "final_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Configurations to Compare
configs = [
    {'name': 'PhD Champion (nu=5, mu=0.08)', 'N': 3000, 'nu': 5,  'mu': 0.08, 'color': '#1f77b4'},
    {'name': 'Legacy Baseline (nu=10, mu=0.04)', 'N': 3000, 'nu': 10, 'mu': 0.04, 'color': '#ff7f0e'},
    {'name': 'Low Exploration (mu=0.01)', 'N': 3000, 'nu': 10, 'mu': 0.01, 'color': '#2ca02c'},
    {'name': 'Reduced Pop (N=1500)', 'N': 1500, 'nu': 5,  'mu': 0.04, 'color': '#d62728'},
]

# Experimental Params
TOTAL_ITERATIONS = 100000
SAMPLE_INTERVAL = 5000
SEEDS = [42] # Use 1 seed for speed today, can expand for confidence intervals later

print("="*60)
print("GENERATING LEARNING CURVE COMPARISON")
print("="*60)

# Load Data
print("Loading Top 100 features...")
df_isic = pd.read_csv(IMPROV_PATH / "champion_top100_features.csv")
X = df_isic.drop('label', axis=1).values
y = df_isic['label'].values

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

plt.figure(figsize=(10, 6))

for config in configs:
    print(f"\nProcessing: {config['name']}...")
    all_scores = []
    
    # ExSTraCS doesn't have a partial_fit/callback in this version that returns accuracy easily
    # So we train in increments to simulate the learning curve
    
    iters_range = range(SAMPLE_INTERVAL, TOTAL_ITERATIONS + SAMPLE_INTERVAL, SAMPLE_INTERVAL)
    scores = []
    
    for iterations in iters_range:
        # Note: We re-train from scratch to each point to ensure consistency with Fit()
        # In a real run, this data is captured internally, but here we simulate it.
        # Speed hack: Use slightly fewer iterations for the early points
        model = ExSTraCS(
            learning_iterations=iterations,
            N=config['N'],
            nu=config['nu'],
            mu=config['mu'],
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = balanced_accuracy_score(y_val, y_pred)
        scores.append(acc)
        print(f"  {iterations} iters: {acc:.4f}")
    
    plt.plot(list(iters_range), scores, label=config['name'], color=config['color'], marker='o', markersize=4, lw=2)

# Finalize Plot
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Balanced Accuracy', fontsize=12)
plt.title('Performance Evolution Across LCS Configurations', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')
plt.ylim(0.4, 0.8) # Focus on top performance range

save_path = RESULTS_DIR / "champion_learning_curves.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Plot saved to: {save_path}")
print("="*60)
