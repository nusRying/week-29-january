"""
Phase 1.1: Hyperparameter Grid Search for Champion Model
Uses insights from Covering Study (p_spec) and Mutation Study (mu) to optimize ExSTraCS.
"""
import sys
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, make_scorer
from itertools import product
import time

# Add ExSTraCS to path
CHAMPION_PATH = Path(__file__).parent.parent / "Derived_Features_Champion"
EXSTRAC_PATH = CHAMPION_PATH / "scikit-ExSTraCS-master"
sys.path.insert(0, str(EXSTRAC_PATH))

from skExSTraCS.ExSTraCS import ExSTraCS

# Paths
RESULTS_DIR = Path(__file__).parent / "phase1_results"
MODELS_DIR = Path(__file__).parent / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Load Top 100 champion features
print("Loading Top 100 champion feature dataset...")
champion_data_path = Path(__file__).parent / "champion_top100_features.csv"

if not champion_data_path.exists():
    print("ERROR: champion_top100_features.csv not found!")
    print("Please run generate_top100_dataset.py first.")
    sys.exit(1)

df_full = pd.read_csv(champion_data_path)

print(f"Dataset loaded: {len(df_full)} samples, {len(df_full.columns)-1} features")

# Prepare data
X_full = df_full.drop('label', axis=1).values
y_full = df_full['label'].values

# Split: 70% train/val, 30% final test (never touched during grid search)
from sklearn.model_selection import train_test_split
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_full, y_full, test_size=0.30, random_state=42, stratify=y_full
)

print(f"Train/Val: {len(X_trainval)} | Final Test: {len(X_test)}")

# ============================================================
# HYPERPARAMETER GRID (Informed by Covering/Mutation Studies)
# ============================================================

param_grid = {
    'N': [2000, 3000, 5000],                      # Population size
    'nu': [5, 10, 15],                            # Specificity threshold
    'mu': [0.01, 0.04, 0.08],                     # Mutation rate (from Mutation Study)
    'theta_GA': [15, 25, 50],                     # GA activation frequency
    'rule_specificity_limit': [50, 75, 100],      # Max features per rule
}

# Learning iterations (fixed for speed)
LEARNING_ITERATIONS = 100000  # Reduced from 500k for grid search speed

print("\n" + "="*60)
print("PHASE 1.1: HYPERPARAMETER GRID SEARCH")
print("="*60)
print(f"Grid size: {np.prod([len(v) for v in param_grid.values()])} configurations")
print(f"Estimated time: {np.prod([len(v) for v in param_grid.values()]) * 0.5:.1f} hours (0.5h per config)")
print("="*60 + "\n")

# Generate all parameter combinations
param_combinations = list(product(*param_grid.values()))
param_names = list(param_grid.keys())

results = []

for idx, params in enumerate(param_combinations, 1):
    config = dict(zip(param_names, params))
    
    print(f"\n[{idx}/{len(param_combinations)}] Testing configuration:")
    print(f"  N={config['N']}, nu={config['nu']}, mu={config['mu']}, "
          f"theta_GA={config['theta_GA']}, RSL={config['rule_specificity_limit']}")
    
    # 3-Fold Cross-Validation
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_trainval, y_trainval), 1):
        X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]
        
        # Train ExSTraCS
        model = ExSTraCS(
            learning_iterations=LEARNING_ITERATIONS,
            N=config['N'],
            nu=config['nu'],
            mu=config['mu'],
            theta_GA=config['theta_GA'],
            rule_specificity_limit=config['rule_specificity_limit'],
            random_state=42 + fold
        )
        
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Validate
        y_pred = model.predict(X_val)
        ba = balanced_accuracy_score(y_val, y_pred)
        cv_scores.append(ba)
        
        print(f"    Fold {fold}: BA = {ba:.4f} ({train_time:.1f}s)")
    
    # Record results
    mean_cv_ba = np.mean(cv_scores)
    std_cv_ba = np.std(cv_scores)
    
    results.append({
        **config,
        'mean_cv_ba': mean_cv_ba,
        'std_cv_ba': std_cv_ba,
        'cv_scores': cv_scores,
        'train_time_avg': train_time / 3
    })
    
    print(f"  → Mean CV BA: {mean_cv_ba:.4f} ± {std_cv_ba:.4f}")
    
    # Save intermediate results
    pd.DataFrame(results).to_csv(RESULTS_DIR / "hyperparameter_search_results.csv", index=False)

# ============================================================
# BEST CONFIGURATION SELECTION
# ============================================================

results_df = pd.DataFrame(results)
best_idx = results_df['mean_cv_ba'].idxmax()
best_config = results_df.iloc[best_idx].to_dict()

print("\n" + "="*60)
print("BEST CONFIGURATION FOUND:")
print("="*60)
for param in param_names:
    print(f"  {param}: {best_config[param]}")
print(f"\nMean CV BA: {best_config['mean_cv_ba']:.4f} ± {best_config['std_cv_ba']:.4f}")
print("="*60 + "\n")

# ============================================================
# TRAIN FINAL MODEL WITH BEST CONFIG
# ============================================================

print("Training final model on full train/val set with best configuration...")

final_model = ExSTraCS(
    learning_iterations=500000,  # Full iterations for final model
    N=int(best_config['N']),
    nu=int(best_config['nu']),
    mu=best_config['mu'],
    theta_GA=int(best_config['theta_GA']),
    rule_specificity_limit=int(best_config['rule_specificity_limit']),
    random_state=42
)

final_model.fit(X_trainval, y_trainval)

# Test on held-out test set
y_test_pred = final_model.predict(X_test)
test_ba = balanced_accuracy_score(y_test, y_test_pred)

print(f"\nFinal Test BA (ISIC2019 held-out): {test_ba:.4f}")

# Save best model
model_path = MODELS_DIR / "best_hyperparam_model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)
print(f"Saved best model to: {model_path}")

# Save best configuration
config_path = RESULTS_DIR / "best_configuration.json"
best_config_clean = {k: v for k, v in best_config.items() if k in param_names}
best_config_clean['final_test_ba'] = float(test_ba)
best_config_clean['mean_cv_ba'] = float(best_config['mean_cv_ba'])

with open(config_path, 'w') as f:
    json.dump(best_config_clean, f, indent=2)
print(f"Saved configuration to: {config_path}")

print("\n✅ Phase 1.1 Complete! Run phase1_threshold_optimization.py next.")
