"""
Baseline Comparisons for Covering Study
Compares ExSTraCS p_spec variants against simple baseline strategies.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

# LCS Path
LCS_PATH = Path("c:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/scikit-ExSTraCS-master")
sys.path.append(str(LCS_PATH))
from skExSTraCS import ExSTraCS

SCRIPT_PATH = Path(__file__).resolve()
STUDY_ROOT = SCRIPT_PATH.parent
DATA_DIR = STUDY_ROOT / "data"
RESULTS_DIR = STUDY_ROOT / "results"

np.random.seed(42)

def run_baseline_experiment(strategy, dataset_name):
    """Run single baseline experiment."""
    data_file = DATA_DIR / f"{dataset_name}_clean.csv"
    df = pd.read_csv(data_file)
    
    train_idx, val_idx = train_test_split(
        range(len(df)), test_size=0.2, random_state=42, 
        stratify=df['label']
    )
    
    X_train = df.drop('label', axis=1).iloc[train_idx].values
    y_train = df['label'].iloc[train_idx].values
    X_val = df.drop('label', axis=1).iloc[val_idx].values
    y_val = df['label'].iloc[val_idx].values
    
    d = X_train.shape[1]
    
    # Configure based on strategy
    if strategy == 'random':
        # Default: p_spec ~= 0.5 (RSL = d//2)
        rsl = d // 2
    elif strategy == 'greedy':
        # Maximally specific
        rsl = d  # All features
    elif strategy == 'minimal':
        # Minimally specific
        rsl = 10  # Very few features
    
    model = ExSTraCS(
        learning_iterations=10000,
        rule_specificity_limit=rsl,
        do_GA_subsumption=False,
        do_correct_set_subsumption=False,
        chi=0.0,  # No GA
        mu=0.0    # No mutation
    )
    
    # Train
    print(f"    Training with RSL={rsl}...", end='', flush=True)
    model.fit(X_train, y_train)
    
    # Validate
    y_pred = model.predict(X_val)
    val_ba = balanced_accuracy_score(y_val, y_pred)
    
    return {
        'strategy': strategy,
        'dataset': dataset_name,
        'val_ba': val_ba,
        'rsl': rsl
    }

def run_all_baselines():
    """Run baseline comparisons."""
    results = []
    
    for dataset in ['ham', 'isic']:
        print(f"\nDataset: {dataset.upper()}")
        for strategy in ['minimal', 'random', 'greedy']:
            print(f"  [{strategy:8s}]", end=' ')
            try:
                result = run_baseline_experiment(strategy, dataset)
                results.append(result)
                print(f" Val BA = {result['val_ba']:.4f}")
            except Exception as e:
                print(f" FAILED: {e}")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'baseline_comparisons.csv', index=False)
    
    print("\n" + "="*60)
    print("BASELINE COMPARISON RESULTS")
    print("="*60)
    print(df.to_string(index=False))
    print(f"\nSaved to: results/baseline_comparisons.csv")
    
    return df

if __name__ == "__main__":
    print("=== Baseline Comparison Experiments ===\n")
    print("Strategies:")
    print("  - Minimal: RSL=10 (very general, few features)")
    print("  - Random: RSL=d/2 (moderate, ~113 features)")
    print("  - Greedy: RSL=d (maximally specific, all 226 features)")
    print("\nGA and Mutation disabled to isolate covering effects.")
    
    run_all_baselines()
    print("\n=== Done ===")
