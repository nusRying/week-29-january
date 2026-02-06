import pandas as pd
import numpy as np
import sys
import os
import csv
import time
import json
from pathlib import Path

# Paths
SCRIPT_PATH = Path(__file__).resolve()
STUDY_ROOT = SCRIPT_PATH.parent
DATA_DIR = STUDY_ROOT / "data"
RUNS_DIR = STUDY_ROOT / "runs"
LOGS_DIR = STUDY_ROOT / "logs"

# Ensure LCS library is accessible
PROJECT_ROOT = STUDY_ROOT.parent
LCS_PATH = PROJECT_ROOT / "Derived_Features_Champion" / "scikit-ExSTraCS-master"
if str(LCS_PATH) not in sys.path: sys.path.append(str(LCS_PATH))

from skExSTraCS.ExSTraCS import ExSTraCS

SEEDS = [42, 43, 44, 45, 46]
P_SPEC_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
DATASETS = ["ham", "isic"]

class CoveringStudyExSTraCS(ExSTraCS):
    """Extended ExSTraCS with covering event logging via callback hook."""
    
    def __init__(self, log_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_path = log_path
        self.covering_events = []
        
        # Set up CSV logger
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'num_specified', 'generality', 'fitness', 'accuracy', 'specified_features'])
        
        # Register covering callback
        self.covering_logger = self._log_covering_event
    
    def _log_covering_event(self, classifier, iteration):
        """Callback invoked during covering to log classifier details."""
        num_specified = len(classifier.specifiedAttList)
        d = self.env.formatData.numAttributes
        generality = 1.0 - (num_specified / d)
        
        # Log to CSV
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration,
                num_specified,
                generality,
                classifier.fitness,
                classifier.accuracy,
                ','.join(map(str, sorted(classifier.specifiedAttList)))
            ])

def run_part_a(dataset_name, p_spec, seed):
    tag = f"{dataset_name}_pspec{p_spec}_seed{seed}"
    print(f"\n[COVERING RUN] {tag} ...")
    
    log_file = LOGS_DIR / f"log_{tag}.csv"
    run_dir = RUNS_DIR / dataset_name / f"pspec_{p_spec}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    # Load data with train/val splits
    data_path = DATA_DIR / f"{dataset_name}_clean.csv"
    df = pd.read_csv(data_path)
    
    # Load pre-computed splits
    train_idx = np.load(DATA_DIR / f"{dataset_name}_train_indices.npy")
    val_idx = np.load(DATA_DIR / f"{dataset_name}_val_indices.npy")
    
    X_train = df.drop('label', axis=1).iloc[train_idx].values
    y_train = df['label'].iloc[train_idx].values
    X_val = df.drop('label', axis=1).iloc[val_idx].values
    y_val = df['label'].iloc[val_idx].values
    
    # Map p_spec to RSL for ExSTraCS implementation
    # Note: In ExSTraCS covering, a rule is initialized by randomly selecting 
    # a number of attributes to specify (between 1 and rule_specificity_limit).
    # To approximate p_spec (probability of specifying each attribute), 
    # we set RSL such that the expected number of specified attributes = p_spec * d
    # Since selection is uniform random from [1, RSL], mean = (RSL + 1) / 2
    # Therefore: (RSL + 1) / 2 = p_spec * d  =>  RSL = 2 * p_spec * d - 1
    # CRITICAL: RSL must not exceed d (can't specify more attributes than exist)
    d = X_train.shape[1]
    rsl = max(1, min(d, int(round(2 * p_spec * d - 1))))

    model = CoveringStudyExSTraCS(
        log_path = log_file,
        learning_iterations=10000, # Focused covering isolation
        N=3000,
        nu=10,
        rule_specificity_limit=rsl,
        random_state=seed,
        theta_GA=10**9, # GA OFF
        mu=0.0          # Mutation OFF
    )
    
    start = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - start
    
    # Validation accuracy
    from sklearn.metrics import balanced_accuracy_score
    y_pred = model.predict(X_val)
    val_ba = balanced_accuracy_score(y_val, y_pred)
    
    # Save report
    report = {
        'tag': tag,
        'dataset': dataset_name,
        'p_spec_target': p_spec,
        'rsl': rsl,
        'seed': seed,
        'duration': duration,
        'final_micro_pop': model.population.microPopSize,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'val_balanced_accuracy': val_ba
    }
    with open(run_dir / "report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"  Done in {duration:.1f}s. Val BA: {val_ba:.4f}")


if __name__ == "__main__":
    # Example loop for HAM
    for ds in DATASETS:
        for ps in P_SPEC_VALUES:
            for sd in SEEDS:
                run_part_a(ds, ps, sd)
