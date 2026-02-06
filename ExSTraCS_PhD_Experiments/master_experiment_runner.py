import pandas as pd
import numpy as np
import sys
import os
import csv
import pickle
import time
import json
from pathlib import Path

# Paths
PROJECT_ROOT = Path("c:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code")
LCS_PATH = PROJECT_ROOT / "Derived_Features_Champion" / "scikit-ExSTraCS-master"
if str(LCS_PATH) not in sys.path: sys.path.append(str(LCS_PATH))

from skExSTraCS.ExSTraCS import ExSTraCS

# Global Experiment Settings
STUDY_ROOT = PROJECT_ROOT / "ExSTraCS_PhD_Experiments"
DATA_DIR = STUDY_ROOT / "data"
RUNS_DIR = STUDY_ROOT / "runs"
SNAPSHOTS_DIR = STUDY_ROOT / "snapshots"

CHECKPOINTS = [0, 10000, 50000, 100000, 200000, 300000, 400000, 500000]
SEEDS = [42, 43, 44, 45, 46]

class MasterMonitoredExSTraCS(ExSTraCS):
    """
    Unified Monitoring for ExSTraCS Experiments:
    1. CSV Snapshots at checkpoints
    2. Covering Log (every new rule created by covering)
    3. Initialization Audit (fitness, accuracy, etc.)
    """
    def __init__(self, run_tag, snapshot_dir, covering_log_path, **kwargs):
        super().__init__(**kwargs)
        self.run_tag = run_tag
        self.snapshot_dir = snapshot_dir
        self.covering_log_path = covering_log_path
        
        # Initialize covering log header
        with open(self.covering_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'id', 'action', 'num_specified', 'generality', 'fitness', 'accuracy', 'experience', 'specified_features'])

    def runIteration(self, state_phenotype):
        # Snapshoting at fixed checkpoints
        if self.iterationCount in CHECKPOINTS:
            self.export_snapshot(self.iterationCount)
        
        super().runIteration(state_phenotype)
        
        # End of training snapshot
        if self.iterationCount == self.learning_iterations - 1:
            self.export_snapshot(self.learning_iterations)

    def addClassifierToPopulation(self, model, cl, covering):
        """
        Captured by override to log covering events and initialization states.
        """
        if covering:
            num_feats = self.env.formatData.numAttributes
            gen = (num_feats - len(cl.specifiedAttList)) / num_feats
            feat_str = "|".join(map(str, sorted(cl.specifiedAttList)))
            rule_id = hash(tuple(sorted(cl.specifiedAttList)) + (cl.phenotype,))
            
            with open(self.covering_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.iterationCount, rule_id, cl.phenotype, 
                    len(cl.specifiedAttList), gen, cl.fitness, 
                    cl.accuracy, cl.matchCount, feat_str
                ])
        
        super().addClassifierToPopulation(model, cl, covering)

    def export_snapshot(self, it):
        """Exports the entire population to a CSV."""
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        filename = self.snapshot_dir / f"iter_{it:06d}.csv"
        num_features = self.env.formatData.numAttributes
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['rule_id', 'action', 'specified_features', 'mask', 'num_specified', 'generality', 'fitness', 'accuracy', 'numerosity', 'experience']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.header_written = False
            
            pop = self.population.popSet
            if not pop:
                writer.writeheader()
                return

            writer.writeheader()
            for cl in pop:
                mask_bits = ['0'] * num_features
                for idx in cl.specifiedAttList:
                    if idx < num_features: mask_bits[idx] = '1'
                mask_str = "".join(mask_bits)
                rule_id = hash(tuple(sorted(cl.specifiedAttList)) + (cl.phenotype,))
                
                writer.writerow({
                    'rule_id': rule_id,
                    'action': cl.phenotype,
                    'specified_features': "|".join(map(str, sorted(cl.specifiedAttList))),
                    'mask': mask_str,
                    'num_specified': len(cl.specifiedAttList),
                    'generality': (num_features - len(cl.specifiedAttList)) / num_features,
                    'fitness': cl.fitness,
                    'accuracy': cl.accuracy,
                    'numerosity': cl.numerosity,
                    'experience': cl.matchCount
                })

def run_experiment(part_name, condition_name, dataset_name, params, seed):
    """Orchestrates a single run configuration."""
    tag = f"{part_name}_{condition_name}_{dataset_name}_seed_{seed}"
    print(f"\n[RUN] {tag} ...")
    
    # Setup paths
    run_root = RUNS_DIR / part_name / condition_name / dataset_name / f"seed_{seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    
    snap_dir = SNAPSHOTS_DIR / part_name / condition_name / dataset_name / f"seed_{seed}"
    snap_dir.mkdir(parents=True, exist_ok=True)
    
    covering_log = run_root / "covering_log.csv"
    
    # Load standardized data
    data_file = DATA_DIR / f"{dataset_name.lower()}_clean.csv"
    df = pd.read_csv(data_file)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    # Fixed Params (from Plan)
    # N=3000, iterations=500000, nu=10
    # selection=tournament, random_state=seed
    
    model = MasterMonitoredExSTraCS(
        run_tag=tag,
        snapshot_dir=snap_dir,
        covering_log_path=covering_log,
        learning_iterations=500000,
        N=3000,
        nu=10,
        random_state=seed,
        mu=params.get('mu', 0.04),
        chi=params.get('chi', 0.8),
        # rule_specificity_limit behaves as the 1/p_spec if we consider it as max features to specify
        # For ExSTraCS, it is the actual integer limit. 
        # If p_spec = 0.1, and d=226, RSL should be ~22? 
        # Actually ExSTraCS initializeByCovering uses: random.randint(1, model.rule_specificity_limit)
        # So mean specification is RSL/2. 
        # To get mean spec = p_spec * d, we need RSL = 2 * p_spec * d.
        rule_specificity_limit=params.get('rsl', 15), 
        theta_GA=params.get('theta_ga', 25)
    )
    
    start = time.time()
    model.fit(X, y)
    duration = time.time() - start
    
    # Save results summary
    results = {
        'tag': tag,
        'duration': duration,
        'final_accuracy': model.get_final_training_accuracy(),
        'params': params
    }
    with open(run_root / "results.json", 'w') as f:
        json.dump(results, f)
        
    print(f"  Done in {duration:.1f}s. Acc: {results['final_accuracy']:.4f}")

if __name__ == "__main__":
    # This script is designed to be called with specific loops or arguments.
    # To avoid accidentally running 230 jobs in one thread, 
    # I'll expose the functions so I can batch them.
    pass
