import pandas as pd
import numpy as np
import sys
import os
import csv
import time
import json
import pickle
from pathlib import Path

# Paths
SCRIPT_PATH = Path(__file__).resolve()
STUDY_ROOT = SCRIPT_PATH.parent
DATA_DIR = STUDY_ROOT / "data"
RUNS_DIR = STUDY_ROOT / "runs"
SNAPSHOTS_DIR = STUDY_ROOT / "snapshots"

PROJECT_ROOT = STUDY_ROOT.parent
LCS_PATH = PROJECT_ROOT / "Derived_Features_Champion" / "scikit-ExSTraCS-master"
if str(LCS_PATH) not in sys.path: sys.path.append(str(LCS_PATH))

from skExSTraCS.ExSTraCS import ExSTraCS
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed

SEEDS = [42, 43, 44, 45, 46]
CHECKPOINTS = [0, 10000, 50000, 100000, 200000, 300000, 400000, 500000]

class MutationStudyExSTraCS(ExSTraCS):
    def __init__(self, snapshot_dir, **kwargs):
        super().__init__(**kwargs)
        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def runIteration(self, state_phenotype):
        if self.iterationCount in CHECKPOINTS:
            self.export_snapshot(self.iterationCount)
        super().runIteration(state_phenotype)
        if self.iterationCount == self.learning_iterations - 1:
            self.export_snapshot(self.learning_iterations)

    def export_snapshot(self, it):
        filename = self.snapshot_dir / f"iter_{it:06d}.csv"
        num_features = self.env.formatData.numAttributes
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['rule_id', 'action', 'specified_features', 'mask', 'num_specified', 'generality', 'fitness', 'accuracy', 'numerosity', 'experience']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            pop = self.population.popSet
            if not pop: return

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

def run_mutation_run(part_name, condition, dataset, seed, mu, theta_ga=25, pspec=0.5):
    tag = f"{dataset}_{condition}_seed{seed}"
    print(f"\n[{part_name}] {tag} (mu={mu}, ga={theta_ga}, pspec={pspec}) ...")
    
    run_dir = RUNS_DIR / part_name / condition / dataset / f"seed_{seed}"
    snap_dir = SNAPSHOTS_DIR / part_name / condition / dataset / f"seed_{seed}"
    
    if (run_dir / "results.json").exists():
        print(f"  [SKIP] {tag} already completed.")
        return

    run_dir.mkdir(parents=True, exist_ok=True)

    data_path = DATA_DIR / f"{dataset}_clean.csv"
    df = pd.read_csv(data_path)
    X = df.drop('label', axis=1).values.astype(float)
    y = df['label'].values.astype(int)
    
    # NORMALIZATION (Protocol: Always Normalize)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    d = X.shape[1]
    rsl = min(int(round(2 * pspec * d)), d) # Cap at num_features to prevent error

    model = MutationStudyExSTraCS(
        snapshot_dir=snap_dir,
        learning_iterations=500000,
        N=3000,
        nu=10,
        mu=mu,
        theta_GA=theta_ga,
        rule_specificity_limit=rsl,
        random_state=seed
    )
    
    start = time.time()
    model.fit(X, y)
    duration = time.time() - start
    
    with open(run_dir / "results.json", 'w') as f:
        json.dump({'tag': tag, 'duration': duration, 'final_acc': model.get_final_training_accuracy()}, f)
    
    print(f"  Done in {duration:.1f}s.")

if __name__ == "__main__":
    # Experiment conditions as defined in PhD Task List
    DATASETS = ["ham", "isic"]
    
    tasks = []
    
    # PART B.1: Core Factorial (High/Low/None Mutation)
    for ds in DATASETS:
        for seed in SEEDS:
            tasks.append(("Part_B1", "HighMut", ds, seed, 0.08))
        for seed in SEEDS:
            tasks.append(("Part_B1", "LowMut", ds, seed, 0.01))
        for seed in SEEDS:
            tasks.append(("Part_B1", "NoMut", ds, seed, 0.0))

    # PART B.2: Dose-Response Sweep
    MU_SWEEP = [0.0, 0.01, 0.04, 0.08, 0.12]
    DR_SEEDS = [42, 43, 44]
    for ds in DATASETS:
        for mu in MU_SWEEP:
            for seed in DR_SEEDS:
                tasks.append(("Part_B2", f"mu_{mu}", ds, seed, mu))

    print("\n" + "="*80)
    print("STARTING PARALLEL MUTATION STUDY (BATCHED)")
    print(f"Total tasks: {len(tasks)}")
    print("="*80)

    # BATCHED EXECUTION
    # Process in batches of 10 to ensure stability and regular reporting
    BATCH_SIZE = 10
    total_batches = (len(tasks) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"Batches: {total_batches} | Batch Size: {BATCH_SIZE}")
    
    for i in range(total_batches):
        batch = tasks[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        print(f"\n--- Starting Batch {i+1}/{total_batches} ({len(batch)} jobs) ---")
        
        # Execute batch
        # n_jobs=5 is a safe balance
        Parallel(n_jobs=5)(delayed(run_mutation_run)(part, cond, ds, sd, mu) for part, cond, ds, sd, mu in batch)

    print("\n--- ALL MUTATION RUNS COMPLETE ---")
