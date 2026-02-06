import pandas as pd
import numpy as np
import os
import sys
import json
import time
import shutil
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

# -----------------------------------------------------------------------------
# 1. SETUP & PATHS
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
RUNS_DIR = SCRIPT_DIR / "runs"
PLOTS_DIR = SCRIPT_DIR / "plots"

# Source Paths (Champion Model)
CHAMPION_DIR = SCRIPT_DIR.parent / "Derived_Features_Champion"
SOURCE_DATA = CHAMPION_DIR / "binary_features_all.csv"
SOURCE_METADATA = CHAMPION_DIR / "feature_metadata.json"
HAM_DATA = CHAMPION_DIR / "HAM10000_extracted_features.csv" # For external validation features
# Note: "binary_features_all.csv" already contains the ISIC dataset labels. 
# We need to ensure we can replicate the exact feature extraction for HAM if needed, 
# or use a pre-existing HAM feature file if compatible.
# Checking the Champion folder, it has "HAM10000_extracted_features.csv". 
# However, the "binary_features_all.csv" is the ISIC training set. 
# We need to trust the Champion protocol: Base features -> Poly -> Selection -> Scaling.

# Add ExSTraCS to path
EXSTRACS_PATH = CHAMPION_DIR / "scikit-ExSTraCS-master"
if str(EXSTRACS_PATH) not in sys.path:
    sys.path.append(str(EXSTRACS_PATH))
from skExSTraCS.ExSTraCS import ExSTraCS

# -----------------------------------------------------------------------------
# 2. EXPERIMENT PARAMETERS
# -----------------------------------------------------------------------------
MULTIPLIERS = [0.12, 0.25, 0.5, 1, 2, 3, 4]
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
BASE_NUM_IMAGES = 24464
ITERATIONS = 500000
NU = 10

# -----------------------------------------------------------------------------
# 3. HELPER: METRICS CALCULATOR
# -----------------------------------------------------------------------------
def calculate_metrics(y_true, y_pred, rule_count=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Formulas as requested
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    bal_acc = (rec + (tn / (tn + fp))) / 2 # (TPR + TNR) / 2
    
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "balanced_accuracy": bal_acc,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
    }
    
    if rule_count is not None and bal_acc > 0:
        metrics["rule_efficiency"] = rule_count / bal_acc
        metrics["rules_active"] = rule_count
        
    return metrics

# -----------------------------------------------------------------------------
# 4. CORE ENGINE using "Protocol Parity"
# -----------------------------------------------------------------------------
def run_experiment(multiplier, seed):
    tag = f"N{multiplier}_seed{seed}"
    run_output = RUNS_DIR / f"report_{tag}.json"
    
    if run_output.exists():
        print(f"[{tag}] Skipping (Already Exists)")
        return
    
    print(f"[{tag}] Starting...")
    start_time = time.time()
    
    # A. LOAD DATA
    # We load afresh to ensure no leakage between parallel threads (though joblib handles this)
    df = pd.read_csv(DATA_DIR / "binary_features_all.csv")
    with open(DATA_DIR / "feature_metadata.json", 'r') as f:
        meta = json.load(f)
        
    base_cols = meta['base_features']
    top_indices = meta['top_indices']
    
    X = df[base_cols].values
    y = df['label'].values
    X = np.nan_to_num(X) # 2. Imputation
    
    # B. STRATIFIED SPLIT (80/20)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed # 3. Seeded Split
    )
    
# -----------------------------------------------------------------------------
# 4. CORE ENGINE using "Protocol Parity" (Memory Optimized)
# -----------------------------------------------------------------------------
def run_experiment(multiplier, seed):
    tag = f"N{multiplier}_seed{seed}"
    run_output = RUNS_DIR / f"report_{tag}.json"
    
    if run_output.exists():
        return # Skip silently for batching resume capability
    
    try:
        start_time = time.time()
        
        # A. LOAD DATA
        # We load afresh to ensure no leakage between parallel threads
        df = pd.read_csv(DATA_DIR / "binary_features_all.csv")
        with open(DATA_DIR / "feature_metadata.json", 'r') as f:
            meta = json.load(f)
            
        base_cols = meta['base_features']
        top_feature_names = meta['top_feature_names'] # USE DIRECT NAMES TO SAVE MEMORY
        
        # Helper to construct features on the fly without massive Poly matrix
        # This replaces the 4GB+ allocation with a ~15MB allocation
        selected_features = []
        for feat_name in top_feature_names:
            parts = feat_name.split(' ')
            if len(parts) == 1:
                col = df[parts[0]].values
            elif len(parts) == 2:
                col = df[parts[0]].values * df[parts[1]].values
            else:
                raise ValueError(f"Unexpected feature name format: {feat_name}")
            selected_features.append(col)
        
        X_selected = np.column_stack(selected_features)
        y = df['label'].values
        X_selected = np.nan_to_num(X_selected)
        
        # B. STRATIFIED SPLIT (80/20)
        # Note: We split the ALREADY SELECTED features. 
        # This is mathematically identical to Splitting Base -> Poly -> Select
        # provided the row order is preserved (it is).
        X_train_sel, X_test_sel, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, stratify=y, random_state=seed
        )
        
        # E. NORMALIZATION (Train-Fit, Transform All)
        scaler = MinMaxScaler(feature_range=(0,1))
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)
        
        # F. CLASS BALANCING (Train Only)
        ros = RandomOverSampler(random_state=seed)
        X_train_res, y_train_res = ros.fit_resample(X_train_scaled, y_train)
        
        # G. MODEL INIT & TRAIN
        N = int(multiplier * BASE_NUM_IMAGES)
        
        model = ExSTraCS(
            learning_iterations=ITERATIONS,
            N=N,
            nu=NU,
            random_state=seed
        )
        
        final_train_start = time.time()
        model.fit(X_train_res, y_train_res) # Fit on balanced data
        train_duration = time.time() - final_train_start
        
        # H. EVALUATION
        # 1. Test Set (Natural Distribution)
        y_pred_test = model.predict(X_test_scaled)
        test_metrics = calculate_metrics(y_test, y_pred_test, model.population.microPopSize)
        
        # 2. Train Set (Natural Distribution - for "Did it learn?" check)
        y_pred_train = model.predict(X_train_scaled)
        train_metrics = calculate_metrics(y_train, y_pred_train, model.population.microPopSize)
        
        duration = time.time() - start_time
        
        # SAVE REPORT
        report = {
            "multiplier": multiplier,
            "N": N,
            "seed": seed,
            "duration_total": duration,
            "duration_training": train_duration,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "final_rule_count": model.population.microPopSize
        }
        
        with open(run_output, 'w') as f:
            json.dump(report, f, indent=4)
            
        print(f"[{tag}] Done. Test BA: {test_metrics['balanced_accuracy']:.4f}")
    
    except Exception as e:
        print(f"[{tag}] FAILED: {e}")
        raise e


# -----------------------------------------------------------------------------
# 5. ORCHESTRATOR
# -----------------------------------------------------------------------------
def main():
    # Setup Data Standalone
    DATA_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    
    if not (DATA_DIR / "binary_features_all.csv").exists():
        print("Copying dataset to standalone folder...")
        shutil.copy(SOURCE_DATA, DATA_DIR / "binary_features_all.csv")
    
    if not (DATA_DIR / "feature_metadata.json").exists():
        print("Copying metadata to standalone folder...")
        shutil.copy(SOURCE_METADATA, DATA_DIR / "feature_metadata.json")

    print(f"--- Population Multiplier Study (Batched) ---")
    
    # Generate job list
    jobs = []
    for m in MULTIPLIERS:
        for s in SEEDS:
            jobs.append((m, s))
            
    # BATCHED EXECUTION
    # We process in batches of 10 to ensure stability and regular reporting
    BATCH_SIZE = 10
    total_batches = (len(jobs) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"Total Runs: {len(jobs)} | Batches: {total_batches} | Batch Size: {BATCH_SIZE}")
    
    for i in range(total_batches):
        batch = jobs[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        print(f"\n--- Starting Batch {i+1}/{total_batches} ({len(batch)} jobs) ---")
        
        # Execute batch
        # n_jobs=5 is a safe balance between speed and CPU load
        Parallel(n_jobs=5)(delayed(run_experiment)(m, s) for m, s in batch)
        
    print("\nALL EXPERIMENTS COMPLETE.")

if __name__ == "__main__":
    main()
