"""
Champion Pipeline Phase H: Model Training (ExSTraCS)
Trains the final ExSTraCS model using the selected Top 100 interaction features.
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

# 1. SETUP
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_FILE = SCRIPT_DIR / "binary_features_all.csv"
METADATA_FILE = SCRIPT_DIR / "feature_metadata.json"
MODELS_DIR = SCRIPT_DIR / "Models"

# Timestamp for unique filenames
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
MODEL_OUTPUT = MODELS_DIR / f"champion_model_{TIMESTAMP}.pkl"
REPORT_OUTPUT = MODELS_DIR / f"champion_report_{TIMESTAMP}.json"

# Reference to "Latest" for the master script/inference
LATEST_MODEL = SCRIPT_DIR / "batched_fe_model.pkl"
LATEST_REPORT = SCRIPT_DIR / "batched_fe_report.json"

# Add local ExSTraCS to path
EXSTRACS_LOCAL_PATH = SCRIPT_DIR / "scikit-ExSTraCS-master"
if EXSTRACS_LOCAL_PATH.exists():
    sys.path.append(str(EXSTRACS_LOCAL_PATH))
    from skExSTraCS.ExSTraCS import ExSTraCS
else:
    from skrebate import ExSTraCS

def run_training_pipeline():
    # 1. Load Metadata & Data
    with open(METADATA_FILE, 'r') as f:
        meta = json.load(f)
    df = pd.read_csv(DATA_FILE)
    
    # 2. Prepare Features (Phase B/C)
    base_cols = meta['base_features']
    X = df[base_cols].values
    y = df['label'].values
    X = np.nan_to_num(X)
    
    # 3. Split (Must match the selection split)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=meta['seed']
    )
    
    # 4. Phase D-G: Generate Polynomial Features & Select Top 100
    print("Generating interaction features...")
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    # Important: In this phase, we use the top_indices from our selection
    X_train_poly = poly.fit_transform(X_train_raw)
    X_train_champion = X_train_poly[:, meta['top_indices']]
    
    X_test_poly = poly.transform(X_test_raw)
    X_test_champion = X_test_poly[:, meta['top_indices']]
    
    # 5. Handle Class Imbalance
    print("Resampling training data for balance...")
    ros = RandomOverSampler(random_state=meta['seed'])
    X_train_res, y_train_res = ros.fit_resample(X_train_champion, y_train)
    
    # 6. Phase H: ExSTraCS Training
    params = {
        'learningIterations': 500000, 
        'N': 3000, 
        'nu': 10
    }
    print(f"Training ExSTraCS with interactions: {params}")
    
    model = ExSTraCS(
        learning_iterations=params['learningIterations'],
        N=params['N'],
        nu=params['nu'],
        random_state=meta['seed']
    )
    
    start = time.time()
    model.fit(X_train_res, y_train_res)
    duration = time.time() - start
    
    # 7. Evaluate
    y_pred = model.predict(X_test_champion)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    print(f"\nTraining Complete in {duration/60:.2f} minutes")
    print(f"Test Balanced Accuracy: {bal_acc:.4f}")
    
    # 8. Save Model & Stats (Archive with Timestamp)
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODEL_OUTPUT, 'wb') as f:
        pickle.dump(model, f)
    
    # Also update "Latest" reference in the main folder
    with open(LATEST_MODEL, 'wb') as f:
        pickle.dump(model, f)
        
    stats = {
        "name": "Derived_Features_Champion",
        "timestamp": TIMESTAMP,
        "params": params,
        "duration": duration,
        "metrics": {
            "balanced_accuracy": float(bal_acc),
            "sensitivity": float(tp / (tp + fn)),
            "specificity": float(tn / (tn + fp))
        }
    }
    # Save archive report
    with open(REPORT_OUTPUT, 'w') as f:
        json.dump(stats, f, indent=4)
        
    # Update latest report
    with open(LATEST_REPORT, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"Archive saved to: {MODEL_OUTPUT.relative_to(SCRIPT_DIR)}")
    print(f"Latest reference updated: {LATEST_MODEL.name}")

if __name__ == "__main__":
    run_training_pipeline()
