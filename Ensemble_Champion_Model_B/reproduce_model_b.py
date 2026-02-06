"""
REPRODUCTION SCRIPT: ENSEMBLE CHAMPION (MODEL B)
Dir: Ensemble_Champion_Model_B

This script is 100% self-contained. It uses:
- Local data (isic_clean.csv, ham_clean.csv)
- Local LCS library (scikit-ExSTraCS-master)
- Model B hyperparameters (mu=0.04, nu=10, 500k iterations, 5-model ensemble)
"""
import sys
import numpy as np
import pandas as pd
import pickle
import json
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

# Setup local paths
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scikit-ExSTraCS-master"))

from skExSTraCS.ExSTraCS import ExSTraCS
from joblib import Parallel, delayed

# Config
SEED = 42
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

print("="*80)
print("REPRODUCING ENSEMBLE CHAMPION (MODEL B)")
print("="*80)

# 1. LOAD DATA
print("\n[1/4] Loading local data files...")
isic_df = pd.read_csv(ROOT / "isic_clean.csv")
ham_df = pd.read_csv(ROOT / "ham_clean.csv")

X_isic = isic_df.drop('label', axis=1).values
y_isic = isic_df['label'].values
X_ham = ham_df.drop('label', axis=1).values
y_ham = ham_df['label'].values

# Split ISIC 70/30 (Matches Model B setup)
X_train, X_test, y_train, y_test = train_test_split(
    X_isic, y_isic, test_size=0.30, random_state=SEED, stratify=y_isic
)

print(f"  ISIC Train: {len(y_train)} | ISIC Test: {len(y_test)}")
print(f"  HAM External: {len(y_ham)}")

# 2. DEFINE TRAINING
def train_member(i):
    print(f"  Starting Ensemble Member #{i}...")
    model = ExSTraCS(
        learning_iterations=500000,
        N=3000,
        mu=0.04,
        nu=10,
        theta_GA=25,
        rule_specificity_limit=100,
        random_state=SEED + i
    )
    model.fit(X_train, y_train)
    return model

# 3. OPTION TO RE-TRAIN OR LOAD
MODEL_FILE = MODELS_DIR / "isic_ensemble_champion.pkl"
if MODEL_FILE.exists():
    print(f"\n[2/4] Found existing model: {MODEL_FILE.name}")
    print("      Loading model for validation...")
    with open(MODEL_FILE, 'rb') as f:
        ensemble = pickle.load(f)
else:
    print("\n[2/4] Model not found. Starting training (5 models)...")
    start = time.time()
    ensemble = Parallel(n_jobs=-1)(delayed(train_member)(i) for i in range(5))
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(ensemble, f)
    print(f"  ✅ Training complete in {(time.time()-start)/60:.1f} min")

# 4. PREDICT & VALIDATE
def ensemble_predict_proba(models, X):
    proba_sum = np.zeros((X.shape[0], 2))
    for m in models:
        proba_sum += m.predict_proba(X)
    return proba_sum / len(models)

print("\n[3/4] Running Validation...")
# Use the threshold optimized for Model B (0.13)
THRESH = 0.13

# ISIC Test
isic_proba = ensemble_predict_proba(ensemble, X_test)[:, 1]
isic_pred = (isic_proba >= THRESH).astype(int)
isic_ba = balanced_accuracy_score(y_test, isic_pred)

# HAM External
ham_proba = ensemble_predict_proba(ensemble, X_ham)[:, 1]
ham_pred = (ham_proba >= THRESH).astype(int)
ham_ba = balanced_accuracy_score(y_ham, ham_pred)

print(f"  ISIC Test BA: {isic_ba:.2%}")
print(f"  HAM External BA: {ham_ba:.2%}")

# 5. SUMMARY
print("\n" + "="*80)
print("REPRODUCTION COMPLETE")
print(f"Target Results (Model B): ISIC 74.00%, HAM 71.80%")
print(f"Current Results:        ISIC {isic_ba:.2%}, HAM {ham_ba:.2%}")
print("="*80)
