"""
EXACT MUTATION STUDY REPLICATION
Replicates the exact configuration that achieved:
- HAM: 86.06% BA (mu=0.0, p_spec=0.7)
- ISIC: 84.18% BA (mu=0.0, p_spec=0.7)

Uses:
- EXACT train/val splits from mutation study
- mu = 0.0 (NO mutation)
- Single model (not ensemble)
- Full dataset training
"""
import sys
import numpy as np
import pandas as pd
import pickle
import json
import time
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

# Add ExSTraCS to path
CHAMPION_PATH = Path("C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/Derived_Features_Champion")
IMPROV_PATH = Path("C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/Champion_Improvements")
MUTATION_DIR = Path("C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/GA_Mutation_Study")
sys.path.insert(0, str(CHAMPION_PATH / "scikit-ExSTraCS-master"))

from skExSTraCS.ExSTraCS import ExSTraCS

# Paths
MUTATION_DATA_DIR = MUTATION_DIR / "data"
RESULTS_DIR = IMPROV_PATH / "mutation_replication_results"
MODELS_DIR = IMPROV_PATH / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

SEED = 42

print("="*80)
print("EXACT MUTATION STUDY REPLICATION")
print("="*80)
print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("\n🎯 GOAL: Replicate mutation study's 86% HAM / 84% ISIC results")
print("\nConfiguration (EXACT mutation study baseline):")
print("  ✓ mu = 0.0        (NO mutation - baseline condition)")
print("  ✓ nu = 10         (Standard)")
print("  ✓ N = 3000        (Standard)")
print("  ✓ Iterations = 500k")
print("  ✓ p_spec = 0.7    (via RSL calculation)")
print("  ✓ EXACT train/val splits from mutation study")
print("="*80)

# Results storage
results = {}

# ==================== HAM REPLICATION ====================
print("\n" + "="*80)
print("PART 1: HAM10000 REPLICATION (Target: 86.06%)")
print("="*80)

print("\n[1/4] Loading HAM data with EXACT mutation study splits...")
ham_df = pd.read_csv(MUTATION_DATA_DIR / "ham_clean.csv")
X_ham_full = ham_df.drop('label', axis=1).values
y_ham_full = ham_df['label'].values

# Load EXACT indices from mutation study
ham_train_idx = np.load(MUTATION_DATA_DIR / "ham_train_indices.npy")
ham_val_idx = np.load(MUTATION_DATA_DIR / "ham_val_indices.npy")

X_ham_train = X_ham_full[ham_train_idx]
y_ham_train = y_ham_full[ham_train_idx]
X_ham_val = X_ham_full[ham_val_idx]
y_ham_val = y_ham_full[ham_val_idx]

print(f"  HAM Full: {len(y_ham_full)} samples × {X_ham_full.shape[1]} features")
print(f"  HAM Train: {len(y_ham_train)} (from mutation study indices)")
print(f"  HAM Val: {len(y_ham_val)} (from mutation study indices)")

print("\n[2/4] Training HAM model (mutation study baseline config)...")
d_ham = X_ham_full.shape[1]
rsl_ham = min(int(round(2 * 0.7 * d_ham)), d_ham)  # p_spec=0.7 via RSL

print(f"  Calculated RSL: {rsl_ham} (for p_spec≈0.7, d={d_ham})")
print(f"  Training started at: {time.strftime('%H:%M:%S')}")
print("  Estimated time: ~30-45 minutes")

start_time = time.time()

ham_model = ExSTraCS(
    learning_iterations=500000,
    N=3000,
    nu=10,
    mu=0.0,  # NO MUTATION - baseline condition
    theta_GA=25,
    rule_specificity_limit=rsl_ham,
    random_state=SEED
)

ham_model.fit(X_ham_train, y_ham_train)

ham_train_time = (time.time() - start_time) / 60
print(f"\n  ✅ HAM Training Complete in {ham_train_time:.2f} minutes")

print("\n[3/4] Evaluating HAM model...")
y_ham_train_pred = ham_model.predict(X_ham_train)
y_ham_val_pred = ham_model.predict(X_ham_val)

ham_train_ba = balanced_accuracy_score(y_ham_train, y_ham_train_pred)
ham_val_ba = balanced_accuracy_score(y_ham_val, y_ham_val_pred)

tn, fp, fn, tp = confusion_matrix(y_ham_val, y_ham_val_pred).ravel()
ham_sens = tp / (tp + fn)
ham_spec = tn / (tn + fp)

print(f"  HAM Train BA: {ham_train_ba:.4f}")
print(f"  HAM Val BA:   {ham_val_ba:.4f}")
print(f"  HAM Val Sensitivity: {ham_sens:.4f}")
print(f"  HAM Val Specificity: {ham_spec:.4f}")

print("\n[4/4] Comparison to Mutation Study...")
mutation_ham = 0.8606
diff_ham = ham_val_ba - mutation_ham
print(f"  Mutation Study HAM: {mutation_ham:.2%}")
print(f"  This Replication:   {ham_val_ba:.2%}")
print(f"  Difference:         {diff_ham*100:+.2f}%")

if abs(diff_ham) < 0.02:
    print("  ✅ SUCCESSFULLY REPLICATED (within ±2%)")
else:
    print(f"  ⚠️  REPLICATION MISMATCH ({abs(diff_ham)*100:.1f}% difference)")

results['ham'] = {
    'train_ba': float(ham_train_ba),
    'val_ba': float(ham_val_ba),
    'sensitivity': float(ham_sens),
    'specificity': float(ham_spec),
    'mutation_study_ba': mutation_ham,
    'difference': float(diff_ham),
    'replicated': abs(diff_ham) < 0.02,
    'training_time_minutes': ham_train_time
}

# Save HAM model
ham_model_path = MODELS_DIR / "ham_mutation_replication.pkl"
with open(ham_model_path, 'wb') as f:
    pickle.dump(ham_model, f)

# ==================== ISIC REPLICATION ====================
print("\n" + "="*80)
print("PART 2: ISIC2019 REPLICATION (Target: 84.18%)")
print("="*80)

print("\n[1/4] Loading ISIC data with EXACT mutation study splits...")
isic_df = pd.read_csv(MUTATION_DATA_DIR / "isic_clean.csv")
X_isic_full = isic_df.drop('label', axis=1).values
y_isic_full = isic_df['label'].values

# Load EXACT indices from mutation study
isic_train_idx = np.load(MUTATION_DATA_DIR / "isic_train_indices.npy")
isic_val_idx = np.load(MUTATION_DATA_DIR / "isic_val_indices.npy")

X_isic_train = X_isic_full[isic_train_idx]
y_isic_train = y_isic_full[isic_train_idx]
X_isic_val = X_isic_full[isic_val_idx]
y_isic_val = y_isic_full[isic_val_idx]

print(f"  ISIC Full: {len(y_isic_full)} samples × {X_isic_full.shape[1]} features")
print(f"  ISIC Train: {len(y_isic_train)} (from mutation study indices)")
print(f"  ISIC Val: {len(y_isic_val)} (from mutation study indices)")

print("\n[2/4] Training ISIC model (mutation study baseline config)...")
d_isic = X_isic_full.shape[1]
rsl_isic = min(int(round(2 * 0.7 * d_isic)), d_isic)  # p_spec=0.7 via RSL

print(f"  Calculated RSL: {rsl_isic} (for p_spec≈0.7, d={d_isic})")
print(f"  Training started at: {time.strftime('%H:%M:%S')}")
print("  Estimated time: ~45-60 minutes (ISIC is larger)")

start_time = time.time()

isic_model = ExSTraCS(
    learning_iterations=500000,
    N=3000,
    nu=10,
    mu=0.0,  # NO MUTATION - baseline condition
    theta_GA=25,
    rule_specificity_limit=rsl_isic,
    random_state=SEED
)

isic_model.fit(X_isic_train, y_isic_train)

isic_train_time = (time.time() - start_time) / 60
print(f"\n  ✅ ISIC Training Complete in {isic_train_time:.2f} minutes")

print("\n[3/4] Evaluating ISIC model...")
y_isic_train_pred = isic_model.predict(X_isic_train)
y_isic_val_pred = isic_model.predict(X_isic_val)

isic_train_ba = balanced_accuracy_score(y_isic_train, y_isic_train_pred)
isic_val_ba = balanced_accuracy_score(y_isic_val, y_isic_val_pred)

tn, fp, fn, tp = confusion_matrix(y_isic_val, y_isic_val_pred).ravel()
isic_sens = tp / (tp + fn)
isic_spec = tn / (tn + fp)

print(f"  ISIC Train BA: {isic_train_ba:.4f}")
print(f"  ISIC Val BA:   {isic_val_ba:.4f}")
print(f"  ISIC Val Sensitivity: {isic_sens:.4f}")
print(f"  ISIC Val Specificity: {isic_spec:.4f}")

print("\n[4/4] Comparison to Mutation Study...")
mutation_isic = 0.8418
diff_isic = isic_val_ba - mutation_isic
print(f"  Mutation Study ISIC: {mutation_isic:.2%}")
print(f"  This Replication:    {isic_val_ba:.2%}")
print(f"  Difference:          {diff_isic*100:+.2f}%")

if abs(diff_isic) < 0.02:
    print("  ✅ SUCCESSFULLY REPLICATED (within ±2%)")
else:
    print(f"  ⚠️  REPLICATION MISMATCH ({abs(diff_isic)*100:.1f}% difference)")

results['isic'] = {
    'train_ba': float(isic_train_ba),
    'val_ba': float(isic_val_ba),
    'sensitivity': float(isic_sens),
    'specificity': float(isic_spec),
    'mutation_study_ba': mutation_isic,
    'difference': float(diff_isic),
    'replicated': abs(diff_isic) < 0.02,
    'training_time_minutes': isic_train_time
}

# Save ISIC model
isic_model_path = MODELS_DIR / "isic_mutation_replication.pkl"
with open(isic_model_path, 'wb') as f:
    pickle.dump(isic_model, f)

# ==================== FINAL REPORT ====================
print("\n" + "="*80)
print("🏆 MUTATION STUDY REPLICATION COMPLETE")
print("="*80)
print(f"Completion Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total Time: HAM {ham_train_time:.1f}min + ISIC {isic_train_time:.1f}min = {ham_train_time + isic_train_time:.1f}min")

print("\n📊 REPLICATION SUMMARY:")
print(f"  HAM10000:")
print(f"    Mutation Study: 86.06%")
print(f"    Replication:    {ham_val_ba:.2%} ({diff_ham*100:+.2f}%)")
print(f"    Status:         {'✅ REPLICATED' if results['ham']['replicated'] else '❌ FAILED'}")

print(f"\n  ISIC2019:")
print(f"    Mutation Study: 84.18%")
print(f"    Replication:    {isic_val_ba:.2%} ({diff_isic*100:+.2f}%)")
print(f"    Status:         {'✅ REPLICATED' if results['isic']['replicated'] else '❌ FAILED'}")

# Overall verdict
both_replicated = results['ham']['replicated'] and results['isic']['replicated']
print("\n" + "="*80)
if both_replicated:
    print("✅ ✅ ✅ FULL REPLICATION SUCCESSFUL ✅ ✅ ✅")
    print("All mutation study results verified!")
else:
    print("⚠️  PARTIAL OR FAILED REPLICATION")
    if not results['ham']['replicated']:
        print(f"  - HAM failed to replicate ({abs(diff_ham)*100:.1f}% off)")
    if not results['isic']['replicated']:
        print(f"  - ISIC failed to replicate ({abs(diff_isic)*100:.1f}% off)")
print("="*80)

# Save comprehensive report
final_report = {
    'replication_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'configuration': {
        'mu': 0.0,
        'nu': 10,
        'N': 3000,
        'iterations': 500000,
        'p_spec_target': 0.7,
        'rsl_ham': rsl_ham,
        'rsl_isic': rsl_isic,
        'used_exact_splits': True
    },
    'results': results,
    'overall_success': both_replicated
}

report_path = RESULTS_DIR / "replication_report.json"
with open(report_path, 'w') as f:
    json.dump(final_report, f, indent=2)

print(f"\n  Models saved:")
print(f"    HAM:  {ham_model_path}")
print(f"    ISIC: {isic_model_path}")
print(f"  Report: {report_path}")
print("="*80)
