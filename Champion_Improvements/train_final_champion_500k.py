"""
FINAL PhD CHAMPION MODEL TRAINING
Combines optimal parameters (nu=5, mu=0.08) with full 500k iterations.
Expected performance: 74-75% BA on both ISIC and HAM with strong generalization.
"""
import sys
import numpy as np
import pandas as pd
import pickle
import json
import time
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import PolynomialFeatures

# Add ExSTraCS to path
CHAMPION_PATH = Path("C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/Derived_Features_Champion")
IMPROV_PATH = Path("C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/Champion_Improvements")
sys.path.insert(0, str(CHAMPION_PATH / "scikit-ExSTraCS-master"))

from skExSTraCS.ExSTraCS import ExSTraCS

# Paths
RESULTS_DIR = IMPROV_PATH / "final_champion_results"
MODELS_DIR = IMPROV_PATH / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

OPTIMAL_THRESHOLD = 0.31  # From previous optimization
SEED = 42

print("="*70)
print("FINAL PhD CHAMPION MODEL TRAINING")
print("="*70)
print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("\nConfiguration:")
print("  - Iterations: 500,000 (Full Training)")
print("  - Population Size: 3,000")
print("  - nu (Generalization): 5 (Optimized)")
print("  - mu (Mutation): 0.08 (Optimized)")
print("  - RSL: 100")
print("  - Features: Top 100 Degree-2 Interactions")
print("="*70)

# 1. LOAD DATA
print("\n[1/5] Loading Top 100 Feature Dataset...")
df_full = pd.read_csv(IMPROV_PATH / "champion_top100_features.csv")
X = df_full.drop('label', axis=1).values
y = df_full['label'].values
print(f"  Dataset: {len(y)} samples × {X.shape[1]} features")

# Split: 70% Train, 30% Test (matching original baseline)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
print(f"  Train: {len(y_train)} samples | Test: {len(y_test)} samples")

# 2. TRAIN FINAL CHAMPION MODEL
print("\n[2/5] Training Final Champion Model (500k iterations)...")
print("  This will take approximately 5-6 hours. Progress updates every 100k iterations.")
print(f"  Training started at: {time.strftime('%H:%M:%S')}")

start_time = time.time()

model = ExSTraCS(
    learning_iterations=500000,
    N=3000,
    nu=5,           # Optimized from Fast study
    mu=0.08,        # Optimized from Fast study
    theta_GA=25,
    rule_specificity_limit=100,
    random_state=SEED
)

model.fit(X_train, y_train)

training_time = (time.time() - start_time) / 60
print(f"\n  ✅ Training Complete in {training_time:.2f} minutes ({training_time/60:.2f} hours)")

# Save model
model_path = MODELS_DIR / "final_champion_500k.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"  Model saved: {model_path.name}")

# 3. EVALUATE ON ISIC2019 (INTERNAL)
print("\n[3/5] Evaluating on ISIC2019 (Internal Test Set)...")

# Get predictions
y_train_proba = model.predict_proba(X_train)[:, 1]
y_test_proba = model.predict_proba(X_test)[:, 1]

# Apply optimal threshold
y_train_pred = (y_train_proba >= OPTIMAL_THRESHOLD).astype(int)
y_test_pred = (y_test_proba >= OPTIMAL_THRESHOLD).astype(int)

# Calculate metrics
train_ba = balanced_accuracy_score(y_train, y_train_pred)
test_ba = balanced_accuracy_score(y_test, y_test_pred)

tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
test_sens = tp / (tp + fn)
test_spec = tn / (tn + fp)

print(f"  Train BA: {train_ba:.4f}")
print(f"  Test BA:  {test_ba:.4f}")
print(f"  Test Sensitivity: {test_sens:.4f}")
print(f"  Test Specificity: {test_spec:.4f}")

# 4. EXTERNAL VALIDATION (HAM10000)
print("\n[4/5] Performing External Validation (HAM10000)...")

# Load HAM data
ham_raw = pd.read_csv(CHAMPION_PATH / "HAM10000_extracted_features.csv")
ham_meta = pd.read_csv(Path("C:/Users/umair/Videos/PhD/PhD Data/Week 15 January/Code V2/Dataset/clean/CleanData/HAM10000/HAM10000_metadata"))

# Merge and create labels
ham_merged = ham_raw.merge(ham_meta[['image_id', 'dx']], on='image_id', how='left')
MALIGNANT_CLASSES = ['mel', 'bcc', 'akiec']
ham_merged['label'] = ham_merged['dx'].apply(lambda x: 1 if x in MALIGNANT_CLASSES else 0)

# Extract features
meta_cols = ['image_id', 'label', 'dx', 'age', 'sex', 'localization']
feature_cols = [c for c in ham_merged.columns if c not in meta_cols]
X_ham_base = ham_merged[feature_cols].values
y_ham = ham_merged['label'].values

# Expand features (Degree 2)
print("  Expanding HAM features (Degree-2)...")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_ham_poly = poly.fit_transform(X_ham_base)

# Select Top 100
with open(CHAMPION_PATH / "feature_metadata.json") as f:
    meta = json.load(f)
    top_indices = meta['top_indices']

X_ham_top = X_ham_poly[:, top_indices]

# Predict
ham_proba = model.predict_proba(X_ham_top)[:, 1]
ham_pred = (ham_proba >= OPTIMAL_THRESHOLD).astype(int)

# Metrics
ham_ba = balanced_accuracy_score(y_ham, ham_pred)
tn_h, fp_h, fn_h, tp_h = confusion_matrix(y_ham, ham_pred).ravel()
ham_sens = tp_h / (tp_h + fn_h)
ham_spec = tn_h / (tn_h + fp_h)

print(f"  HAM10000 BA: {ham_ba:.4f}")
print(f"  HAM10000 Sensitivity: {ham_sens:.4f}")
print(f"  HAM10000 Specificity: {ham_spec:.4f}")

# 5. GENERATE FINAL REPORT
print("\n[5/5] Generating Final Report...")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'Final Champion (AUC = {roc_auc:.3f})', lw=2, color='#1f77b4')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Final PhD Champion Model - ROC Curve (ISIC2019)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig(RESULTS_DIR / "final_champion_roc.png", dpi=300, bbox_inches='tight')
print(f"  ROC curve saved: final_champion_roc.png")

# Comprehensive Report
final_report = {
    "model_name": "Final PhD Champion (500k)",
    "training_config": {
        "iterations": 500000,
        "N": 3000,
        "nu": 5,
        "mu": 0.08,
        "rsl": 100,
        "training_time_hours": round(training_time / 60, 2)
    },
    "threshold": OPTIMAL_THRESHOLD,
    "isic2019_internal": {
        "train_ba": float(train_ba),
        "test_ba": float(test_ba),
        "test_sensitivity": float(test_sens),
        "test_specificity": float(test_spec),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    },
    "ham10000_external": {
        "ba": float(ham_ba),
        "sensitivity": float(ham_sens),
        "specificity": float(ham_spec),
        "confusion_matrix": {"tn": int(tn_h), "fp": int(fp_h), "fn": int(fn_h), "tp": int(tp_h)}
    },
    "roc_auc": float(roc_auc)
}

report_path = RESULTS_DIR / "final_champion_report.json"
with open(report_path, 'w') as f:
    json.dump(final_report, f, indent=2)

print(f"  Final report saved: final_champion_report.json")

# Summary
print("\n" + "="*70)
print("🏆 TRAINING COMPLETE - FINAL CHAMPION MODEL")
print("="*70)
print(f"Completion Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total Training Duration: {training_time/60:.2f} hours")
print("\n📊 FINAL PERFORMANCE SUMMARY:")
print(f"  ISIC2019 Test BA:  {test_ba:.2%} (Sensitivity: {test_sens:.2%}, Specificity: {test_spec:.2%})")
print(f"  HAM10000 External: {ham_ba:.2%} (Sensitivity: {ham_sens:.2%}, Specificity: {ham_spec:.2%})")
print(f"\n  Model saved to: {model_path}")
print(f"  Full report: {report_path}")
print("="*70)
