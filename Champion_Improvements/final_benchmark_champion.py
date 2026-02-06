import sys
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import PolynomialFeatures

# Add ExSTraCS to path
CHAMPION_PATH = Path("C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/Derived_Features_Champion")
IMPROV_PATH = Path("C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/Champion_Improvements")
sys.path.insert(0, str(CHAMPION_PATH / "scikit-ExSTraCS-master"))

from skExSTraCS.ExSTraCS import ExSTraCS

# Constants
THRESHOLD = 0.31
SEED = 42

def calculate_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    ba = balanced_accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {"ba": float(ba), "sens": float(sens), "spec": float(spec)}

print("="*60)
print("INTEGRATED BENCHMARK: NEW PhD CHAMPION MODEL")
print("="*60)

# 1. LOAD MODEL
model_path = IMPROV_PATH / "models" / "best_hyperparam_model_fast.pkl"
print(f"\n[1/3] Loading Model: {model_path.name}")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# 2. INTERNAL VALIDATION (ISIC 2019)
print("\n[2/3] Performing Internal Validation (ISIC 2019)...")
df_isic = pd.read_csv(IMPROV_PATH / "champion_top100_features.csv")
X_full = df_isic.drop('label', axis=1).values
y_full = df_isic['label'].values

# Recreate training split
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_full, y_full, test_size=0.30, random_state=SEED, stratify=y_full
)

trainval_proba = model.predict_proba(X_trainval)[:, 1]
test_proba = model.predict_proba(X_test)[:, 1]

isic_train_stats = calculate_metrics(y_trainval, trainval_proba, THRESHOLD)
isic_test_stats = calculate_metrics(y_test, test_proba, THRESHOLD)

print(f"  Train/Val (70%) - BA: {isic_train_stats['ba']:.4f}, Sens: {isic_train_stats['sens']:.4f}, Spec: {isic_train_stats['spec']:.4f}")
print(f"  Test (30%)      - BA: {isic_test_stats['ba']:.4f}, Sens: {isic_test_stats['sens']:.4f}, Spec: {isic_test_stats['spec']:.4f}")

# 3. EXTERNAL VALIDATION (HAM10000)
print("\n[3/3] Performing External Validation (HAM10000)...")
ham_raw = pd.read_csv(CHAMPION_PATH / "HAM10000_extracted_features.csv")
ham_meta = pd.read_csv(Path("C:/Users/umair/Videos/PhD/PhD Data/Week 15 January/Code V2/Dataset/clean/CleanData/HAM10000/HAM10000_metadata"))

# Merge to get dx labels
ham_merged = ham_raw.merge(ham_meta[['image_id', 'dx']], on='image_id', how='left')

# Binary mapping (from test_generalization.py)
MALIGNANT_CLASSES = ['mel', 'bcc', 'akiec']
ham_merged['label'] = ham_merged['dx'].apply(lambda x: 1 if x in MALIGNANT_CLASSES else 0)

# Identify features and label
meta_cols = ['image_id', 'label', 'dx', 'age', 'sex', 'localization']
feature_cols = [c for c in ham_merged.columns if c not in meta_cols]
X_ham_base = ham_merged[feature_cols].values
y_ham = ham_merged['label'].values

# Expand features (Degree 2)
print("  Expanding HAM features (Degree-2)...")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_ham_poly = poly.fit_transform(X_ham_base)

# Select Top 100 features
with open(CHAMPION_PATH / "feature_metadata.json") as f:
    meta = json.load(f)
    top_indices = meta['top_indices']

X_ham_top = X_ham_poly[:, top_indices]

# Predict
ham_proba = model.predict_proba(X_ham_top)[:, 1]
ham_stats = calculate_metrics(y_ham, ham_proba, THRESHOLD)

print(f"  HAM10000 (Full) - BA: {ham_stats['ba']:.4f}, Sens: {ham_stats['sens']:.4f}, Spec: {ham_stats['spec']:.4f}")

# SAVE FINAL REPORT
final_results = {
    "model": "PhD_Champion_Fast_1.1",
    "threshold": THRESHOLD,
    "internal_isic": {
        "train_val": isic_train_stats,
        "test": isic_test_stats
    },
    "external_ham10000": ham_stats
}

output_path = IMPROV_PATH / "final_results" / "full_benchmark_stats.json"
with open(output_path, 'w') as f:
    json.dump(final_results, f, indent=2)

print(f"\n✅ Benchmark Complete! Results saved to: {output_path}")
print("="*60)
