"""
FINAL VALIDATION: INTEGRATED CHAMPION MODEL
Integrates all successful improvement phases into a single pipeline:
1. Features: Top 120 Degree-3 Interactions (Phase 1.3)
2. Data: SMOTE + Hard Negative Mining + Noise Augmentation (Phase 3.2)
3. Model: Multi-Configuration Ensemble (Phase 2)
4. Threshold: Youden's J Optimization (Phase 1.2)

Expected Performance: >79% Balanced Accuracy
"""
import sys
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE
import time

# Add ExSTraCS to path
CHAMPION_PATH = Path(__file__).parent.parent / "Derived_Features_Champion"
EXSTRAC_PATH = CHAMPION_PATH / "scikit-ExSTraCS-master"
sys.path.insert(0, str(EXSTRAC_PATH))

from skExSTraCS.ExSTraCS import ExSTraCS

# Paths
RESULTS_DIR = Path(__file__).parent / "final_results"
MODELS_DIR = Path(__file__).parent / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

print("="*60)
print("FINAL VALIDATION: THE ULTIMATE CHAMPION MODEL")
print("="*60)

# ============================================================
# 1. LOAD ENHANCED FEATURES (Phase 1.3)
# ============================================================
degree3_enhanced = Path(__file__).parent / "phase1_results" / "degree3_enhanced_features.csv"

if not degree3_enhanced.exists():
    print("ERROR: degree3_enhanced_features.csv not found!")
    print("Please run phase1_degree3_features.py first.")
    sys.exit(1)

print("\n[1/5] Loading Top 120 Degree-3 Features...")
df_full = pd.read_csv(degree3_enhanced)
X = df_full.drop('label', axis=1).values
y = df_full['label'].values
feature_names = df_full.drop('label', axis=1).columns.tolist()

print(f"  Dataset: {len(y)} samples × {X.shape[1]} features")

# Split: 60% Train, 20% Val, 20% Test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train_raw, X_val, y_train_raw, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

# ============================================================
# 2. APPLY DATA IMPROVEMENTS (Phase 3.2)
# ============================================================
print("\n[2/5] Applying Data Improvements (SMOTE + Mining + Augmentation)...")

# A. SMOTE Balancing
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train_raw, y_train_raw)
print(f"  SMOTE: {len(y_train_raw)} -> {len(y_train_smote)} samples")

# B. Hard Negative Mining (Simulated by 2x weighting known hard patterns)
# Train quick probe model to find hard cases
probe = ExSTraCS(learning_iterations=10000, N=500, nu=10, random_state=42)
probe.fit(X_train_smote, y_train_smote)
y_probe_pred = probe.predict(X_train_smote)
hard_mask = y_probe_pred != y_train_smote
X_hard = X_train_smote[hard_mask]
y_hard = y_train_smote[hard_mask]
print(f"  Hard Mining: Found {len(X_hard)} hard cases to reinject")

# C. Noise Augmentation on Malignant Class
mal_indices = np.where(y_train_smote == 1)[0]
num_aug = int(len(mal_indices) * 0.3)
X_aug = X_train_smote[mal_indices[:num_aug]] + np.random.normal(0, 0.05, X_train_smote[mal_indices[:num_aug]].shape)
y_aug = np.ones(num_aug)
print(f"  Augmentation: Added {num_aug} noisy malignant samples")

# Combine all
X_train = np.vstack([X_train_smote, X_hard, X_aug])
y_train = np.hstack([y_train_smote, y_hard, y_aug])

print(f"  Final Training Set: {len(y_train)} samples")

# ============================================================
# 3. TRAIN MULTI-CONFIG ENSEMBLE (Phase 2)
# ============================================================
print("\n[3/5] Training Ensemble (5 Models in Parallel)...")
print("Maximize CPU usage activated: Training all 5 models simultaneously 🚀")

from joblib import Parallel, delayed

# ============================================================
# 3. TRAIN MULTI-CONFIG ENSEMBLE (Phase 2 - ADJUSTED TO LEGACY SPECS)
# ============================================================
# Legacy Model Analysis (batched_fe_model.pkl) showed 72.75% BA with:
# N=3000, nu=10, iterations=500k, RSL=15 (Strict Generality!)
print("\n[3/5] Training Ensemble (Legacy-Derived Configs)...")
print("Maximize CPU usage activated: Training 5 models (500k iters, RSL=15) 🚀")

from joblib import Parallel, delayed

# Ensemble: All based on the successful Legacy Config, varying Mutation/Seed
ensemble_configs = [
    {'name': 'Legacy_Base',  'p_spec': 0.15, 'mu': 0.04, 'weight': 0.30}, # The Champion (RSL~15-18)
    {'name': 'Legacy_LowMut','p_spec': 0.15, 'mu': 0.01, 'weight': 0.20}, # Stable
    {'name': 'Legacy_HighMut','p_spec': 0.15, 'mu': 0.08, 'weight': 0.20}, # Explorer
    {'name': 'Legacy_Seed2', 'p_spec': 0.15, 'mu': 0.04, 'weight': 0.15}, # Diversity 1 (Seed variation handled by Parallel randomness)
    {'name': 'Legacy_Seed3', 'p_spec': 0.15, 'mu': 0.04, 'weight': 0.15}  # Diversity 2
]

iterations = 500000 # Increased from 200k back to 500k (Legacy standard)

def train_single_model(config, X_train, y_train, X_val, y_val, seed_offset):
    print(f"  Starting {config['name']} (p={config['p_spec']}, mu={config['mu']})...")
    
    # Strict RSL from Legacy Model (RSL=15 for 100 features -> ~15-18 for 120 features)
    # 0.15 * 120 = 18. This forces generalization.
    rsl = 18 
    
    model = ExSTraCS(
        learning_iterations=iterations,
        N=3000,
        nu=10,
        mu=config['mu'],
        theta_GA=25,
        rule_specificity_limit=rsl,
        random_state=42 + seed_offset # Vary seed for diversity
    )
    
    model.fit(X_train, y_train)
    
    # Quick Val Check
    val_acc = balanced_accuracy_score(y_val, model.predict(X_val))
    print(f"  ✅ {config['name']} Finished -> Val BA: {val_acc:.4f}")
    
    return {'model': model, 'config': config}

# Train in parallel using all available cores
ensemble_models = Parallel(n_jobs=-1, verbose=10)(
    delayed(train_single_model)(config, X_train, y_train, X_val, y_val, idx) 
    for idx, config in enumerate(ensemble_configs)
)

# ============================================================
# 4. OPTIMIZE THRESHOLD (Phase 1.2)
# ============================================================
print("\n[4/5] Optimizing Ensemble Threshold...")

def get_ensemble_proba(models, X):
    total_proba = np.zeros(len(X))
    for item in models:
        proba = item['model'].predict_proba(X)[:, 1]
        total_proba += proba * item['config']['weight']
    return total_proba

y_val_proba = get_ensemble_proba(ensemble_models, X_val)

# Find Best Threshold via Youden's J
fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

print(f"  Optimal Threshold: {best_threshold:.4f}")

# ============================================================
# 5. FINAL EVALUATION
# ============================================================
print("\n[5/5] Final Test Set Evaluation...")

y_test_proba = get_ensemble_proba(ensemble_models, X_test)
y_test_pred = (y_test_proba >= best_threshold).astype(int)

final_ba = balanced_accuracy_score(y_test, y_test_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print("\n" + "="*60)
print(f"🏆 FINAL RESULT: {final_ba:.4f} Balanced Accuracy")
print("="*60)
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Improvement over Baseline (72.75%): +{(final_ba - 0.7275)*100:.2f}%")

# Save detailed report
report = {
    'final_ba': float(final_ba),
    'sensitivity': float(sensitivity),
    'specificity': float(specificity),
    'optimal_threshold': float(best_threshold),
    'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
}

with open(RESULTS_DIR / "final_report.json", 'w') as f:
    json.dump(report, f, indent=2)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'Ensemble AUC = {auc(fpr, tpr):.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Final Champion Ensemble ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig(RESULTS_DIR / "final_roc.png")

print(f"\nFinal report saved to {RESULTS_DIR}")
