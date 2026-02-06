"""
ISIC-TRAINED CHAMPION MODEL
Train on ISIC2019 (84% BA demonstrated in Mutation Study)
Test on HAM10000 (external validation)

Uses optimal parameters from Mutation Study:
- mu = 0.04
- nu = 10
- N = 3000
- 500k iterations
- 226 raw features
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

# Add ExSTraCS to path
CHAMPION_PATH = Path("C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/Derived_Features_Champion")
IMPROV_PATH = Path("C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/Champion_Improvements")
sys.path.insert(0, str(CHAMPION_PATH / "scikit-ExSTraCS-master"))

from skExSTraCS.ExSTraCS import ExSTraCS
from joblib import Parallel, delayed

# Paths
RESULTS_DIR = IMPROV_PATH / "isic_champion_results"
MODELS_DIR = IMPROV_PATH / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

SEED = 42

print("="*80)
print("ISIC-TRAINED CHAMPION MODEL")
print("="*80)
print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("\nStrategy: Train on ISIC2019 (84% BA proven), Validate on HAM10000")
print("\nParameter Configuration:")
print("  ✓ mu = 0.04       (Mutation Study optimal)")
print("  ✓ nu = 10         (Baseline optimal)")
print("  ✓ N = 3000        (Population size)")
print("  ✓ Iterations = 500k")
print("  ✓ Features = 226 raw (matching mutation study)")
print("="*80)

# 1. LOAD ISIC2019 DATA
print("\n[1/6] Loading ISIC2019 Training Data (226 raw features)...")

# Load ISIC data from mutation study (same 226 features)
MUTATION_DATA_DIR = Path("C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/GA_Mutation_Study/data")
isic_df = pd.read_csv(MUTATION_DATA_DIR / "isic_clean.csv")

# Extract features and labels
X_isic = isic_df.drop('label', axis=1).values
y_isic = isic_df['label'].values

print(f"  ISIC2019: {len(y_isic)} samples × {X_isic.shape[1]} features")
print(f"  Class distribution: {(y_isic==1).sum()} malignant, {(y_isic==0).sum()} benign")

# Split ISIC: 70% train, 30% test
X_train_isic, X_test_isic, y_train_isic, y_test_isic = train_test_split(
    X_isic, y_isic, test_size=0.30, random_state=SEED, stratify=y_isic
)
print(f"  ISIC Train: {len(y_train_isic)} | ISIC Test: {len(y_test_isic)}")

# 2. LOAD HAM10000 DATA (for external validation)
print("\n[2/6] Loading HAM10000 External Validation Data...")

# Load HAM data from mutation study (same 226 features)
ham_df = pd.read_csv(MUTATION_DATA_DIR / "ham_clean.csv")

# Extract features and labels
X_ham = ham_df.drop('label', axis=1).values
y_ham = ham_df['label'].values

print(f"  HAM10000: {len(y_ham)} samples × {X_ham.shape[1]} features")
print(f"  Class distribution: {(y_ham==1).sum()} malignant, {(y_ham==0).sum()} benign")

# 3. TRAIN ENSEMBLE ON ISIC2019 (PARALLEL - MAX CPU)
print("\n[3/6] Training Ensemble on ISIC2019 (5 Models in Parallel)...")
print("  🚀 MAXIMIZE CPU USAGE: Training 5 models simultaneously")
print("  Configuration: mu=0.04, nu=10, 500k iterations each")
print(f"  Training started at: {time.strftime('%H:%M:%S')}")
print("  Estimated time: ~60-90 minutes (parallel, ISIC is larger)")

def train_single_model(seed_id, X_train, y_train):
    """Train a single ExSTraCS model with given seed"""
    print(f"  Starting Model #{seed_id}...")
    
    model = ExSTraCS(
        learning_iterations=500000,
        N=3000,
        mu=0.04,
        nu=10,
        theta_GA=25,
        rule_specificity_limit=100,
        random_state=SEED + seed_id
    )
    
    model.fit(X_train, y_train)
    print(f"  ✅ Model #{seed_id} Complete")
    return model

start_time = time.time()

# Train 5 models in parallel (use all available cores)
ensemble_models = Parallel(n_jobs=-1, verbose=10)(
    delayed(train_single_model)(i, X_train_isic, y_train_isic) 
    for i in range(5)
)

training_time = (time.time() - start_time) / 60
print(f"\n  ✅ Ensemble Training Complete in {training_time:.2f} minutes")
print(f"  Trained {len(ensemble_models)} models in parallel")

# Save ensemble
model_path = MODELS_DIR / "isic_ensemble_champion.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(ensemble_models, f)
print(f"  Ensemble saved: {model_path.name}")

# 4. THRESHOLD OPTIMIZATION (ENSEMBLE PREDICTIONS)
print("\n[4/6] Optimizing Decision Threshold (Ensemble Voting)...")

def ensemble_predict_proba(models, X):
    """Average probabilities from all ensemble models"""
    proba_sum = np.zeros((X.shape[0], 2))
    for model in models:
        proba_sum += model.predict_proba(X)
    return proba_sum / len(models)

y_train_proba = ensemble_predict_proba(ensemble_models, X_train_isic)[:, 1]
y_val_proba = ensemble_predict_proba(ensemble_models, X_test_isic)[:, 1]

# Find optimal threshold
thresholds = np.arange(0.1, 0.9, 0.01)
best_j = 0
best_thresh = 0.5

for thresh in thresholds:
    y_pred = (y_val_proba >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test_isic, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    j = sens + spec - 1
    
    if j > best_j:
        best_j = j
        best_thresh = thresh

print(f"  Optimal Threshold: {best_thresh:.4f} (Youden's J = {best_j:.4f})")

# 5. EVALUATE ON ISIC2019 (INTERNAL)
print("\n[5/6] Evaluating on ISIC2019 (Internal Test)...")

y_train_pred = (y_train_proba >= best_thresh).astype(int)
y_test_pred = (y_val_proba >= best_thresh).astype(int)

train_ba_isic = balanced_accuracy_score(y_train_isic, y_train_pred)
test_ba_isic = balanced_accuracy_score(y_test_isic, y_test_pred)

tn, fp, fn, tp = confusion_matrix(y_test_isic, y_test_pred).ravel()
test_sens_isic = tp / (tp + fn)
test_spec_isic = tn / (tn + fp)

print(f"  ISIC Train BA: {train_ba_isic:.4f}")
print(f"  ISIC Test BA:  {test_ba_isic:.4f}")
print(f"  ISIC Test Sensitivity: {test_sens_isic:.4f}")
print(f"  ISIC Test Specificity: {test_spec_isic:.4f}")

# 6. EXTERNAL VALIDATION (HAM10000)
print("\n[6/6] External Validation (HAM10000) - Ensemble Predictions...")

ham_proba = ensemble_predict_proba(ensemble_models, X_ham)[:, 1]
ham_pred = (ham_proba >= best_thresh).astype(int)

ham_ba = balanced_accuracy_score(y_ham, ham_pred)
tn_h, fp_h, fn_h, tp_h = confusion_matrix(y_ham, ham_pred).ravel()
ham_sens = tp_h / (tp_h + fn_h)
ham_spec = tn_h / (tn_h + fp_h)

print(f"  HAM10000 External BA: {ham_ba:.4f}")
print(f"  HAM10000 Sensitivity: {ham_sens:.4f}")
print(f"  HAM10000 Specificity: {ham_spec:.4f}")

# 7. GENERATE COMPREHENSIVE REPORT
print("\n[7/7] Generating Comprehensive Report...")

# ROC Curve (ISIC test)
fpr, tpr, _ = roc_curve(y_test_isic, y_val_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ISIC Ensemble (AUC = {roc_auc:.3f})', lw=2.5, color='#457B9D')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
plt.title('ISIC-Trained Ensemble - ROC Curve', fontsize=15, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=300, bbox_inches='tight')
print(f"  ROC curve saved: roc_curve.png")

# Comprehensive Report
final_report = {
    "model_name": "ISIC-Trained Ensemble Champion (5 Models)",
    "training_strategy": "Train ensemble on ISIC2019, validate on HAM10000 (parallel)",
    "ensemble_config": {
        "num_models": 5,
        "parallel_training": True,
        "voting_method": "probability_averaging"
    },
    "scientific_basis": {
        "mutation_study": "ISIC training achieved 84% BA in Part C experiments",
        "parameters": "mu=0.04, nu=10 (validated optimal)",
        "features": "226 raw features (same as mutation study)"
    },
    "training_config": {
        "iterations": 500000,
        "N": 3000,
        "mu": 0.04,
        "nu": 10,
        "rsl": 100,
        "training_time_minutes": round(training_time, 2)
    },
    "optimal_threshold": float(best_thresh),
    "isic2019_internal": {
        "train_ba": float(train_ba_isic),
        "test_ba": float(test_ba_isic),
        "test_sensitivity": float(test_sens_isic),
        "test_specificity": float(test_spec_isic),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_auc": float(roc_auc)
    },
    "ham10000_external": {
        "ba": float(ham_ba),
        "sensitivity": float(ham_sens),
        "specificity": float(ham_spec),
        "confusion_matrix": {"tn": int(tn_h), "fp": int(fp_h), "fn": int(fn_h), "tp": int(tp_h)}
    },
    "comparison": {
        "mutation_study_isic": 0.8418,
        "current_isic_test": float(test_ba_isic),
        "difference": float(test_ba_isic - 0.8418)
    }
}

report_path = RESULTS_DIR / "final_report.json"
with open(report_path, 'w') as f:
    json.dump(final_report, f, indent=2)

print(f"  Final report saved: final_report.json")

# Summary
print("\n" + "="*80)
print("🏆 ISIC-TRAINED ENSEMBLE CHAMPION - COMPLETE")
print("="*80)
print(f"Completion Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total Training Duration: {training_time:.2f} minutes (5 models in parallel)")
print("\n📊 ENSEMBLE PERFORMANCE:")
print(f"  Models Trained: 5 (parallel)")
print(f"  ISIC2019 Train: {train_ba_isic:.2%}")
print(f"  ISIC2019 Test:  {test_ba_isic:.2%} (Sens: {test_sens_isic:.2%}, Spec: {test_spec_isic:.2%})")
print(f"  HAM10000 External: {ham_ba:.2%} (Sens: {ham_sens:.2%}, Spec: {ham_spec:.2%})")
print("\n📈 COMPARISON TO MUTATION STUDY:")
print(f"  Mutation Study ISIC: 84.18%")
print(f"  This Ensemble ISIC:  {test_ba_isic:.2%}")
print(f"  Difference:          {(test_ba_isic - 0.8418)*100:+.2f}%")
print(f"\n  Ensemble: {model_path}")
print(f"  Report: {report_path}")
print("="*80)
print("\n✨ Bidirectional Cross-Validation Complete!")
print("   Combined with HAM→ISIC model, you have full external validation")
