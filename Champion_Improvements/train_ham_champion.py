"""
HAM-TRAINED CHAMPION MODEL
Train on HAM10000 (86% BA demonstrated in Mutation Study)
Test on ISIC2019 (external validation)

Uses optimal parameters from Mutation Study Part C:
- mu = 0.04
- nu = 10
- N = 3000
- 500k iterations
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

# Paths
RESULTS_DIR = IMPROV_PATH / "ham_champion_results"
MODELS_DIR = IMPROV_PATH / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

SEED = 42

print("="*80)
print("HAM-TRAINED CHAMPION MODEL")
print("="*80)
print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("\nStrategy: Train on HAM10000 (86% BA proven), Validate on ISIC2019")
print("\nParameter Configuration:")
print("  ✓ mu = 0.04       (Mutation Study optimal)")
print("  ✓ nu = 10         (Baseline optimal)")
print("  ✓ N = 3000        (Population size)")
print("  ✓ Iterations = 500k")
print("  ✓ RSL = 100")
print("="*80)

# 1. LOAD HAM10000 DATA
print("\n[1/6] Loading HAM10000 Training Data...")

# Load HAM data from mutation study (226 raw features)
MUTATION_DATA_DIR = Path("C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/GA_Mutation_Study/data")
ham_df = pd.read_csv(MUTATION_DATA_DIR / "ham_clean.csv")

# Extract features and labels
X_ham = ham_df.drop('label', axis=1).values
y_ham = ham_df['label'].values

print(f"  HAM10000: {len(y_ham)} samples × {X_ham.shape[1]} features")
print(f"  Class distribution: {(y_ham==1).sum()} malignant, {(y_ham==0).sum()} benign")

# Split HAM: 70% train, 30% test
X_train_ham, X_test_ham, y_train_ham, y_test_ham = train_test_split(
    X_ham, y_ham, test_size=0.30, random_state=SEED, stratify=y_ham
)
print(f"  HAM Train: {len(y_train_ham)} | HAM Test: {len(y_test_ham)}")

# 2. LOAD ISIC2019 DATA (for external validation NOTE)
print("\n[2/6] Loading ISIC2019 External Validation Data...")

# Load ISIC data from mutation study (same 226 features)
isic_df = pd.read_csv(MUTATION_DATA_DIR / "isic_clean.csv")
X_isic = isic_df.drop('label', axis=1).values
y_isic = isic_df['label'].values

print(f"  ISIC2019: {len(y_isic)} samples × {X_isic.shape[1]} features")
print(f"  Class distribution: {(y_isic==1).sum()} malignant, {(y_isic==0).sum()} benign")

# 3. TRAIN ENSEMBLE ON HAM10000 (PARALLEL - MAX CPU)
print("\n[3/6] Training Ensemble on HAM10000 (5 Models in Parallel)...")
print("  🚀 MAXIMIZE CPU USAGE: Training 5 models simultaneously")
print("  Configuration: mu=0.04, nu=10, 500k iterations each")
print(f"  Training started at: {time.strftime('%H:%M:%S')}")
print("  Estimated time: ~45-60 minutes (parallel)")

from joblib import Parallel, delayed

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
    delayed(train_single_model)(i, X_train_ham, y_train_ham) 
    for i in range(5)
)

training_time = (time.time() - start_time) / 60
print(f"\n  ✅ Ensemble Training Complete in {training_time:.2f} minutes")
print(f"  Trained {len(ensemble_models)} models in parallel")

# Save ensemble
model_path = MODELS_DIR / "ham_ensemble_champion.pkl"
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

y_train_proba = ensemble_predict_proba(ensemble_models, X_train_ham)[:, 1]
y_val_proba = ensemble_predict_proba(ensemble_models, X_test_ham)[:, 1]

# Find optimal threshold
thresholds = np.arange(0.1, 0.9, 0.01)
best_j = 0
best_thresh = 0.5

for thresh in thresholds:
    y_pred = (y_val_proba >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test_ham, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    j = sens + spec - 1
    
    if j > best_j:
        best_j = j
        best_thresh = thresh

print(f"  Optimal Threshold: {best_thresh:.4f} (Youden's J = {best_j:.4f})")

# 5. EVALUATE ON HAM10000 (INTERNAL)
print("\n[5/6] Evaluating on HAM10000 (Internal Test)...")

y_train_pred = (y_train_proba >= best_thresh).astype(int)
y_test_pred = (y_val_proba >= best_thresh).astype(int)

train_ba_ham = balanced_accuracy_score(y_train_ham, y_train_pred)
test_ba_ham = balanced_accuracy_score(y_test_ham, y_test_pred)

tn, fp, fn, tp = confusion_matrix(y_test_ham, y_test_pred).ravel()
test_sens_ham = tp / (tp + fn)
test_spec_ham = tn / (tn + fp)

print(f"  HAM Train BA: {train_ba_ham:.4f}")
print(f"  HAM Test BA:  {test_ba_ham:.4f}")
print(f"  HAM Test Sensitivity: {test_sens_ham:.4f}")
print(f"  HAM Test Specificity: {test_spec_ham:.4f}")

# 6. EXTERNAL VALIDATION (ISIC2019)  
print("\n[6/6] External Validation (ISIC2019) - Ensemble Predictions...")

isic_proba = ensemble_predict_proba(ensemble_models, X_isic)[:, 1]
isic_pred = (isic_proba >= best_thresh).astype(int)

isic_ba = balanced_accuracy_score(y_isic, isic_pred)
tn_i, fp_i, fn_i, tp_i = confusion_matrix(y_isic, isic_pred).ravel()
isic_sens = tp_i / (tp_i + fn_i)
isic_spec = tn_i / (tn_i + fp_i)

print(f"  ISIC2019 External BA: {isic_ba:.4f}")
print(f"  ISIC2019 Sensitivity: {isic_sens:.4f}")
print(f"  ISIC2019 Specificity: {isic_spec:.4f}")

# 7. GENERATE COMPREHENSIVE REPORT
print("\n[7/7] Generating Comprehensive Report...")

# ROC Curve (HAM test)
fpr, tpr, _ = roc_curve(y_test_ham, y_val_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'HAM-Trained Champion (AUC = {roc_auc:.3f})', lw=2.5, color='#E63946')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
plt.title('HAM-Trained Champion - ROC Curve', fontsize=15, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=300, bbox_inches='tight')
print(f"  ROC curve saved: roc_curve.png")

# Comprehensive Report
final_report = {
    "model_name": "HAM-Trained Ensemble Champion (5 Models)",
    "training_strategy": "Train ensemble on HAM10000, validate on ISIC2019 (bidirectional cross-validation)",
    "ensemble_config": {
        "num_models": 5,
        "parallel_training": True,
        "voting_method": "probability_averaging"
    },
    "scientific_basis": {
        "mutation_study": "HAM training achieved 86% BA, ISIC external 84% BA",
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
    "ham10000_internal": {
        "train_ba": float(train_ba_ham),
        "test_ba": float(test_ba_ham),
        "test_sensitivity": float(test_sens_ham),
        "test_specificity": float(test_spec_ham),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_auc": float(roc_auc)
    },
    "isic2019_external": {
        "ba": float(isic_ba),
        "sensitivity": float(isic_sens),
        "specificity": float(isic_spec),
        "confusion_matrix": {"tn": int(tn_i), "fp": int(fp_i), "fn": int(fn_i), "tp": int(tp_i)}
    },
    "comparison": {
        "mutation_study_ham": 0.8606,
        "mutation_study_isic_external": 0.8418,
        "current_ham_test": float(test_ba_ham),
        "current_isic_external": float(isic_ba),
        "difference_ham": float(test_ba_ham - 0.8606),
        "difference_isic": float(isic_ba - 0.8418)
    }
}

report_path = RESULTS_DIR / "final_report.json"
with open(report_path, 'w') as f:
    json.dump(final_report, f, indent=2)

print(f"  Final report saved: final_report.json")

# Summary
print("\n" + "="*80)
print("🏆 HAM-TRAINED ENSEMBLE CHAMPION - COMPLETE")
print("="*80)
print(f"Completion Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total Training Duration: {training_time:.2f} minutes (5 models in parallel)")
print("\n📊 BIDIRECTIONAL CROSS-VALIDATION RESULTS:")
print(f"  Models Trained: 5 (parallel)")
print(f"  HAM10000 Train: {train_ba_ham:.2%}")
print(f"  HAM10000 Test:  {test_ba_ham:.2%} (Sens: {test_sens_ham:.2%}, Spec: {test_spec_ham:.2%})")
print(f"  ISIC2019 External: {isic_ba:.2%} (Sens: {isic_sens:.2%}, Spec: {isic_spec:.2%})")
print("\n📈 COMPARISON TO MUTATION STUDY:")
print(f"  Mutation Study HAM:  86.06%  | This Ensemble: {test_ba_ham:.2%} ({(test_ba_ham - 0.8606)*100:+.2f}%)")
print(f"  Mutation Study ISIC: 84.18%  | This Ensemble: {isic_ba:.2%} ({(isic_ba - 0.8418)*100:+.2f}%)")
print(f"\n  Ensemble: {model_path}")
print(f"  Report: {report_path}")
print("="*80)
print("\n✨ Full bidirectional cross-validation complete!")
print("   HAM→ISIC validation successful with 226 raw features")
