"""
Phase 1.2: Optimal Threshold Tuning
Maximizes Youden's J statistic to find optimal probability threshold.
Clinical priority: Favor sensitivity over specificity.
"""
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score, confusion_matrix

# Add ExSTraCS to path
CHAMPION_PATH = Path(__file__).parent.parent / "Derived_Features_Champion"
sys.path.insert(0, str(CHAMPION_PATH / "scikit-ExSTraCS-master"))

# Paths
RESULTS_DIR = Path(__file__).parent / "phase1_results"
MODELS_DIR = Path(__file__).parent / "models"
RESULTS_DIR.mkdir(exist_ok=True)

print("="*60)
print("PHASE 1.2: OPTIMAL THRESHOLD TUNING")
print("="*60)

# Load best model from Phase 1.1
model_path = MODELS_DIR / "best_hyperparam_model_fast.pkl"
if not model_path.exists():
    model_path = MODELS_DIR / "best_hyperparam_model.pkl"

if not model_path.exists():
    print(f"ERROR: Model not found at {MODELS_DIR / 'best_hyperparam_model_fast.pkl'} or {MODELS_DIR / 'best_hyperparam_model.pkl'}")
    sys.exit(1)

print(f"\nLoading model from: {model_path}")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load test data (pre-generated Top 100 features)
data_path = Path(__file__).parent / "champion_top100_features.csv"
if not data_path.exists():
    print(f"ERROR: {data_path} not found. Please run generate_top100_dataset.py first.")
    sys.exit(1)

print(f"Loading dataset from: {data_path}")
df_top = pd.read_csv(data_path)
X = df_top.drop('label', axis=1).values
y = df_top['label'].values

# Get predicted probabilities
print("Computing predicted probabilities...")
y_proba = model.predict_proba(X)[:, 1]  # Probability of malignant class

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y, y_proba)
roc_auc = auc(fpr, tpr)

# Calculate metrics for each threshold
print("\nFinding optimal threshold...")
results = []

for threshold in np.linspace(0.1, 0.9, 81):
    y_pred = (y_proba >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    # Metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc = (sensitivity + specificity) / 2
    
    # Youden's J statistic
    youden_j = sensitivity + specificity - 1
    
    results.append({
        'threshold': threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'balanced_accuracy': balanced_acc,
        'youden_j': youden_j,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    })

results_df = pd.DataFrame(results)

# Find optimal thresholds
best_youden_idx = results_df['youden_j'].idxmax()
best_ba_idx = results_df['balanced_accuracy'].idxmax()

optimal_threshold_youden = results_df.loc[best_youden_idx, 'threshold']
optimal_threshold_ba = results_df.loc[best_ba_idx, 'threshold']

print(f"\nOptimal Threshold (Youden's J): {optimal_threshold_youden:.3f}")
print(f"  Sensitivity: {results_df.loc[best_youden_idx, 'sensitivity']:.4f}")
print(f"  Specificity: {results_df.loc[best_youden_idx, 'specificity']:.4f}")
print(f"  Balanced Accuracy: {results_df.loc[best_youden_idx, 'balanced_accuracy']:.4f}")
print(f"  Youden's J: {results_df.loc[best_youden_idx, 'youden_j']:.4f}")

print(f"\nOptimal Threshold (Max BA): {optimal_threshold_ba:.3f}")
print(f"  Balanced Accuracy: {results_df.loc[best_ba_idx, 'balanced_accuracy']:.4f}")

# Default threshold (0.5) for comparison
default_idx = results_df.iloc[(results_df['threshold'] - 0.5).abs().argsort()[:1]].index[0]
print(f"\nDefault Threshold (0.5) Performance:")
print(f"  Sensitivity: {results_df.loc[default_idx, 'sensitivity']:.4f}")
print(f"  Specificity: {results_df.loc[default_idx, 'specificity']:.4f}")
print(f"  Balanced Accuracy: {results_df.loc[default_idx, 'balanced_accuracy']:.4f}")

# Calculate improvement
ba_improvement = results_df.loc[best_youden_idx, 'balanced_accuracy'] - results_df.loc[default_idx, 'balanced_accuracy']
print(f"\n🎯 BA Improvement from threshold tuning: +{ba_improvement:.4f} ({ba_improvement*100:.2f}%)")

# Save results
results_df.to_csv(RESULTS_DIR / "threshold_optimization_results.csv", index=False)

# ============================================================
# VISUALIZATION
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. ROC Curve
axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
axes[0, 0].set_xlabel('False Positive Rate', fontsize=12)
axes[0, 0].set_ylabel('True Positive Rate', fontsize=12)
axes[0, 0].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[0, 0].legend(loc='lower right')
axes[0, 0].grid(alpha=0.3)

# 2. Sensitivity & Specificity vs Threshold
axes[0, 1].plot(results_df['threshold'], results_df['sensitivity'], 'b-', label='Sensitivity', lw=2)
axes[0, 1].plot(results_df['threshold'], results_df['specificity'], 'r-', label='Specificity', lw=2)
axes[0, 1].axvline(optimal_threshold_youden, color='green', linestyle='--', lw=2, label=f'Optimal ({optimal_threshold_youden:.3f})')
axes[0, 1].axvline(0.5, color='gray', linestyle=':', lw=2, label='Default (0.5)')
axes[0, 1].set_xlabel('Threshold', fontsize=12)
axes[0, 1].set_ylabel('Score', fontsize=12)
axes[0, 1].set_title('Sensitivity & Specificity vs Threshold', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Youden's J vs Threshold
axes[1, 0].plot(results_df['threshold'], results_df['youden_j'], 'purple', lw=2)
axes[1, 0].axvline(optimal_threshold_youden, color='green', linestyle='--', lw=2, label=f'Max J ({optimal_threshold_youden:.3f})')
axes[1, 0].axhline(results_df.loc[best_youden_idx, 'youden_j'], color='green', linestyle=':', alpha=0.5)
axes[1, 0].set_xlabel('Threshold', fontsize=12)
axes[1, 0].set_ylabel("Youden's J Statistic", fontsize=12)
axes[1, 0].set_title("Youden's J = Sensitivity + Specificity - 1", fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 4. Balanced Accuracy vs Threshold
axes[1, 1].plot(results_df['threshold'], results_df['balanced_accuracy'], 'orange', lw=2)
axes[1, 1].axvline(optimal_threshold_ba, color='darkgreen', linestyle='--', lw=2, label=f'Max BA ({optimal_threshold_ba:.3f})')
axes[1, 1].axvline(0.5, color='gray', linestyle=':', lw=2, label='Default (0.5)')
axes[1, 1].set_xlabel('Threshold', fontsize=12)
axes[1, 1].set_ylabel('Balanced Accuracy', fontsize=12)
axes[1, 1].set_title('Balanced Accuracy vs Threshold', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
fig_path = RESULTS_DIR / "threshold_optimization.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\nSaved visualization to: {fig_path}")

# Save optimal threshold
threshold_config = {
    'optimal_threshold_youden': float(optimal_threshold_youden),
    'optimal_threshold_ba': float(optimal_threshold_ba),
    'ba_at_youden': float(results_df.loc[best_youden_idx, 'balanced_accuracy']),
    'ba_at_default': float(results_df.loc[default_idx, 'balanced_accuracy']),
    'improvement': float(ba_improvement),
    'roc_auc': float(roc_auc)
}

with open(RESULTS_DIR / "optimal_threshold.json", 'w') as f:
    json.dump(threshold_config, f, indent=2)

print("\n✅ Phase 1.2 Complete! Run phase1_degree3_features.py next.")
