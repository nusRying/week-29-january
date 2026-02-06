"""
UNIFIED CHAMPION JOINT TRAINING (ISIC + HAM)
Location: C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/Unified_Champion_Experiment

Protocol:
1. Load both ISIC2019 and HAM10000 datasets.
2. Align features using Top 100 Interactions (from metadata).
3. Concatenate and Shuffle training instances for joint learning.
4. Train ExSTraCS for 500,000 iterations (Science Informed config).
5. Dual Evaluation: Benchmark on separate ISIC and HAM test sets.
6. Generate Dynamics and Advanced Plots.
"""
import sys
import os
import numpy as np
import pandas as pd
import pickle
import json
import time
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.utils import shuffle

# Paths
BASE_DIR = Path("C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code")
REPO_PATH = BASE_DIR / "Derived_Features_Champion"
EXP_DIR = BASE_DIR / "Unified_Champion_Experiment"
RESULTS_DIR = EXP_DIR / "Results"
PLOTS_DIR = RESULTS_DIR / "Plots"
MUTATION_DIR = BASE_DIR / "GA_Mutation_Study"
MUTATION_DATA_DIR = MUTATION_DIR / "data"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

# Setup Paths for imports
sys.path.insert(0, str(REPO_PATH / "scikit-ExSTraCS-master"))
from skExSTraCS.ExSTraCS import ExSTraCS

# 1. LOAD FEATURE METADATA FOR ALIGNMENT
print("Loading feature metadata for alignment...")
with open(REPO_PATH / "feature_metadata.json") as f:
    metadata = json.load(f)
    base_feature_names = metadata['base_features']
    top_indices = metadata['top_indices']

def align_dataset(df, base_names, indices):
    """Aligns a base dataframe to the Top 100 interaction features."""
    X_base = df[base_names].values
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_base)
    X_top100 = X_poly[:, indices]
    return X_top100

# 2. PREPARE JOINT DATASET
print("\nPreparing Joint Dataset (ISIC + HAM)...")

# Load ISIC
print("  - Processing ISIC2019...")
isic_df = pd.read_csv(MUTATION_DATA_DIR / "isic_clean.csv")
isic_train_idx = np.load(MUTATION_DATA_DIR / "isic_train_indices.npy")
isic_val_idx = np.load(MUTATION_DATA_DIR / "isic_val_indices.npy")
X_isic_full = align_dataset(isic_df, base_feature_names, top_indices)
y_isic_full = isic_df['label'].values

X_isic_train, y_isic_train = X_isic_full[isic_train_idx], y_isic_full[isic_train_idx]
X_isic_val, y_isic_val = X_isic_full[isic_val_idx], y_isic_full[isic_val_idx]

# Load HAM
print("  - Processing HAM10000...")
ham_df = pd.read_csv(MUTATION_DATA_DIR / "ham_clean.csv")
ham_train_idx = np.load(MUTATION_DATA_DIR / "ham_train_indices.npy")
ham_val_idx = np.load(MUTATION_DATA_DIR / "ham_val_indices.npy")
X_ham_full = align_dataset(ham_df, base_feature_names, top_indices)
y_ham_full = ham_df['label'].values

X_ham_train, y_ham_train = X_ham_full[ham_train_idx], y_ham_full[ham_train_idx]
X_ham_val, y_ham_val = X_ham_full[ham_val_idx], y_ham_full[ham_val_idx]

# CONCATENATE AND SHUFFLE
print(f"\nConcatenating: ISIC Train ({len(X_isic_train)}) + HAM Train ({len(X_ham_train)})...")
X_train_joint = np.vstack([X_isic_train, X_ham_train])
y_train_joint = np.concatenate([y_isic_train, y_ham_train])

# JOINT NORMALIZATION
print("Applying Joint Normalization (MinMaxScaler)...")
scaler = MinMaxScaler()
X_train_joint = scaler.fit_transform(X_train_joint)

# Transform test/val sets using the JOINT scaler
X_isic_val = scaler.transform(X_isic_val)
X_ham_val = scaler.transform(X_ham_val)

X_train_joint, y_train_joint = shuffle(X_train_joint, y_train_joint, random_state=42)
print(f"Total Joint Training Samples: {len(X_train_joint)}")

# 3. INITIALIZE MODEL
# Using Model D configuration: N=3000, nu=10, chi=0.8, mu=0.04 (Science Informed)
print("\nInitializing Unified Champion Model...")
ITERATIONS = 500000
model = ExSTraCS(
    learning_iterations=ITERATIONS,
    N=3000,
    nu=10,
    chi=0.8,
    mu=0.04,
    track_accuracy_while_fit=True
)

# 4. TRAINING
print(f"--- STARTING JOINT TRAINING (500k Iterations) ---")
start_time = time.time()
model.fit(X_train_joint, y_train_joint)
duration_hrs = (time.time() - start_time) / 3600
print(f"Training completed in {duration_hrs:.2f} hours.")

# 5. EXPORT TRACKING DATA
TRACKING_CSV = RESULTS_DIR / "iterationData.csv"
model.export_iteration_tracking_data(str(TRACKING_CSV))

# 6. EVALUATION
print("\n--- DUAL-DOMAIN EVALUATION ---")

# A. Internal Robustness (ISIC)
y_isic_pred = model.predict(X_isic_val)
isic_ba = balanced_accuracy_score(y_isic_val, y_isic_pred)
isic_probs = model.predict_proba(X_isic_val)

# B. Generalization (HAM)
y_ham_pred = model.predict(X_ham_val)
ham_ba = balanced_accuracy_score(y_ham_val, y_ham_pred)
ham_probs = model.predict_proba(X_ham_val)

print(f"Final Performance:")
print(f"  ISIC BA: {isic_ba:.4f} (Baseline ~0.72-0.74)")
print(f"  HAM BA:  {ham_ba:.4f}  (Baseline ~0.63-0.65)")

# 7. VISUALIZATIONS
print("\nGenerating Visualizations...")

# Training Dynamics (CLI)
python_exe = sys.executable
env = os.environ.copy()
env["PYTHONPATH"] = str(BASE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
try:
    subprocess.run([
        python_exe, "-m", "exstracs_viz.cli",
        str(TRACKING_CSV),
        "--out", str(PLOTS_DIR)
    ], env=env, check=True)
except Exception as e:
    print(f"Warning: exstracs_viz failed: {e}")

# Curves and Importance
def plot_curves(y_true, y_probs, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    prec, rec, _ = precision_recall_curve(y_true, y_probs[:, 1])
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC (AUC={auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC - {title}')
    plt.subplot(1, 2, 2)
    plt.plot(rec, prec, label=f'PRC (AUC={auc(rec, prec):.2f})')
    plt.title(f'PRC - {title}')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename)
    plt.close()

plot_curves(y_isic_val, isic_probs, "ISIC (Joint Model)", "isic_curves_joint.png")
plot_curves(y_ham_val, ham_probs, "HAM (Joint Model)", "ham_curves_joint.png")

# Importance
attr_spec = model.get_final_attribute_specificity_list()
plt.figure(figsize=(12, 6))
plt.bar(range(len(attr_spec)), attr_spec)
plt.title("Joint Model Attribute Specificity")
plt.savefig(PLOTS_DIR / "attribute_importance_joint.png")
plt.close()

# 8. SAVE
final_model_path = RESULTS_DIR / "unified_champion_model.pkl"
model.pickle_model(str(final_model_path))

report = {
    "experiment": "Unified Champion (Joint Training)",
    "iterations": ITERATIONS,
    "combined_train_size": len(X_train_joint),
    "isic_test_ba": isic_ba,
    "ham_test_ba": ham_ba,
    "training_duration_hrs": duration_hrs
}
with open(RESULTS_DIR / "unified_champion_report.json", 'w') as f:
    json.dump(report, f, indent=2)

print("\n--- UNIFIED CHAMPION COMPLETE ---")
