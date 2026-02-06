"""
SEQUENTIAL FINE-TUNING EXPERIMENT (Transfer Learning)
Location: C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/Derived_Features_Champion/Transfer_Learning_Experiment

Protocol:
1. Load model pre-trained on ISIC 2019 (Science Informed Champion).
2. Prepare datasets (HAM and ISIC) with EXACT feature alignment (Top 100 Interactions).
3. Resume training using HAM10000 dataset.
4. Evaluate on BOTH ISIC (Source) and HAM (Target) to check for catastrophic forgetting.
5. Generate training dynamics plots using 'exstracs_viz'.
6. Add specialized visualizations from the ExSTraCS User Guide (ROC/PRC, Attribute Importance).
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
from sklearn.preprocessing import PolynomialFeatures

# Paths
BASE_DIR = Path("C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code")
REPO_PATH = BASE_DIR / "Derived_Features_Champion"
EXP_DIR = REPO_PATH / "Transfer_Learning_Experiment"
RESULTS_DIR = EXP_DIR / "Results"
PLOTS_DIR = RESULTS_DIR / "Plots"
MUTATION_DIR = BASE_DIR / "GA_Mutation_Study"
MUTATION_DATA_DIR = MUTATION_DIR / "data"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

# Setup Paths for imports (using the one inside the repo)
sys.path.insert(0, str(REPO_PATH / "scikit-ExSTraCS-master"))
from skExSTraCS.ExSTraCS import ExSTraCS

# 1. LOAD FEATURE METADATA FOR ALIGNMENT
print("Loading feature metadata for alignment...")
with open(REPO_PATH / "feature_metadata.json") as f:
    metadata = json.load(f)
    base_feature_names = metadata['base_features']
    top_indices = metadata['top_indices']
    top_feature_names = metadata['top_feature_names']

def align_dataset(df, base_names, indices):
    """Aligns a base dataframe to the Top 100 interaction features."""
    print(f"  Transforming dataset (Initial columns: {len(df.columns)})")
    # Ensure correct base feature order
    X_base = df[base_names].values
    
    # Apply degree-2 polynomial expansion (Interaction features)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_base)
    
    # Select the Top 100
    X_top100 = X_poly[:, indices]
    return X_top100

# 2. PREPARE THE SOURCE MODEL
SOURCE_MODEL_PATH = EXP_DIR / "source_model.pkl"
LOCAL_REBOOT_PATH = EXP_DIR / "reboot_metadata.pkl"

if not SOURCE_MODEL_PATH.exists():
    print(f"Error: Source model not found at {SOURCE_MODEL_PATH}")
    sys.exit(1)

print(f"Loading source model: {SOURCE_MODEL_PATH.name}...")
with open(SOURCE_MODEL_PATH, 'rb') as f:
    model_obj = pickle.load(f)

orig_iterations = model_obj.iterationCount
print(f"Source model trained for {orig_iterations} iterations on ISIC.")

# Manually construct the reboot list format expected by ExSTraCS
# Format from ExSTraCS.py (saveFinalMetrics): [0:14] Stats, [15] AT, [16] env, [17] popSet
dummy_metrics = [orig_iterations] + [0.0]*14 + [
    model_obj.AT, 
    model_obj.env, 
    model_obj.population.popSet
]

with open(LOCAL_REBOOT_PATH, 'wb') as f:
    pickle.dump(dummy_metrics, f)

# 3. LOAD AND ALIGN HAM DATA FOR FINE-TUNING
print("\nPreparing HAM10000 data for fine-tuning...")
ham_df = pd.read_csv(MUTATION_DATA_DIR / "ham_clean.csv")
X_ham_full = align_dataset(ham_df, base_feature_names, top_indices)
y_ham_full = ham_df['label'].values

# splits
ham_train_idx = np.load(MUTATION_DATA_DIR / "ham_train_indices.npy")
ham_val_idx = np.load(MUTATION_DATA_DIR / "ham_val_indices.npy")
X_ham_train, y_ham_train = X_ham_full[ham_train_idx], y_ham_full[ham_train_idx]
X_ham_val, y_ham_val = X_ham_full[ham_val_idx], y_ham_full[ham_val_idx]

# 4. INITIALIZE MODEL FOR RESUMED TRAINING
ADDITIONAL_ITERATIONS = 50000 
print(f"\nInitializing model for {ADDITIONAL_ITERATIONS} additional iterations...")
# Using default parameters for consistency unless specified
model = ExSTraCS(learning_iterations=ADDITIONAL_ITERATIONS, reboot_filename=str(LOCAL_REBOOT_PATH), track_accuracy_while_fit=True)

# 5. RESUME TRAINING ON HAM
print(f"--- FINE-TUNING ON HAM10000 (Target Features: {X_ham_train.shape[1]}) ---")
start_time = time.time()
model.fit(X_ham_train, y_ham_train)
duration = (time.time() - start_time) / 60
print(f"Fine-tuning completed in {duration:.2f} minutes.")

# 6. EXPORT TRACKING DATA FOR VISUALIZATION
TRACKING_CSV = RESULTS_DIR / "iterationData.csv"
model.export_iteration_tracking_data(str(TRACKING_CSV))
print(f"Exported tracking data to {TRACKING_CSV}")

# 7. EVALUATION
print("\n--- EVALUATING PERFORMANCE ---")

# A. Domain Adaptation (HAM)
y_ham_pred = model.predict(X_ham_val)
ham_ba = balanced_accuracy_score(y_ham_val, y_ham_pred)
ham_probs = model.predict_proba(X_ham_val)

# B. Catastrophic Forgetting check (ISIC)
print("Loading and aligning ISIC2019 test data...")
isic_df = pd.read_csv(MUTATION_DATA_DIR / "isic_clean.csv")
isic_val_idx = np.load(MUTATION_DATA_DIR / "isic_val_indices.npy")
X_isic_val_base = align_dataset(isic_df, base_feature_names, top_indices)
X_isic_val = X_isic_val_base[isic_val_idx]
y_isic_val = isic_df['label'].values[isic_val_idx]

y_isic_pred = model.predict(X_isic_val)
isic_ba = balanced_accuracy_score(y_isic_val, y_isic_pred)
isic_probs = model.predict_proba(X_isic_val)

print(f"\nPost-Fine-Tuning Stats:")
print(f"  HAM Val BA:   {ham_ba:.4f} (Original Target Baseline: ~0.6282)")
print(f"  ISIC Test BA: {isic_ba:.4f} (Original Source Baseline: 0.7137)")

# 8. GENERATE DYNAMICS PLOTS USING exstracs_viz
print("\nGenerating training dynamics plots...")
python_exe = sys.executable
env = os.environ.copy()
env["PYTHONPATH"] = str(BASE_DIR) + os.pathsep + env.get("PYTHONPATH", "")

try:
    subprocess.run([
        python_exe, "-m", "exstracs_viz.cli",
        str(TRACKING_CSV),
        "--out", str(PLOTS_DIR)
    ], env=env, check=True)
    print(f"✅ Dynamics plots generated.")
except Exception as e:
    print(f"⚠️ exstracs_viz failed: {e}")

# 9. ADDITIONAL VISUALIZATIONS (from User Guide)
print("Generating additional visualizations (ROC/PRC, Attribute Importance)...")

def plot_curves(y_true, y_probs, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    prec, rec, _ = precision_recall_curve(y_true, y_probs[:, 1])
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC (AUC={auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC Curve - {title}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(rec, prec, label=f'PRC (AUC={auc(rec, prec):.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve - {title}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename)
    plt.close()

plot_curves(y_ham_val, ham_probs, "HAM (Domain Adaptation)", "ham_curves.png")
plot_curves(y_isic_val, isic_probs, "ISIC (Catastrophic Forgetting)", "isic_curves.png")

# Attribute Importance Plot
attr_spec = model.get_final_attribute_specificity_list()
plt.figure(figsize=(12, 6))
plt.bar(range(len(attr_spec)), attr_spec)
plt.title("Final Attribute Specificity (Rule Coverage)")
plt.xlabel("Feature Index (Top 100 Interaction Space)")
plt.ylabel("Specificity Count")
plt.savefig(PLOTS_DIR / "attribute_importance.png")
plt.close()

# 10. SAVE FINAL LOGS AND MODEL
final_model_path = RESULTS_DIR / "fine_tuned_champion.pkl"
model.pickle_model(str(final_model_path))

report = {
    "source_dataset": "ISIC2019",
    "target_dataset": "HAM10000",
    "orig_isic_ba": 0.7137,
    "orig_ham_ba": 0.6282,
    "final_isic_ba": isic_ba,
    "final_ham_ba": ham_ba,
    "isic_delta": isic_ba - 0.7137,
    "ham_delta": ham_ba - 0.6282,
    "iterations_on_ham": ADDITIONAL_ITERATIONS,
    "total_training_duration_min": duration
}

with open(RESULTS_DIR / "fine_tune_report.json", 'w') as f:
    json.dump(report, f, indent=2)

print("\n--- EXPERIMENT COMPLETE ---")
print(f"Results saved in: {RESULTS_DIR}")
print(f"Adapted Model saved to: {final_model_path}")
