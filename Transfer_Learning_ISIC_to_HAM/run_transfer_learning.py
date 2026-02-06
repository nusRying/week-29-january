"""
SEQUENTIAL FINE-TUNING EXPERIMENT (Transfer Learning) - STANDALONE & NORMALIZED
Location: C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/Transfer_Learning_ISIC_to_HAM

Protocol:
1. Load ISIC2019 (Source) and HAM10000 (Target) datasets.
2. Align features using Top 100 Interactions.
3. Apply normalization (MinMaxScaler) fitted on ISIC Training data.
4. Train Stage 1: Full ISIC Training (500k iterations).
5. Train Stage 2: Fine-tuning on HAM10000 (50k iterations).
6. Evaluate both domains to quantify adaptation and forgetting.
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

# Paths
BASE_DIR = Path("C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code")
CHAMPION_PATH = BASE_DIR / "Derived_Features_Champion"
MUTATION_DIR = BASE_DIR / "GA_Mutation_Study"
TRANSFER_DIR = BASE_DIR / "Transfer_Learning_ISIC_to_HAM"
RESULTS_DIR = TRANSFER_DIR / "Results"
PLOTS_DIR = RESULTS_DIR / "Plots"
MUTATION_DATA_DIR = MUTATION_DIR / "data"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

# Setup Paths for imports
sys.path.insert(0, str(CHAMPION_PATH / "scikit-ExSTraCS-master"))
from skExSTraCS.ExSTraCS import ExSTraCS

# 1. LOAD FEATURE METADATA FOR ALIGNMENT
print("Loading feature metadata for alignment...")
with open(CHAMPION_PATH / "feature_metadata.json") as f:
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

# 2. PREPARE DATASETS
print("\n[1/5] Preparing aligned and normalized datasets...")

# Load ISIC (Source)
print("  - Processing ISIC2019...")
isic_df = pd.read_csv(MUTATION_DATA_DIR / "isic_clean.csv")
isic_train_idx = np.load(MUTATION_DATA_DIR / "isic_train_indices.npy")
isic_val_idx = np.load(MUTATION_DATA_DIR / "isic_val_indices.npy")
X_isic_full = align_dataset(isic_df, base_feature_names, top_indices)
y_isic_full = isic_df['label'].values

X_isic_train, y_isic_train = X_isic_full[isic_train_idx], y_isic_full[isic_train_idx]
X_isic_val, y_isic_val = X_isic_full[isic_val_idx], y_isic_full[isic_val_idx]

# Load HAM (Target)
print("  - Processing HAM10000...")
ham_df = pd.read_csv(MUTATION_DATA_DIR / "ham_clean.csv")
ham_train_idx = np.load(MUTATION_DATA_DIR / "ham_train_indices.npy")
ham_val_idx = np.load(MUTATION_DATA_DIR / "ham_val_indices.npy")
X_ham_full = align_dataset(ham_df, base_feature_names, top_indices)
y_ham_full = ham_df['label'].values

X_ham_train, y_ham_train = X_ham_full[ham_train_idx], y_ham_full[ham_train_idx]
X_ham_val, y_ham_val = X_ham_full[ham_val_idx], y_ham_full[ham_val_idx]

# NORMALIZATION (Protocol: Always Normalize)
print("  - Applying MinMaxScaler fitted on ISIC Training data...")
scaler = MinMaxScaler()
X_isic_train = scaler.fit_transform(X_isic_train)
X_isic_val = scaler.transform(X_isic_val)
X_ham_train = scaler.transform(X_ham_train)
X_ham_val = scaler.transform(X_ham_val)

# 3. STAGE 1: TRAIN ON ISIC (SOURCE)
SOURCE_MODEL_PATH = RESULTS_DIR / "source_isic_normalized.pkl"

if SOURCE_MODEL_PATH.exists():
    print(f"\n[2/5] STAGE 1: Found existing model at {SOURCE_MODEL_PATH}. Skipping training...")
    # Initialize a dummy model and reboot it
    source_model = ExSTraCS(
        learning_iterations=0,
        reboot_filename=str(SOURCE_MODEL_PATH)
    )
    source_duration = 0 # Duration not critical for resume
else:
    print("\n[2/5] STAGE 1: Training on ISIC (500k iterations)...")
    source_model = ExSTraCS(
        learning_iterations=500000,
        N=3000,
        nu=10,
        chi=0.8,
        mu=0.04,
        track_accuracy_while_fit=True
    )

    start_time = time.time()
    source_model.fit(X_isic_train, y_isic_train)
    source_duration = (time.time() - start_time) / 3600
    print(f"  Stage 1 complete in {source_duration:.2f} hours.")
    source_model.pickle_model(str(SOURCE_MODEL_PATH))

# 4. STAGE 2: FINE-TUNE ON HAM (TARGET)
print("\n[3/5] STAGE 2: Fine-tuning on HAM10000 (50k iterations)...")

# Reboot configuration
REBOOT_PATH = TRANSFER_DIR / "reboot_normalized.pkl"
# Construct reboot data [iterations, stats(14), AT, env, population]
dummy_metrics = [source_model.iterationCount] + [0.0]*14 + [
    source_model.AT, 
    source_model.env, 
    source_model.population.popSet
]
with open(REBOOT_PATH, 'wb') as f:
    pickle.dump(dummy_metrics, f)

ADDITIONAL_ITERATIONS = 50000
fine_tuned_model = ExSTraCS(
    learning_iterations=ADDITIONAL_ITERATIONS,
    reboot_filename=str(REBOOT_PATH),
    track_accuracy_while_fit=True
)

start_time = time.time()
fine_tuned_model.fit(X_ham_train, y_ham_train)
fine_tune_duration = (time.time() - start_time) / 60
print(f"  Fine-tuning complete in {fine_tune_duration:.2f} minutes.")

# 5. EXPORT TRACKING & EVALUATE
print("\n[4/5] Evaluating Dual-Domain Performance...")
TRACKING_CSV = RESULTS_DIR / "iterationData.csv"
fine_tuned_model.export_iteration_tracking_data(str(TRACKING_CSV))

# Evaluation
y_isic_pred = fine_tuned_model.predict(X_isic_val)
isic_ba = balanced_accuracy_score(y_isic_val, y_isic_pred)

y_ham_pred = fine_tuned_model.predict(X_ham_val)
ham_ba = balanced_accuracy_score(y_ham_val, y_ham_pred)

print(f"  Post-Fine-Tuning ISIC BA: {isic_ba:.4f}")
print(f"  Post-Fine-Tuning HAM BA:  {ham_ba:.4f}")

# 6. VISUALIZATION
print("\n[5/5] Generating Visualizations...")
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
    print(f"Warning: Dynamics plots failed: {e}")

# Save final model
FINAL_MODEL_PATH = RESULTS_DIR / "fine_tuned_normalized_sequental.pkl"
fine_tuned_model.pickle_model(str(FINAL_MODEL_PATH))

# Save summary report
report = {
    "experiment": "Sequential Transfer learning (Normalized)",
    "stage1_iterations": 500000,
    "stage2_iterations": 50000,
    "final_isic_ba": isic_ba,
    "final_ham_ba": ham_ba,
    "source_duration_hrs": source_duration,
    "fine_tune_duration_min": fine_tune_duration
}
with open(RESULTS_DIR / "normalized_transfer_report.json", 'w') as f:
    json.dump(report, f, indent=2)

print("\n--- NORMALIZED TRANSFER EXPERIMENT COMPLETE ---")
