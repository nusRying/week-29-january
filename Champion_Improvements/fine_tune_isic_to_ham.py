"""
SEQUENTIAL FINE-TUNING EXPERIMENT (Transfer Learning)
Protocol:
1. Load model pre-trained on ISIC 2019.
2. Resume training using HAM10000 dataset.
3. Evaluate on BOTH ISIC (Internal) and HAM (External) to check for catastrophic forgetting.
"""
import sys
import numpy as np
import pandas as pd
import pickle
import json
import time
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score

# Paths
CHAMPION_PATH = Path("C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/Derived_Features_Champion")
IMPROV_PATH = Path("C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/Champion_Improvements")
MUTATION_DIR = Path("C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/GA_Mutation_Study")
sys.path.insert(0, str(CHAMPION_PATH / "scikit-ExSTraCS-master"))

from skExSTraCS.ExSTraCS import ExSTraCS

# Data Paths
MUTATION_DATA_DIR = MUTATION_DIR / "data"
RESULTS_DIR = IMPROV_PATH / "fine_tuning_results"
MODELS_DIR = IMPROV_PATH / "models"
RESULTS_DIR.mkdir(exist_ok=True)

# 1. LOAD MODEL (ISIC Champion)
BASE_MODEL_PATH = MODELS_DIR / "science_informed_champion.pkl"  # This was trained on ISIC
print(f"Loading base model: {BASE_MODEL_PATH.name}...")
with open(BASE_MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# PRE-FINE-TUNING STATS (From Report)
print("\n--- PRE-FINE-TUNING PERFORMANCE (ISIC TRAINED) ---")
print("ISIC Test BA: 71.37%")
print("HAM External BA: 62.82%")

# 2. LOAD HAM DATA FOR FINE-TUNING
print("\nLoading HAM10000 data for fine-tuning...")
ham_df = pd.read_csv(MUTATION_DATA_DIR / "ham_clean.csv")
X_ham = ham_df.drop('label', axis=1).values
y_ham = ham_df['label'].values

# Splits
ham_train_idx = np.load(MUTATION_DATA_DIR / "ham_train_indices.npy")
ham_val_idx = np.load(MUTATION_DATA_DIR / "ham_val_indices.npy")
X_ham_train, y_ham_train = X_ham[ham_train_idx], y_ham[ham_train_idx]
X_ham_val, y_ham_val = X_ham[ham_val_idx], y_ham[ham_val_idx]

# 3. RESUME TRAINING ON HAM
print("\n--- RESUMING TRAINING ON HAM10000 (100k iterations) ---")
print("Target: Refine ISIC rules with HAM domain knowledge.")
model.learning_iterations = 100000  # Set additional iterations
start_time = time.time()
model.fit(X_ham_train, y_ham_train)
duration = (time.time() - start_time) / 60
print(f"Fine-tuning completed in {duration:.2f} minutes.")

# 4. LOAD ISIC TEST DATA FOR EVALUATION
print("\nLoading ISIC2019 test data to check for forgetting...")
isic_df = pd.read_csv(MUTATION_DATA_DIR / "isic_clean.csv")
isic_val_idx = np.load(MUTATION_DATA_DIR / "isic_val_indices.npy")
X_isic_val = isic_df.drop('label', axis=1).values[isic_val_idx]
y_isic_val = isic_df['label'].values[isic_val_idx]

# 5. EVALUATION
print("\n--- EVALUATING POST-FINE-TUNING ---")
y_isic_pred = model.predict(X_isic_val)
y_ham_pred = model.predict(X_ham_val)

isic_ba = balanced_accuracy_score(y_isic_val, y_isic_pred)
ham_ba = balanced_accuracy_score(y_ham_val, y_ham_pred)

print(f"NEW ISIC Test BA: {isic_ba:.4f} (Original: 0.7137)")
print(f"NEW HAM Val BA:   {ham_ba:.4f} (Original: 0.6282)")

# Compare
isic_diff = isic_ba - 0.7137
ham_diff = ham_ba - 0.6282

print(f"\nISIC Delta: {isic_diff:+.2%} ({'Forgotten' if isic_diff < 0 else 'Improved'})")
print(f"HAM Delta:  {ham_diff:+.2%} ({'Improved' if ham_diff > 0 else 'Failed'})")

# 6. SAVE RESULTS
final_model_path = MODELS_DIR / "fine_tuned_isic_to_ham.pkl"
with open(final_model_path, 'wb') as f:
    pickle.dump(model, f)

report = {
    "isic_orig": 0.7137,
    "ham_orig": 0.6282,
    "isic_new": isic_ba,
    "ham_new": ham_ba,
    "isic_delta": isic_diff,
    "ham_delta": ham_diff,
    "fine_tune_iterations": 100000
}

with open(RESULTS_DIR / "fine_tune_report.json", 'w') as f:
    json.dump(report, f, indent=2)

print(f"\nFine-tuned model saved to {final_model_path}")
