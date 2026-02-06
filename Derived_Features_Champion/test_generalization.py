"""
Comprehensive Evaluation of ISIC2019 Models on HAM10000 Dataset.
Handles:
1. Feature Extraction (ABCD, Wavelets, Hu, Correlo, CHOG)
2. Disease class mapping (7 classes -> Binary)
3. Model evaluation (Standard, Resampled, Derived, Ensembles)
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import Counter
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report

# 1. PATH SETUP
SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR.parent
RESOURCES_DIR = BASE_DIR / "Resources"
MODELS_DIR = BASE_DIR / "Models"
HAM10000_DIR = Path(r"C:\Users\umair\Videos\PhD\PhD Data\Week 15 January\Code V2\Dataset\clean\CleanData\HAM10000")
HAM_IMAGES_DIR = HAM10000_DIR / "images"
HAM_METADATA = HAM10000_DIR / "HAM10000_metadata"

# Add Resources to path for feature_engine
if str(RESOURCES_DIR) not in sys.path:
    sys.path.append(str(RESOURCES_DIR))

# Add SCRIPT_DIR to path to ensure LOCAL feature_engine.py is used
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import feature_engine
except ImportError:
    print("Error: Could not import feature_engine.py from Resources.")
    sys.exit(1)

# Add Local ExSTraCS to path
EXSTRACS_LOCAL_PATH = SCRIPT_DIR / "scikit-ExSTraCS-master"
if EXSTRACS_LOCAL_PATH.exists():
    sys.path.append(str(EXSTRACS_LOCAL_PATH))

# 2. CONFIGURATION
RANDOM_SEED = 42
BATCH_SIZE = 50 # For feature extraction saving
LIMIT_SAMPLES = None # Set to e.g. 500 for a quick test run

# Label Mapping (HAM10000 dx -> Binary)
MALIGNANT_CLASSES = ['mel', 'bcc', 'akiec']
BENIGN_CLASSES = ['nv', 'bkl', 'df', 'vasc']

# 3. MODEL DEFINITIONS (Local references)
MODEL_CONFIGS = {
    "Derived_Features": {
        "model": str(SCRIPT_DIR / "batched_fe_model.pkl"),
        "metadata": str(SCRIPT_DIR / "feature_metadata.json")
    }
}

def extract_features_for_batch(image_ids):
    all_features = []
    skipped = []
    
    for img_id in image_ids:
        img_path = HAM_IMAGES_DIR / f"{img_id}.jpg"
        if not img_path.exists():
            skipped.append(img_id)
            continue
            
        try:
            feats = feature_engine.extract_raw_features_only(str(img_path))
            feats['image_id'] = img_id
            all_features.append(feats)
        except Exception as e:
            print(f"Error processing {img_id}: {e}")
            skipped.append(img_id)
            
    return all_features, skipped

def load_or_extract_features():
    csv_path = SCRIPT_DIR / "HAM10000_extracted_features.csv"
    
    if csv_path.exists():
        print(f"Loading existing features from {csv_path}...")
        df = pd.read_csv(csv_path)
        # Check if we need more
        metadata = pd.read_csv(HAM_METADATA)
        if LIMIT_SAMPLES:
            metadata = metadata.sample(n=LIMIT_SAMPLES, random_state=RANDOM_SEED)
        
        existing_ids = set(df['image_id'].tolist())
        image_ids = [idx for idx in metadata['image_id'].tolist() if idx not in existing_ids]
        
        if not image_ids:
            return df
        print(f"Resuming extraction for {len(image_ids)} missing images...")
        processed_data = df.to_dict('records')
    else:
        print("Starting Feature Extraction...")
        metadata = pd.read_csv(HAM_METADATA)
        if LIMIT_SAMPLES:
            metadata = metadata.sample(n=LIMIT_SAMPLES, random_state=RANDOM_SEED)
            print(f"Limiting to {LIMIT_SAMPLES} random samples.")
        image_ids = metadata['image_id'].tolist()
        processed_data = []

    total = len(image_ids)
    if total == 0: return pd.DataFrame(processed_data)

    from multiprocessing import Pool
    num_workers = min(os.cpu_count(), 4) # Avoid overloading
    
    for i in range(0, total, BATCH_SIZE):
        batch = image_ids[i:i+BATCH_SIZE]
        print(f"[{time.strftime('%H:%M:%S')}] Processing batch {i//BATCH_SIZE + 1} ({len(batch)} images)...")
        
        with Pool(num_workers) as p:
            # We need a wrapper because extract_raw_features_only only takes one arg
            results = p.map(extract_single_wrapper, batch)
        
        for r in results:
            if r: processed_data.append(r)
        
        # Intermediate Save
        pd.DataFrame(processed_data).to_csv(csv_path, index=False)
        
    return pd.DataFrame(processed_data)

def extract_single_wrapper(img_id):
    import feature_engine
    img_path = HAM_IMAGES_DIR / f"{img_id}.jpg"
    try:
        # Use local feature_engine
        feats = feature_engine.extract_raw_features_only(str(img_path))
        feats['image_id'] = img_id
        return feats
    except:
        return None

def run_evaluation():
    # 1. Data Prep
    df_features = load_or_extract_features()
    df_meta = pd.read_csv(HAM_METADATA)
    
    # Merge label info
    df = df_features.merge(df_meta[['image_id', 'dx']], on='image_id')
    df['label'] = df['dx'].apply(lambda x: 1 if x in MALIGNANT_CLASSES else 0)
    
    y_true = df['label'].values
    
    # Get base features (must match alphabetical order used in training)
    # The training script used feature_cols = [c for c in df.columns if c not in drop_cols]
    drop_cols = ['image_id', 'dx', 'label', 'lesion_id', 'image', 'label_name', 'dx_type', 'age', 'sex', 'localization', 'dataset']
    feature_cols = sorted([c for c in df.columns if c not in drop_cols])
    
    X_base = df[feature_cols].values
    X_base = np.nan_to_num(X_base)
    
    results_summary = []

    # 2. Test Standard Models (Skipped in local champion folder)
    print("Skipping standard models (this folder is for the Derived Features Champion only).")

    # 3. Test Derived Model
    derived_conf = MODEL_CONFIGS["Derived_Features"]
    if os.path.exists(derived_conf["model"]) and os.path.exists(derived_conf["metadata"]):
        print("Evaluating Derived Features Model...")
        with open(derived_conf["metadata"], 'r') as f:
            meta = json.load(f)
        
        # Ensure base features align
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        # We need to fit with the SAME base features as training
        # The metadata saved 'base_features' which we should use to ensure order
        base_cols_train = meta['base_features']
        X_base_aligned = df[base_cols_train].values
        X_base_aligned = np.nan_to_num(X_base_aligned)
        
        X_poly = poly.fit_transform(X_base_aligned)
        X_derived = X_poly[:, meta['top_indices']]
        
        with open(derived_conf["model"], 'rb') as f:
            model = pickle.load(f)
        
        y_pred = model.predict(X_derived)
        metrics = calculate_metrics(y_true, y_pred, "Derived_Features")
        results_summary.append(metrics)

    # 4. Ensembles (Skipped in local champion folder)
    print("Skipping Ensembles (this folder is for the Derived Features Champion only).")

    # 5. Final Output
    df_results = pd.DataFrame(results_summary)
    df_results.to_csv(SCRIPT_DIR / "HAM10000_Comparison_Results.csv", index=False)
    print("\n" + "="*50)
    print("FINISHED ALL EVALUATIONS")
    print("="*50)
    print(df_results[['Model', 'Balanced_Accuracy', 'Sensitivity', 'Specificity']])

def calculate_metrics(y_true, y_pred, model_name):
    # Skip if all zeros (failed)
    if len(np.unique(y_pred)) == 1:
        print(f"Warning: Model {model_name} predicted only one class.")
        
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "Model": model_name,
        "Balanced_Accuracy": bal_acc,
        "Sensitivity": sens,
        "Specificity": spec,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }

if __name__ == "__main__":
    run_evaluation()
