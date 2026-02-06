"""
Champion Pipeline Phase F-G: Feature Interaction Selection
Generates 25,651 degree-2 interaction candidates and ranks them using 
Mutual Information to select the Top 100 Champion Features.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_classif

# Configuration
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_FILE = SCRIPT_DIR / "binary_features_all.csv"
RANDOM_SEED = 42
METADATA_FILE = SCRIPT_DIR / "feature_metadata.json"

def run_feature_selection_pipeline():
    if not DATA_FILE.exists():
        print(f"Error: {DATA_FILE} not found in parent directory.")
        return

    print("Step 1: Loading 167 Base Features...")
    df = pd.read_csv(DATA_FILE)
    drop_cols = ['image_id', 'dx', 'label', 'lesion_id', 'image', 'label_name']
    feature_cols = sorted([c for c in df.columns if c not in drop_cols])
    
    X = df[feature_cols].values
    y = df['label'].values
    X = np.nan_to_num(X)
    
    # Split
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    
    print("Step 2: Generating 25,651 Polynomial Interactions (Degree 2)...")
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    # We fit to get names
    poly.fit(X_train) 
    interaction_names = poly.get_feature_names_out(feature_cols)
    
    # Transform in batches to save memory
    n_features = len(interaction_names)
    print(f"Total candidates: {n_features}")
    
    # Note: We don't transform the whole X_train here to save RAM. 
    # We do it during MI calculation in batches.
    
    print("Step 3: Ranking by Mutual Information (Batched)...")
    batch_size = 2000
    mi_scores = np.zeros(n_features)
    
    for i in range(0, n_features, batch_size):
        end = min(i + batch_size, n_features)
        # Extract specific interaction columns for this batch
        # This is a bit slow but memory safe
        X_batch = poly.transform(X_train)[:, i:end]
        
        batch_scores = mutual_info_classif(
            X_batch, 
            y_train, 
            random_state=RANDOM_SEED
        )
        mi_scores[i:end] = batch_scores
        print(f"  Processed {end}/{n_features}...")

    print("Step 4: Selecting Top 100 Champion Features...")
    top_indices = np.argsort(mi_scores)[-100:][::-1]
    top_feature_names = [interaction_names[j] for j in top_indices]
    
    # Save Metadata
    metadata = {
        "base_features": feature_cols,
        "top_indices": top_indices.tolist(),
        "top_feature_names": top_feature_names,
        "seed": RANDOM_SEED
    }
    
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"SUCCESS! Metadata saved to {METADATA_FILE}")
    print(f"Top 5 Bio-Markers Found:")
    for i, name in enumerate(top_feature_names[:5]):
        print(f"  {i+1}. {name}")

if __name__ == "__main__":
    run_feature_selection_pipeline()
