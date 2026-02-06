"""
Helper script to generate Top 100 polynomial features dataset for Phase 1.
This creates the actual feature matrix that the champion model was trained on.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
import json

CHAMPION_PATH = Path("C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/Derived_Features_Champion")
OUTPUT_DIR = Path("C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/Champion_Improvements")

print("="*60)
print("GENERATING TOP 100 POLYNOMIAL FEATURES DATASET")
print("="*60)

# Load base features
print("\nLoading base feature dataset...")
df = pd.read_csv(CHAMPION_PATH / "binary_features_all.csv")

# Identify metadata columns
meta_cols = ['image_id', 'label', 'dx', 'age', 'sex', 'localization']
existing_meta = [c for c in meta_cols if c in df.columns]
feature_cols = [c for c in df.columns if c not in existing_meta]

print(f"Total columns: {len(df.columns)}")
print(f"Metadata columns: {len(existing_meta)}")
print(f"Base feature columns: {len(feature_cols)}")

# Extract base features and label
X_base = df[feature_cols].values
y = df['label'].values

print(f"\nBase features shape: {X_base.shape}")
print(f"Labels: {len(y)} (Malignant: {sum(y)}, Benign: {len(y)-sum(y)})")

# Apply polynomial expansion (degree-2)
print("\nApplying degree-2 polynomial expansion...")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_base)

print(f"Expanded features shape: {X_poly.shape}")

# Load top indices
with open(CHAMPION_PATH / "feature_metadata.json") as f:
    metadata = json.load(f)
    top_indices = metadata['top_indices']
    top_names = metadata['top_feature_names']

print(f"\nSelecting Top 100 features by indices...")
X_top100 = X_poly[:, top_indices]

print(f"Final Top 100 shape: {X_top100.shape}")

# Create dataframe with Top 100 + label
df_top100 = pd.DataFrame(X_top100, columns=top_names)
df_top100['label'] = y

# Save
output_path = OUTPUT_DIR / "champion_top100_features.csv"
df_top100.to_csv(output_path, index=False)

print(f"\n✅ Saved Top 100 features to: {output_path}")
print(f"   Rows: {len(df_top100)}")
print(f"   Columns: {len(df_top100.columns)} (100 features + label)")
print("\nThis dataset is ready for Phase 1 hyperparameter tuning!")
