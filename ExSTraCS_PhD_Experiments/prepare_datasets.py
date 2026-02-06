import pandas as pd
import numpy as np
from pathlib import Path

# Paths
CWD = Path("c:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code")
DATA_DIR = CWD / "ExSTraCS_PhD_Experiments" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ISIC_SOURCE = CWD / "Derived_Features_Champion" / "binary_features_all.csv"
HAM_FEATURES_SOURCE = CWD / "Derived_Features_Champion" / "HAM10000_extracted_features.csv"
HAM_METADATA_SOURCE = Path(r"C:\Users\umair\Videos\PhD\PhD Data\Week 15 January\Code V2\Dataset\clean\CleanData\HAM10000\HAM10000_metadata")

# Label Mapping
MALIGNANT_CLASSES = ['mel', 'bcc', 'akiec']
BENIGN_CLASSES = ['nv', 'bkl', 'df', 'vasc']

def prepare_isic():
    print("Preparing ISIC dataset...")
    df = pd.read_csv(ISIC_SOURCE)
    # Ensure standard binary labels (already present as 0/1)
    # Drop non-numeric metadata
    cols_to_drop = ['image_id', 'dx'] # Keep 'label'
    df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Save to data dir
    df_clean.to_csv(DATA_DIR / "isic_clean.csv", index=False)
    print(f"  Saved ISIC: {df_clean.shape}")
    return df_clean.columns

def prepare_ham(standard_columns):
    print("Preparing HAM dataset...")
    df_feats = pd.read_csv(HAM_FEATURES_SOURCE)
    df_meta = pd.read_csv(HAM_METADATA_SOURCE)
    
    # Merge on image_id
    df = pd.merge(df_feats, df_meta[['image_id', 'dx']], on='image_id')
    
    # Map dx to binary label
    def map_label(dx):
        if dx in MALIGNANT_CLASSES: return 1
        if dx in BENIGN_CLASSES: return 0
        return -1 # Should not happen
        
    df['label'] = df['dx'].apply(map_label)
    
    # Filter out any unmapped
    df = df[df['label'] != -1]
    
    # Align columns with ISIC
    # Drop image_id, dx
    df_clean = df.drop(columns=['image_id', 'dx'])
    
    # Ensure exact same columns as ISIC (ordering and presence)
    # Most features should be there, but let's be careful.
    final_cols = list(standard_columns)
    # If HAM is missing something, fill with 0? (Should not happen if extracted with parity)
    for col in final_cols:
        if col not in df_clean.columns:
            print(f"  Warning: HAM missing column {col}, filling with 0.")
            df_clean[col] = 0
            
    df_clean = df_clean[final_cols]
    
    df_clean.to_csv(DATA_DIR / "ham_clean.csv", index=False)
    print(f"  Saved HAM: {df_clean.shape}")

if __name__ == "__main__":
    isic_cols = prepare_isic()
    prepare_ham(isic_cols)
    print("\n[SUCCESS] Datasets synchronized and saved to ExSTraCS_PhD_Experiments/data/")
