"""
Data Splitter for Reproducible Train/Val Splits
Ensures all experiments use identical data partitions for fair comparison.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

SCRIPT_PATH = Path(__file__).resolve()
STUDY_ROOT = SCRIPT_PATH.parent
DATA_DIR = STUDY_ROOT / "data"

RANDOM_SEED = 42
TEST_SIZE = 0.2  # 80/20 split

def create_splits(dataset_name):
    """Create and save train/val indices for a dataset."""
    data_path = DATA_DIR / f"{dataset_name}_clean.csv"
    df = pd.read_csv(data_path)
    
    # Stratified split to maintain class balance
    indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED,
        stratify=df['label']
    )
    
    # Save indices
    np.save(DATA_DIR / f"{dataset_name}_train_indices.npy", train_idx)
    np.save(DATA_DIR / f"{dataset_name}_val_indices.npy", val_idx)
    
    print(f"{dataset_name.upper()}: Train={len(train_idx)}, Val={len(val_idx)}")
    print(f"  Train class dist: {df.iloc[train_idx]['label'].value_counts().to_dict()}")
    print(f"  Val class dist: {df.iloc[val_idx]['label'].value_counts().to_dict()}")

if __name__ == "__main__":
    print("Creating reproducible train/val splits...\n")
    create_splits("ham")
    create_splits("isic")
    print("\nSplits saved to data/ directory.")
