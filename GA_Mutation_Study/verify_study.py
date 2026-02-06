import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path("c:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code")
sys.path.append(str(PROJECT_ROOT / "GA_Mutation_Study"))

from run_mutation_study import SnapshotExSTraCS

# Small test setup
DATASET_PATH = PROJECT_ROOT / "Derived_Features_Champion" / "binary_features_all.csv"
TEST_SNAP_DIR = PROJECT_ROOT / "GA_Mutation_Study" / "test_snapshots"
TEST_SNAP_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":
    print("Loading test data...")
    df = pd.read_csv(DATASET_PATH).head(100).fillna(0)
    cols_to_drop = ['image_id', 'dx', 'label']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).values
    y = df['label'].values
    
    # Run only 100 iterations
    model = SnapshotExSTraCS(
        study_tag="test_env",
        seed_val=42,
        export_dir=TEST_SNAP_DIR,
        learning_iterations=100,
        N=100,
        nu=1,
        mu=0.04,
        theta_GA=25,
        random_state=42
    )
    
    print("Testing fit loop with snapshots...")
    # Trigger snapshots at 0 and final
    global CHECKPOINTS
    import run_mutation_study
    run_mutation_study.CHECKPOINTS = [0, 50, 100]
    
    model.fit(X, y)
    
    print("\n[VERIFICATION] Checking for snapshot files...")
    for it in [0, 50, 100]:
        f = TEST_SNAP_DIR / f"iter_{it:06d}.csv"
        if f.exists():
            print(f"  ✓ Snapshot {it} found.")
            df_snap = pd.read_csv(f)
            if not df_snap.empty:
                print(f"    Rules: {len(df_snap)}, Mask Length: {len(str(df_snap['mask'].iloc[0]))}")
            else:
                print(f"    (Empty Population)")
        else:
            print(f"  ✗ Snapshot {it} MISSING.")
