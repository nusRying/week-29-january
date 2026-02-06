import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add script dir to path
sys.path.append(str(Path.cwd() / "GA_Mutation_Study"))
from run_mutation_study import MonitoredExSTraCS

# Short test params
CHECKPOINTS = [0, 500, 1000]
DATASET_PATH = Path("Derived_Features_Champion") / "binary_features_all.csv"
RESULT_DIR = Path("GA_Mutation_Study") / "test_run"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print("Loading small sample...")
    df = pd.read_csv(DATASET_PATH).head(100)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    model = MonitoredExSTraCS(
        study_name="test_run",
        seed=42,
        checkpoints=CHECKPOINTS,
        result_dir=RESULT_DIR,
        learning_iterations=1000,
        N=500,
        nu=1,
        random_state=42,
        mu=0.04,
        chi=0.8
    )
    
    print("Running short fit...")
    model.fit(X, y)
    print("DONE. Checking snapshots...")
    
    for it in CHECKPOINTS:
        p = RESULT_DIR / f"iteration_{it}" / "population.json"
        if p.exists():
            print(f"Snapshot {it} exists.")
        else:
            print(f"ERROR: Snapshot {it} MISSING.")
