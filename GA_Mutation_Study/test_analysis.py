import pandas as pd
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path("c:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code")
sys.path.append(str(PROJECT_ROOT / "GA_Mutation_Study" / "analysis"))

import compute_metrics

if __name__ == "__main__":
    print("Running analysis on test snapshots...")
    # Override the SNAPSHOTS_DIR for testing
    compute_metrics.SNAPSHOT_DIR = PROJECT_ROOT / "GA_Mutation_Study" / "test_snapshots"
    # Actually, compute_metrics iterates through child dirs. 
    # My test_snapshots doesn't have the condition/seed structure.
    # Let's fix that.
    
    test_root = PROJECT_ROOT / "GA_Mutation_Study" / "test_scientific_structure"
    seed_dir = test_root / "ga_on" / "seed_42"
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files from test_snapshots to scientific structure
    import shutil
    for f in (PROJECT_ROOT / "GA_Mutation_Study" / "test_snapshots").glob("*.csv"):
        shutil.copy(f, seed_dir / f.name)
        
    compute_metrics.SNAPSHOTS_DIR = test_root
    compute_metrics.compute_all_metrics()
    
    summary_file = PROJECT_ROOT / "GA_Mutation_Study" / "analysis" / "metrics_summary.csv"
    if summary_file.exists():
        print(f"\n[SUCCESS] Analysis complete. Summary:\n")
        print(pd.read_csv(summary_file))
    else:
        print("\n[ERROR] Analysis failed to produce summary.")
