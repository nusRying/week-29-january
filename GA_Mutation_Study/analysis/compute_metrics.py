import pandas as pd
import numpy as np
import os
from pathlib import Path
import math

# Dynamic path resolution to prevent CWD issues
SCRIPT_PATH = Path(__file__).resolve()
STUDY_ROOT = SCRIPT_PATH.parents[1]
SNAPSHOTS_DIR = STUDY_ROOT / "snapshots"
OUTPUT_DIR = STUDY_ROOT / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

def compute_all_metrics():
    summary_data = []
    if not SNAPSHOTS_DIR.exists(): return

    # Traverse SNAPSHOTS_DIR/Part/Condition/Dataset/Seed
    for part_dir in SNAPSHOTS_DIR.iterdir():
        if not part_dir.is_dir(): continue
        for cond_dir in part_dir.iterdir():
            if not cond_dir.is_dir(): continue
            for ds_dir in cond_dir.iterdir():
                if not ds_dir.is_dir(): continue
                for seed_dir in ds_dir.iterdir():
                    if not seed_dir.is_dir(): continue
                    
                    print(f"Processing {part_dir.name}/{cond_dir.name}/{ds_dir.name}/{seed_dir.name}...")
                    snapshots = sorted(list(seed_dir.glob("iter_*.csv")))
                    
                    global_seen_masks = set()
                    global_seen_pairs = set()
                    
                    for snap_path in snapshots:
                        it = int(snap_path.stem.split("_")[1])
                        df = pd.read_csv(snap_path)
                        if df.empty: continue
                        
                        # Metrics computation (SNR, NFIDR, Entropy, Generality)
                        current_masks = set(df['mask'].astype(str))
                        new_masks = current_masks - global_seen_masks
                        snr = len(new_masks) / len(current_masks) if current_masks else 0
                        global_seen_masks.update(current_masks)
                        
                        # NFIDR (Pairs)
                        current_pairs = set()
                        for spec_str in df['specified_features'].dropna():
                            feats = [int(i) for i in str(spec_str).split("|")]
                            if len(feats) >= 2:
                                from itertools import combinations
                                for pair in combinations(feats, 2): current_pairs.add(pair)
                        
                        new_pairs = current_pairs - global_seen_pairs
                        nfidr = len(new_pairs) / len(current_pairs) if current_pairs else 0
                        global_seen_pairs.update(current_pairs)
                        
                        summary_data.append({
                            'part': part_dir.name,
                            'condition': cond_dir.name,
                            'dataset': ds_dir.name,
                            'seed': seed_dir.name,
                            'iteration': it,
                            'snr': snr,
                            'nfidr': nfidr,
                            'gen_var': df['generality'].var(),
                            'pop_size': len(df)
                        })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
        print(f"Summary saved to {OUTPUT_DIR / 'metrics_summary.csv'}")
    else:
        print("No data collected.")

if __name__ == "__main__":
    compute_all_metrics()
