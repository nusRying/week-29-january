
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import combinations
import time

# Performance Optimization: Use HSL/Colors for PhD presentation
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

STUDY_ROOT = Path(r"C:\Users\umair\Videos\PhD\PhD Data\Week 29 Jan\Code\GA_Mutation_Study")
SNAPSHOTS_DIR = STUDY_ROOT / "snapshots"
FIGURES_DIR = STUDY_ROOT / "figures"
METRICS_DIR = STUDY_ROOT / "analysis"
FIGURES_DIR.mkdir(exist_ok=True)
METRICS_DIR.mkdir(exist_ok=True)

# 120 features for Champion Derived set
TOTAL_FEATURES = 120 

def analyze_single_run(run_path):
    """
    Analyzes all snapshots for a single seed run.
    Structure: snapshots/{Part}/{Condition}/{Dataset}/seed_{Seed}/iter_{iteration}.csv
    """
    tag = "_".join(run_path.parts[-4:])
    print(f"Processing: {tag}")
    
    csv_snapshots = sorted(list(run_path.glob("iter_*.csv")))
    if not csv_snapshots:
        return None
    
    run_metrics = []
    seen_masks = set()
    seen_pairs = set()
    
    for snap in csv_snapshots:
        iteration = int(snap.stem.split("_")[1])
        df = pd.read_csv(snap)
        
        if df.empty:
            continue
            
        # 1. Structural Novelty Rate (SNR)
        # Masks are stored as strings of '0' and '1'
        current_masks = set(df['mask'].tolist())
        new_masks = current_masks - seen_masks
        snr = len(new_masks) / len(current_masks) if current_masks else 0
        seen_masks.update(current_masks)
        
        # 2. Condition Entropy (Information Content)
        # Using the saved 'specified_features' which is e.g. "1|5|10"
        feature_hit_counts = np.zeros(TOTAL_FEATURES)
        total_rules = len(df)
        
        for sf in df['specified_features'].dropna():
            indices = [int(i) for i in str(sf).split("|") if i]
            for idx in indices:
                if idx < TOTAL_FEATURES:
                    feature_hit_counts[idx] += 1
        
        p = feature_hit_counts / total_rules
        p = np.clip(p, 1e-10, 1-1e-10) # Avoid log(0)
        h_j = -p * np.log2(p) - (1-p) * np.log2(1-p)
        avg_entropy = np.mean(h_j)
        
        # 3. New Feature Interaction Discovery Rate (NFIDR - Pairs)
        current_pairs = set()
        for sf in df['specified_features'].dropna():
            indices = sorted([int(i) for i in str(sf).split("|") if i])
            if len(indices) >= 2:
                from itertools import combinations
                for pair in combinations(indices, 2):
                    current_pairs.add(pair)
        
        new_pairs = current_pairs - seen_pairs
        nfidr = len(new_pairs) / len(current_pairs) if current_pairs else 0
        seen_pairs.update(current_pairs)
        
        # 4. Generality
        mean_gen = df['generality'].mean()
        
        # 5. Performance (Accuracy)
        mean_acc = df['accuracy'].mean()
        
        run_metrics.append({
            'iteration': iteration,
            'snr': snr,
            'entropy': avg_entropy,
            'nfidr': nfidr,
            'generality': mean_gen,
            'accuracy': mean_acc,
            'pop_size': total_rules
        })
        
    if not run_metrics:
        print(f"⚠️ Warning: No metrics derived for {tag}")
        return None
        
    res_df = pd.DataFrame(run_metrics)
    output_name = f"{tag.replace(os.sep, '_')}.csv"
    res_df.to_csv(METRICS_DIR / output_name, index=False)
    return res_df

def run_full_analysis():
    print("🚀 Starting Scientific Analysis Pipeline...")
    start_time = time.time()
    
    all_summaries = []
    
    # Traverse Structure: Part / Condition / Dataset / Seed
    for part in ["PartB1", "PartB2", "PartC"]:
        part_dir = SNAPSHOTS_DIR / part
        if not part_dir.exists(): continue
        
        for condition in part_dir.iterdir():
            if not condition.is_dir(): continue
            
            for dataset in condition.iterdir():
                if not dataset.is_dir(): continue
                
                for seed_dir in dataset.iterdir():
                    if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"): continue
                    
                    df = analyze_single_run(seed_dir)
                    if df is not None:
                        # Extract metadata
                        summary = {
                            'part': part,
                            'condition': condition.name,
                            'dataset': dataset.name,
                            'seed': seed_dir.name,
                            'final_snr': df['snr'].iloc[-1],
                            'final_entropy': df['entropy'].iloc[-1],
                            'final_nfidr': df['nfidr'].iloc[-1],
                            'final_acc': df['accuracy'].iloc[-1],
                            'peak_snr': df['snr'].max(),
                            'path': str(seed_dir)
                        }
                        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(METRICS_DIR / "analysis_summary.csv", index=False)
    
    print(f"\n✅ Analysis complete in {time.time() - start_time:.1f}s")
    print(f"Metrics saved to {METRICS_DIR}")
    return summary_df

if __name__ == "__main__":
    run_full_analysis()
