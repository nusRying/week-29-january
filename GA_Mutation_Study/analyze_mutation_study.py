import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

ROOT_DIR = Path("GA_Mutation_Study")
RUNS_DIR = ROOT_DIR / "2_runs"
OUTPUT_METRICS_DIR = ROOT_DIR / "4_metrics"
OUTPUT_METRICS_DIR.mkdir(exist_ok=True)

def calculate_metrics(run_name):
    print(f"Analyzing run: {run_name}")
    run_path = RUNS_DIR / run_name
    checkpoints = sorted([d for d in run_path.iterdir() if d.is_dir() and d.name.startswith("iteration_")], 
                        key=lambda x: int(x.name.split("_")[1]))
    
    all_metrics = []
    global_seen_masks = set()
    global_seen_pairs = set()
    
    # Get num_features from first snapshot
    with open(checkpoints[0] / "population.json", "r") as f:
        sample_pop = json.load(f)
    # This is tricky if we don't have the feature count. 
    # Let's assume the high count from your previous work (~167 base + derived overhead).
    # We can infer it from the highest index seen? No, better use a fixed value.
    # In your case, it was 25651 for derived? Let's assume the total interaction space.
    # For now, let's use the max index + 1 from snapshots.
    max_idx = 0
    for r in sample_pop:
        if r['specified_features']:
            max_idx = max(max_idx, max(r['specified_features']))
    num_features = max_idx + 1 # Approximate

    for cp in checkpoints:
        it = int(cp.name.split("_")[1])
        with open(cp / "population.json", "r") as f:
            pop = json.load(f)
        
        if not pop: continue
        
        # 1. SNR (Structural Novelty Rate)
        current_masks = set()
        for r in pop:
            current_masks.add(tuple(sorted(r['specified_features'])))
        
        new_masks = current_masks - global_seen_masks
        snr = len(new_masks) / len(current_masks) if current_masks else 0
        global_seen_masks.update(current_masks)
        
        # 2. Condition Entropy
        feature_counts = np.zeros(num_features)
        total_rules = len(pop)
        for r in pop:
            for feat in r['specified_features']:
                if feat < num_features:
                    feature_counts[feat] += 1
        
        p = feature_counts / total_rules
        # Handle 0/1 for log
        p = np.clip(p, 1e-10, 1-1e-10)
        h_j = -p * np.log2(p) - (1-p) * np.log2(1-p)
        avg_h = np.mean(h_j)
        
        # 3. Generality
        generalities = [(num_features - len(r['specified_features'])) / num_features for r in pop]
        mean_g = np.mean(generalities)
        var_g = np.var(generalities)
        
        # 4. NFIDR (New Feature Interaction Discovery Rate) - Pairs
        current_pairs = set()
        for r in pop:
            feats = sorted(r['specified_features'])
            if len(feats) >= 2:
                from itertools import combinations
                for pair in combinations(feats, 2):
                    current_pairs.add(pair)
        
        new_pairs = current_pairs - global_seen_pairs
        nfidr = len(new_pairs) / len(current_pairs) if current_pairs else 0
        global_seen_pairs.update(current_pairs)
        
        all_metrics.append({
            'iteration': it,
            'snr': snr,
            'avg_entropy': avg_h,
            'mean_generality': mean_g,
            'var_generality': var_g,
            'nfidr': nfidr,
            'pop_size': total_rules
        })
    
    df = pd.DataFrame(all_metrics)
    df.to_csv(OUTPUT_METRICS_DIR / f"metrics_{run_name}.csv", index=False)
    return df

def plot_comparison():
    metric_files = list(OUTPUT_METRICS_DIR.glob("metrics_*.csv"))
    if not metric_files: return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metrics_to_plot = ['snr', 'avg_entropy', 'var_generality', 'nfidr']
    titles = ['Structural Novelty Rate', 'Condition Entropy', 'Generality Variance', 'Interaction Discovery (Pairs)']
    
    for i, m in enumerate(metrics_to_plot):
        ax = axes[i//2, i%2]
        for f in metric_files:
            df = pd.read_csv(f)
            label = f.stem.replace("metrics_", "").replace("_seed_42", "")
            ax.plot(df['iteration'], df[m], marker='o', label=label)
        ax.set_title(titles[i])
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(ROOT_DIR / "6_figures" / "mutation_impact_summary.png")
    print(f"Summary plot saved to {ROOT_DIR / '6_figures' / 'mutation_impact_summary.png'}")

if __name__ == "__main__":
    runs = [d.name for d in RUNS_DIR.iterdir() if d.is_dir()]
    for r in runs:
        calculate_metrics(r)
    plot_comparison()
