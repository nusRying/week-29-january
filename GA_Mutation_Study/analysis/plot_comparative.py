import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
STUDY_ROOT = SCRIPT_PATH.parents[1]
RESULTS_DIR = STUDY_ROOT / "results"
FIGURES_DIR = STUDY_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")
sns.set_palette("tab10")

def plot_snr_comparison():
    """Plot SNR curves across conditions."""
    metrics_file = RESULTS_DIR / "metrics_summary.csv"
    if not metrics_file.exists():
        print("No metrics file found. Run compute_metrics.py first.")
        return
    
    df = pd.read_csv(metrics_file)
    
    for part in df['part'].unique():
        for dataset in df['dataset'].unique():
            subset = df[(df['part'] == part) & (df['dataset'] == dataset)]
            if subset.empty: continue
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for condition in subset['condition'].unique():
                cond_data = subset[subset['condition'] == condition]
                # Aggregate across seeds
                grouped = cond_data.groupby('iteration')['snr'].agg(['mean', 'std']).reset_index()
                
                ax.plot(grouped['iteration'], grouped['mean'], label=condition, linewidth=2, marker='o')
                ax.fill_between(grouped['iteration'], 
                               grouped['mean'] - grouped['std'], 
                               grouped['mean'] + grouped['std'], 
                               alpha=0.2)
            
            ax.set_title(f'Structural Novelty Rate - {part} - {dataset.upper()}', fontsize=14)
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('SNR', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f'snr_{part}_{dataset}.png', dpi=150)
            plt.close()
            print(f"Saved: snr_{part}_{dataset}.png")

def plot_nfidr_comparison():
    """Plot NFIDR (interaction discovery) curves."""
    metrics_file = RESULTS_DIR / "metrics_summary.csv"
    if not metrics_file.exists(): return
    
    df = pd.read_csv(metrics_file)
    
    for part in df['part'].unique():
        for dataset in df['dataset'].unique():
            subset = df[(df['part'] == part) & (df['dataset'] == dataset)]
            if subset.empty: continue
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for condition in subset['condition'].unique():
                cond_data = subset[subset['condition'] == condition]
                grouped = cond_data.groupby('iteration')['nfidr'].agg(['mean', 'std']).reset_index()
                
                ax.plot(grouped['iteration'], grouped['mean'], label=condition, linewidth=2, marker='s')
                ax.fill_between(grouped['iteration'], 
                               grouped['mean'] - grouped['std'], 
                               grouped['mean'] + grouped['std'], 
                               alpha=0.2)
            
            ax.set_title(f'Feature Interaction Discovery Rate - {part} - {dataset.upper()}', fontsize=14)
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('NFIDR', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f'nfidr_{part}_{dataset}.png', dpi=150)
            plt.close()
            print(f"Saved: nfidr_{part}_{dataset}.png")

def plot_generality_variance():
    """Plot generality variance over time."""
    metrics_file = RESULTS_DIR / "metrics_summary.csv"
    if not metrics_file.exists(): return
    
    df = pd.read_csv(metrics_file)
    
    for part in df['part'].unique():
        for dataset in df['dataset'].unique():
            subset = df[(df['part'] == part) & (df['dataset'] == dataset)]
            if subset.empty: continue
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for condition in subset['condition'].unique():
                cond_data = subset[subset['condition'] == condition]
                grouped = cond_data.groupby('iteration')['gen_var'].agg(['mean', 'std']).reset_index()
                
                ax.plot(grouped['iteration'], grouped['mean'], label=condition, linewidth=2, marker='^')
            
            ax.set_title(f'Generality Variance - {part} - {dataset.upper()}', fontsize=14)
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Variance of Generality', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f'gen_var_{part}_{dataset}.png', dpi=150)
            plt.close()
            print(f"Saved: gen_var_{part}_{dataset}.png")

if __name__ == "__main__":
    print("=== Generating GA Mutation Study Visualizations ===\n")
    plot_snr_comparison()
    plot_nfidr_comparison()
    plot_generality_variance()
    print("\n=== Done ===")
