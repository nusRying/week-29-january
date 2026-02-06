
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# PhD Style Guidelines
plt.style.use('seaborn-v0_8-muted')
plt.rcParams.update({
    'font.size': 12, 
    'figure.dpi': 300, 
    'savefig.dpi': 300,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10
})

STUDY_ROOT = Path(r"C:\Users\umair\Videos\PhD\PhD Data\Week 29 Jan\Code\GA_Mutation_Study")
METRICS_DIR = STUDY_ROOT / "analysis"
FIGURES_DIR = STUDY_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

def load_all_metrics():
    summary_file = METRICS_DIR / "analysis_summary.csv"
    if not summary_file.exists():
        return None
    
    summary_df = pd.read_csv(summary_file)
    if summary_df.empty:
        return None
        
    all_series = []
    for _, row in summary_df.iterrows():
        # Tag looks like 'PartB1_ga_off_ham_seed_42'
        tag = f"{row['part']}_{row['condition']}_{row['dataset']}_{row['seed']}"
        metric_file = METRICS_DIR / f"{tag}.csv"
        
        if metric_file.exists():
            df = pd.read_csv(metric_file)
            df['part'] = row['part']
            df['condition'] = row['condition']
            df['dataset'] = row['dataset']
            df['seed'] = row['seed']
            all_series.append(df)
            
    if not all_series:
        return None
        
    return pd.concat(all_series, ignore_index=True)

def plot_snr_evolution(data):
    """Line plots of Structural Novelty Rate for Part B.1"""
    print("Generating SNR Evolution Plots...")
    b1_data = data[data['part'] == 'PartB1']
    if b1_data.empty: return

    for ds in b1_data['dataset'].unique():
        plt.figure(figsize=(10, 6))
        ds_data = b1_data[b1_data['dataset'] == ds]
        
        sns.lineplot(data=ds_data, x='iteration', y='snr', hue='condition', marker='o')
        
        plt.title(f"Structural Novelty Rate (SNR) - {ds.upper()}")
        plt.xlabel("Learning Iterations")
        plt.ylabel("SNR (New Rule Masks / Pop Size)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"snr_evolution_{ds}.png")
        plt.close()

def plot_mutation_impact(data):
    """Bar plot of Accuracy vs Mutation Rate for Part B.2"""
    print("Generating Mutation Impact Plots...")
    b2_data = data[data['part'] == 'PartB2']
    if b2_data.empty: return

    # Extract mu from condition, e.g., 'mu_0.04'
    b2_data = b2_data.copy()
    b2_data['mu'] = b2_data['condition'].str.replace('mu_', '').astype(float)
    
    # Use final iteration only
    final_it = b2_data['iteration'].max()
    final_data = b2_data[b2_data['iteration'] == final_it]

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=final_data, x='mu', y='accuracy', hue='dataset', style='dataset', markers=True, dashes=False)
    
    plt.title("Impact of Mutation Rate (mu) on Evolution Stability")
    plt.xlabel("Mutation Rate (mu)")
    plt.ylabel("Mean Rule Accuracy")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mutation_impact_accuracy.png")
    plt.close()

def plot_interaction_heatmap(data):
    """Heatmap of p_spec vs mu for Part C"""
    print("Generating Interaction Heatmaps...")
    c_data = data[data['part'] == 'PartC']
    if c_data.empty: return

    # Extract p_spec and mu from condition, e.g., 'pspec0.3_mu0.04'
    c_data = c_data.copy()
    c_data['p_spec'] = c_data['condition'].str.extract(r'pspec([\d.]+)').astype(float)
    c_data['mu'] = c_data['condition'].str.extract(r'mu([\d.]+)').astype(float)
    
    final_it = c_data['iteration'].max()
    final_data = c_data[c_data['iteration'] == final_it]

    for ds in final_data['dataset'].unique():
        ds_data = final_data[final_data['dataset'] == ds]
        pivot = ds_data.groupby(['p_spec', 'mu'])['nfidr'].mean().unstack()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt=".3f")
        plt.title(f"NFIDR: Covering (p_spec) x Mutation (mu) - {ds.upper()}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"interaction_heatmap_{ds}.png")
        plt.close()

def run_viz_pipeline():
    print("🚀 Starting PhD Visualization Pipeline...")
    data = load_all_metrics()
    if data is None:
        print("❌ Error: No metric data found. Run scientific_analysis.py first.")
        return

    plot_snr_evolution(data)
    plot_mutation_impact(data)
    plot_interaction_heatmap(data)
    
    print(f"✅ Visualization complete. All figures saved to {FIGURES_DIR}")

if __name__ == "__main__":
    run_viz_pipeline()
