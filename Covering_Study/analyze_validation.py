"""
Alternative Covering Analysis: Validation Accuracy Approach
Since covering event logging isn't accessible in ExSTraCS, we analyze the 
validation accuracy trends across p_spec values to infer covering behavior.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

SCRIPT_PATH = Path(__file__).resolve()
STUDY_ROOT = SCRIPT_PATH.parent
RUNS_DIR = STUDY_ROOT / "runs"
RESULTS_DIR = STUDY_ROOT / "results"
FIGURES_DIR = STUDY_ROOT / "figures"

sns.set_style("whitegrid")
sns.set_palette("husl")

def collect_validation_accuracies():
    """Collect validation BA from all report.json files."""
    data = []
    
    for dataset_dir in RUNS_DIR.iterdir():
        if not dataset_dir.is_dir(): continue
        dataset = dataset_dir.name
        
        for pspec_dir in dataset_dir.iterdir():
            if not pspec_dir.is_dir(): continue
            # Extract p_spec from folder name like "pspec_0.5"
            pspec_str = pspec_dir.name.replace("pspec_", "")
            pspec = float(pspec_str)
            
            for seed_dir in pspec_dir.iterdir():
                if not seed_dir.is_dir(): continue
                seed_str = seed_dir.name.replace("seed_", "")
                
                report_file = seed_dir / "report.json"
                if report_file.exists():
                    with open(report_file, 'r') as f:
                        report = json.load(f)
                        data.append({
                            'dataset': dataset,
                            'p_spec': pspec,
                            'seed': int(seed_str),
                            'val_ba': report['val_balanced_accuracy'],
                            'duration': report['duration'],
                            'rsl': report['rsl']
                        })
    
    return pd.DataFrame(data)

def plot_pspec_vs_accuracy():
    """Plot p_spec vs validation accuracy with error bars."""
    df = collect_validation_accuracies()
    
    if df.empty:
        print("No validation data found!")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, dataset in enumerate(['ham', 'isic']):
        ax = axes[idx]
        subset = df[df['dataset'] == dataset]
        
        # Group by p_spec and compute mean/std
        grouped = subset.groupby('p_spec')['val_ba'].agg(['mean', 'std', 'count']).reset_index()
        
        # Plot with error bars
        ax.errorbar(grouped['p_spec'], grouped['mean'], yerr=grouped['std'], 
                   marker='o', markersize=8, linewidth=2, capsize=5, capthick=2,
                   label=f'{dataset.upper()} (n={int(grouped["count"].iloc[0])} seeds)')
        
        # Add trend line
        z = np.polyfit(grouped['p_spec'], grouped['mean'], 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(0.1, 0.9, 100)
        ax.plot(x_smooth, p(x_smooth), '--', alpha=0.5, label='Quadratic fit')
        
        ax.set_xlabel('p_spec (Specification Probability)', fontsize=12)
        ax.set_ylabel('Validation Balanced Accuracy', fontsize=12)
        ax.set_title(f'{dataset.upper()} Dataset', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.48, 0.62])
    
    plt.suptitle('Effect of Covering p_spec on Generalization', fontsize=16)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'pspec_vs_validation_accuracy.png', dpi=150)
    plt.close()
    print("Saved: pspec_vs_validation_accuracy.png")

def plot_rsl_actual_values():
    """Plot actual RSL values used vs p_spec."""
    df = collect_validation_accuracies()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for dataset in ['ham', 'isic']:
        subset = df[df['dataset'] == dataset]
        grouped = subset.groupby('p_spec')['rsl'].first().reset_index()
        ax.plot(grouped['p_spec'], grouped['rsl'], marker='s', linewidth=2, 
               markersize=8, label=dataset.upper())
    
    # Add theoretical line
    d = 226  # Number of features
    pspec_range = np.linspace(0.1, 0.9, 100)
    rsl_theoretical = np.minimum(d, 2 * pspec_range * d - 1)
    ax.plot(pspec_range, rsl_theoretical, '--', color='gray', alpha=0.7, 
           label='Theoretical RSL = min(d, 2*p_spec*d - 1)')
    
    ax.axhline(d, color='red', linestyle=':', alpha=0.5, label=f'Max Features (d={d})')
    ax.set_xlabel('p_spec (Target Specification Probability)', fontsize=12)
    ax.set_ylabel('Actual RSL Used', fontsize=12)
    ax.set_title('Rule Specificity Limit Mapping', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'rsl_mapping.png', dpi=150)
    plt.close()
    print("Saved: rsl_mapping.png")

def generate_accuracy_summary_table():
    """Generate summary statistics table."""
    df = collect_validation_accuracies()
    
    summary = df.groupby(['dataset', 'p_spec'])['val_ba'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('n_seeds', 'count')
    ]).reset_index()
    
    summary.to_csv(RESULTS_DIR / 'validation_accuracy_summary.csv', index=False)
    print("\nValidation Accuracy Summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved to: results/validation_accuracy_summary.csv")
    
    # Key insights
    print("\n=== Key Insights ===")
    for dataset in ['ham', 'isic']:
        subset = summary[summary['dataset'] == dataset]
        best_pspec = subset.loc[subset['mean'].idxmax(), 'p_spec']
        best_ba = subset['mean'].max()
        print(f"{dataset.upper()}: Best p_spec = {best_pspec} (Val BA = {best_ba:.4f})")

if __name__ == "__main__":
    print("=== Alternative Covering Analysis (Validation Accuracy Approach) ===\n")
    plot_pspec_vs_accuracy()
    plot_rsl_actual_values()
    generate_accuracy_summary_table()
    print("\n=== Done ===")
