"""
Enhanced Covering Analysis: Publication-Quality Statistical Analysis
Includes ANOVA, effect sizes, confidence intervals, and thesis-ready visualizations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
from scipy.stats import f_oneway

SCRIPT_PATH = Path(__file__).resolve()
STUDY_ROOT = SCRIPT_PATH.parent
RUNS_DIR = STUDY_ROOT / "runs"
RESULTS_DIR = STUDY_ROOT / "results"
FIGURES_DIR = STUDY_ROOT / "figures"

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
COLORS = sns.color_palette("husl", 5)

def collect_validation_accuracies():
    """Collect validation BA from all report.json files."""
    data = []
    
    for dataset_dir in RUNS_DIR.iterdir():
        if not dataset_dir.is_dir(): continue
        dataset = dataset_dir.name
        
        for pspec_dir in dataset_dir.iterdir():
            if not pspec_dir.is_dir(): continue
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

def compute_cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def perform_statistical_tests(df):
    """Perform ANOVA and post-hoc tests."""
    results = []
    
    for dataset in ['ham', 'isic']:
        subset = df[df['dataset'] == dataset]
        
        # Group by p_spec
        groups = [subset[subset['p_spec'] == pspec]['val_ba'].values 
                  for pspec in [0.1, 0.3, 0.5, 0.7, 0.9]]
        
        # One-way ANOVA
        f_stat, p_value = f_oneway(*groups)
        
        results.append({
            'dataset': dataset.upper(),
            'test': 'One-Way ANOVA',
            'F_statistic': f_stat,
            'p_value': p_value,
            'significant': 'Yes' if p_value < 0.05 else 'No'
        })
        
        # Effect sizes (Cohen's d for best vs others)
        best_pspec = 0.5
        best_vals = subset[subset['p_spec'] == best_pspec]['val_ba'].values
        for other_pspec in [0.1, 0.3, 0.7, 0.9]:
            other_vals = subset[subset['p_spec'] == other_pspec]['val_ba'].values
            if len(other_vals) > 0:
                cohens_d = compute_cohens_d(best_vals, other_vals)
                magnitude = 'negligible' if abs(cohens_d) < 0.2 else \
                           'small' if abs(cohens_d) < 0.5 else \
                           'medium' if abs(cohens_d) < 0.8 else 'large'
                results.append({
                    'dataset': dataset.upper(),
                    'test': f'Cohens_d_p05_vs_p{other_pspec}',
                    'effect_size': cohens_d,
                    'magnitude': magnitude
                })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / 'statistical_tests.csv', index=False)
    return results_df

def plot_box_plots_with_ci(df):
    """Box plots with 95% CI markers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, dataset in enumerate(['ham', 'isic']):
        ax = axes[idx]
        subset = df[df['dataset'] == dataset]
        
        # Box plot
        positions = [0.1, 0.3, 0.5, 0.7, 0.9]
        box_data = [subset[subset['p_spec'] == p]['val_ba'].values for p in positions]
        
        bp = ax.boxplot(box_data, positions=positions, widths=0.08,
                       patch_artist=True, showfliers=True,
                       boxprops=dict(facecolor=COLORS[idx], alpha=0.6),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
        
        # Add mean markers with 95% CI
        for i, p in enumerate(positions):
            vals = box_data[i]
            mean = np.mean(vals)
            ci = 1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals))
            ax.errorbar(p, mean, yerr=ci, fmt='D', color='black', 
                       markersize=6, capsize=4, capthick=2, label='Mean ± 95% CI' if i == 0 else "")
        
        ax.set_xlabel('p_spec (Specification Probability)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Validation Balanced Accuracy', fontsize=13, fontweight='bold')
        ax.set_title(f'{dataset.upper()} Dataset', fontsize=15, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(positions)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0.48, 0.62])
    
    plt.suptitle('Validation Accuracy Distributions with Statistical Markers', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'boxplots_with_ci.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: boxplots_with_ci.png")

def plot_heatmap_performance_matrix(df):
    """Heatmap: p_spec × dataset performance."""
    pivot = df.groupby(['dataset', 'p_spec'])['val_ba'].mean().reset_index()
    matrix = pivot.pivot(index='p_spec', columns='dataset', values='val_ba')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='.4f', cmap='RdYlGn', center=0.52,
                linewidths=0.5, cbar_kws={'label': 'Validation BA'},
                ax=ax, vmin=0.50, vmax=0.54)
    
    ax.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax.set_ylabel('p_spec', fontsize=13, fontweight='bold')
    ax.set_title('Performance Matrix: Mean Validation Accuracy', 
                 fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: performance_heatmap.png")

def generate_publication_summary(df, stats_df):
    """Generate publication-ready summary table."""
    summary = df.groupby(['dataset', 'p_spec'])['val_ba'].agg([
        ('Mean', lambda x: f"{x.mean():.4f}"),
        ('SD', lambda x: f"{x.std(ddof=1):.4f}"),
        ('95pct_CI', lambda x: f"±{1.96*x.std(ddof=1)/np.sqrt(len(x)):.4f}"),
        ('n', 'count')
    ]).reset_index()
    
    summary.to_csv(RESULTS_DIR / 'publication_summary.csv', index=False)
    print("\n" + "="*60)
    print("PUBLICATION-READY SUMMARY TABLE")
    print("="*60)
    print(summary.to_string(index=False))
    print("\nSaved to: results/publication_summary.csv")
    
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*60)
    print(stats_df.to_string(index=False))
    print("\nSaved to: results/statistical_tests.csv")

if __name__ == "__main__":
    print("=== Enhanced Covering Analysis (Publication Quality) ===\n")
    
    df = collect_validation_accuracies()
    
    if df.empty:
        print("❌ No data found!")
    else:
        print(f"📊 Loaded {len(df)} experiment results\n")
        
        # Statistical tests
        print("[1/3] Performing statistical tests...")
        stats_df = perform_statistical_tests(df)
        
        # Enhanced visualizations
        print("[2/3] Generating enhanced visualizations...")
        plot_box_plots_with_ci(df)
        plot_heatmap_performance_matrix(df)
        
        # Publication summary
        print("[3/3] Creating publication summary...")
        generate_publication_summary(df, stats_df)
        
        print("\n=== Done ===")
