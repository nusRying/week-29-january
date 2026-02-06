import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# -----------------------------------------------------------------------------
# 1. SETUP & PATHS
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs"
PLOTS_DIR = SCRIPT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

metrics_to_plot = [
    ('balanced_accuracy', 'Test Balanced Accuracy'),
    ('f1_score', 'Test F1-Score'),
    ('precision', 'Test Precision'),
    ('recall', 'Test Recall'),
    ('rule_efficiency', 'Rule Efficiency (Perf/Rules)'),
    ('final_rule_count', 'Active Rule Population')
]

def load_results():
    data = []
    json_files = list(RUNS_DIR.glob("*.json"))
    
    if not json_files:
        print("No results found in runs/ directory.")
        return pd.DataFrame() # Empty
        
    print(f"Loading {len(json_files)} result files...")
    
    for f in json_files:
        with open(f, 'r') as file:
            run = json.load(file)
            
            # Flatten structure for DataFrame
            entry = {
                'multiplier': run['multiplier'],
                'N': run['N'],
                'seed': run['seed'],
                'duration': run['duration_total'],
                'final_rule_count': run['final_rule_count']
            }
            
            # Add metrics
            for k, v in run['test_metrics'].items():
                entry[k] = v
                
            data.append(entry)
            
    return pd.DataFrame(data)

def perform_statistical_analysis(df):
    print("\n--- 📊 Statistical Analysis (One-Way ANOVA) ---")
    
    # We test Balanced Accuracy across Multiplier groups
    groups = []
    multipliers = sorted(df['multiplier'].unique())
    
    for m in multipliers:
        group_data = df[df['multiplier'] == m]['balanced_accuracy'].values
        groups.append(group_data)
        
    # Check assumptions (Normality/Homogeneity) roughly or just proceed to ANOVA
    # One-way ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    print(f"Metric: Test Balanced Accuracy")
    print(f"F-Statistic: {f_stat:.4f}")
    print(f"P-Value: {p_value:.4e}")
    
    alpha = 0.05
    if p_value < alpha:
        print("✅ RESULT: Statistically Significant Difference detected between population sizes.")
        print("   (Reject Null Hypothesis H0: All means are equal)")
    else:
        print("❌ RESULT: No statistically significant difference found.")
        print("   (Fail to reject H0)")
        
    return f_stat, p_value

def generate_plots(df):
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Performance Curve (Line Plot with Std Dev)
    print("\nGenerating Performance Curves...")
    for metric, label in metrics_to_plot:
        if metric not in df.columns: continue
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df, 
            x='multiplier', 
            y=metric, 
            marker='o',
            err_style='bars', # Show error bars (std/ci)
            errorbar='sd'     # Standard Deviation
        )
        plt.title(f'Impact of Population Multiplier on {label}')
        plt.xlabel('Population Multiplier (x Dataset Size)')
        plt.ylabel(label)
        plt.xscale('log') # Log scale often helps if multipliers are 0.1, 1, 10 etc.
        plt.xticks(sorted(df['multiplier'].unique()), sorted(df['multiplier'].unique())) # Force ticks
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"curve_{metric}.png")
        plt.close()

    # 2. Efficiency Plot (Scatter/Line Overlay)
    print("Generating Efficiency Analysis...")
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    avg_df = df.groupby('multiplier').mean(numeric_only=True).reset_index()
    
    color = 'tab:blue'
    ax1.set_xlabel('Population Multiplier')
    ax1.set_ylabel('Balanced Accuracy', color=color)
    ax1.plot(avg_df['multiplier'], avg_df['balanced_accuracy'], color=color, marker='o', label='Bal Acc')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    ax1.set_xticks(sorted(df['multiplier'].unique()), sorted(df['multiplier'].unique()))

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Active Rule Count (Complexity)', color=color)  # we already handled the x-label with ax1
    ax2.plot(avg_df['multiplier'], avg_df['final_rule_count'], color=color, linestyle='--', marker='x', label='Rule Count')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Trade-off: Performance vs Model Complexity')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(PLOTS_DIR / f"tradeoff_efficiency.png")
    plt.close()

    # 3. Boxplots (Distribution)
    print("Generating Boxplots...")
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='multiplier', y='f1_score', data=df, hue='multiplier', palette="viridis", legend=False)
    plt.title('Distribution of F1-Scores across 10 Seeds per Multiplier')
    plt.ylabel('Test F1-Score')
    plt.xlabel('Population Multiplier')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"boxplot_f1_distribution.png")
    plt.close()

def main():
    df = load_results()
    if df.empty:
        return

    # 1. Generate Summary Table (Mean +/- Std)
    summary = df.groupby('multiplier')[['balanced_accuracy', 'f1_score', 'rule_efficiency', 'final_rule_count']].agg(['mean', 'std'])
    print("\n--- 📋 Statistical Summary (Mean ± Std) ---")
    print(summary.to_markdown())
    
    summary.to_csv(SCRIPT_DIR / "summary_stats.csv")
    
    # 2. Statistical Test
    perform_statistical_analysis(df)
    
    # 3. Visualization
    generate_plots(df)
    
    print("\nAnalysis Complete. Outputs saved to plots/ and summary_stats.csv")

if __name__ == "__main__":
    main()
