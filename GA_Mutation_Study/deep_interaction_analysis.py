
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# PhD Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

STUDY_ROOT = Path(r"C:\Users\umair\Videos\PhD\PhD Data\Week 29 Jan\Code\GA_Mutation_Study")
SUMMARY_FILE = STUDY_ROOT / "analysis" / "analysis_summary.csv"
FIGURES_DIR = STUDY_ROOT / "figures"

def analyze_rescue_effect():
    if not SUMMARY_FILE.exists():
        print(f"❌ Error: {SUMMARY_FILE} not found.")
        return

    df = pd.read_csv(SUMMARY_FILE)
    c_data = df[df['part'] == 'PartC'].copy()
    if c_data.empty:
        print("❌ Error: No Part C data found.")
        return

    # Extract p_spec and mu
    c_data['p_spec'] = c_data['condition'].str.extract(r'pspec([\d.]+)').astype(float)
    c_data['mu'] = c_data['condition'].str.extract(r'mu([\d.]+)').astype(float)

    # Hypothesis: Does Mutation rescue poor (high) p_spec?
    # Compare mu=0.12 (High) vs mu=0.0 (Baseline)
    comparison = c_data[c_data['mu'].isin([0.0, 0.12])]
    
    # Calculate Mean Accuracy per (p_spec, mu, dataset)
    pivot = comparison.groupby(['dataset', 'p_spec', 'mu'])['final_acc'].mean().unstack()
    pivot['delta_acc'] = pivot[0.12] - pivot[0.0]
    pivot['perc_improvement'] = (pivot['delta_acc'] / pivot[0.0]) * 100

    print("--- The Rescue Effect Analysis ---")
    print(pivot)

    # Plotting the Rescue Magnitude
    plt.figure(figsize=(10, 6))
    pivot_reset = pivot.reset_index()
    sns.barplot(data=pivot_reset, x='p_spec', y='perc_improvement', hue='dataset', palette='viridis')
    
    plt.axhline(0, color='black', linestyle='-', linewidth=1)
    plt.title("The 'Rescue Effect': Mutation's Relative Impact by Initial Specificity")
    plt.xlabel("Initial Specification Probability (p_spec)")
    plt.ylabel("% Improvement in Rule Accuracy (mu=0.12 vs 0.0)")
    plt.legend(title="Dataset")
    
    # Add annotations
    for i, p in enumerate(plt.gca().patches):
        if p.get_height() != 0:
            plt.gca().annotate(f"{p.get_height():.2f}%", 
                               (p.get_x() + p.get_width() / 2., p.get_height()), 
                               ha='center', va='center', xytext=(0, 9), 
                               textcoords='offset points')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rescue_effect_interaction.png")
    print(f"✅ Rescue Effect plot saved to {FIGURES_DIR / 'rescue_effect_interaction.png'}")

if __name__ == "__main__":
    analyze_rescue_effect()
