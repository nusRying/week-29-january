import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np

# Paths
STUDY_ROOT = Path(__file__).resolve().parent
RUNS_DIR = STUDY_ROOT / "runs"
PLOTS_DIR = STUDY_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

def aggregate_results():
    results = []
    for results_file in RUNS_DIR.glob("**/results.json"):
        # Path structure: runs/Part_Name/Condition/Dataset/seed_X/results.json
        parts = results_file.parts
        # Relative to RUNS_DIR: Part_Name, Condition, Dataset, seed_X, results.json
        rel_parts = results_file.relative_to(RUNS_DIR).parts
        
        with open(results_file, 'r') as f:
            data = json.load(f)
            data['part'] = rel_parts[0]
            data['condition'] = rel_parts[1]
            data['dataset'] = rel_parts[2]
            # Extract mu from condition if possible (e.g. mu_0.04) or manual mapping
            results.append(data)
    return pd.DataFrame(results)

def plot_mutation_performance(df):
    if df.empty: return
    plt.figure(figsize=(12, 6))
    # Filter for B.1 Factorial
    b1_df = df[df['part'] == 'Part_B1']
    if not b1_df.empty:
        sns.boxplot(x='condition', y='final_acc', hue='dataset', data=b1_df)
        plt.title("Mutation Impact on Final Population Accuracy (Factorial Study)")
        plt.xlabel("Mutation Condition")
        plt.ylabel("System Accuracy")
        plt.grid(alpha=0.3)
        plt.savefig(PLOTS_DIR / "mutation_b1_performance.png")
    plt.close()

if __name__ == "__main__":
    print("Aggregating mutation study results...")
    df = aggregate_results()
    if not df.empty:
        print(f"Found {len(df)} results. Generating plots...")
        plot_mutation_performance(df)
        print(f"Plots saved to {PLOTS_DIR}")
    else:
        print("No results found yet.")
