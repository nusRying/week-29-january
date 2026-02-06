import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Paths
STUDY_ROOT = Path(__file__).resolve().parent
RUNS_DIR = STUDY_ROOT / "runs"
PLOTS_DIR = STUDY_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

def aggregate_results():
    results = []
    for report_file in RUNS_DIR.glob("**/report.json"):
        with open(report_file, 'r') as f:
            data = json.load(f)
            results.append(data)
    return pd.DataFrame(results)

def plot_covering_accuracy(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='p_spec_target', y='val_balanced_accuracy', hue='dataset', data=df)
    plt.title("Impact of Initial Bio-Informed Specificity (p_spec) on Accuracy")
    plt.xlabel("Target p_spec (Bio-Informed Prior)")
    plt.ylabel("Validation Balanced Accuracy")
    plt.grid(alpha=0.3)
    plt.savefig(PLOTS_DIR / "covering_accuracy_boxplot.png")
    plt.close()

if __name__ == "__main__":
    print("Aggregating covering study results...")
    df = aggregate_results()
    if not df.empty:
        print(f"Found {len(df)} results. Generating plots...")
        plot_covering_accuracy(df)
        print(f"Plots saved to {PLOTS_DIR}")
    else:
        print("No results found yet.")
