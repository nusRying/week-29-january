import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
STUDY_ROOT = Path(__file__).resolve().parent
LOGS_DIR = STUDY_ROOT / "logs"
PLOTS_DIR = STUDY_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

def plot_dynamics(log_path, tag):
    df = pd.read_csv(log_path)
    # The headers might be renamed already, check both
    cols = {c.lower(): c for c in df.columns}
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Accuracy
    acc_col = cols.get('accuracy')
    if acc_col:
        axes[0].plot(df['Iteration' if 'Iteration' in df.columns else 'iteration'], df[acc_col], color='blue')
        axes[0].set_ylabel('Accuracy (Rolling)')
        axes[0].set_title(f'Covering Dynamics: {tag}')
    
    # generality
    gen_col = cols.get('generality')
    if gen_col:
        axes[1].plot(df['Iteration' if 'Iteration' in df.columns else 'iteration'], df[gen_col], color='green')
        axes[1].set_ylabel('Generality')
    
    # num_specified
    spec_col = cols.get('numspecified') or cols.get('num_specified')
    if spec_col:
        axes[2].plot(df['Iteration' if 'Iteration' in df.columns else 'iteration'], df[spec_col], color='red')
        axes[2].set_ylabel('Specified Features')
        axes[2].set_xlabel('Iteration (Covering Events)')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"dynamics_{tag}.png")
    plt.close()

if __name__ == "__main__":
    # Plot for a few representative tags
    tags = ["isic_pspec0.5_seed42", "ham_pspec0.5_seed42"]
    for tag in tags:
        log_file = LOGS_DIR / f"log_{tag}.csv"
        if log_file.exists():
            print(f"Generating dynamics plot for {tag}...")
            plot_dynamics(log_file, tag)
    print(f"Dynamics plots saved to {PLOTS_DIR}")
