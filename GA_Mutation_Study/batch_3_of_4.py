"""
GA Mutation Study - Batch 3 of 4
Contains 21 experiments
Run this in a separate terminal for parallel execution
"""
import json
from pathlib import Path
import sys

STUDY_ROOT = Path(__file__).resolve().parent
sys.path.append(str(STUDY_ROOT))
from run_mutation_study import run_mutation_run

PROGRESS_FILE = STUDY_ROOT / "results" / f"progress_batch3.json"

EXPERIMENTS = [
    {
        "part": "PartB1",
        "condition": "ga_on_mut_on",
        "dataset": "ham",
        "seed": 44,
        "mu": 0.04,
        "theta_ga": 25,
        "pspec": 0.5
    },
    {
        "part": "PartB1",
        "condition": "ga_on_mut_off",
        "dataset": "ham",
        "seed": 43,
        "mu": 0.0,
        "theta_ga": 25,
        "pspec": 0.5
    },
    {
        "part": "PartB1",
        "condition": "ga_off",
        "dataset": "ham",
        "seed": 42,
        "mu": 0.0,
        "theta_ga": 1000000000,
        "pspec": 0.5
    },
    {
        "part": "PartB1",
        "condition": "ga_off",
        "dataset": "ham",
        "seed": 46,
        "mu": 0.0,
        "theta_ga": 1000000000,
        "pspec": 0.5
    },
    {
        "part": "PartB1",
        "condition": "ga_on_mut_on",
        "dataset": "isic",
        "seed": 45,
        "mu": 0.04,
        "theta_ga": 25,
        "pspec": 0.5
    },
    {
        "part": "PartB1",
        "condition": "ga_on_mut_off",
        "dataset": "isic",
        "seed": 44,
        "mu": 0.0,
        "theta_ga": 25,
        "pspec": 0.5
    },
    {
        "part": "PartB1",
        "condition": "ga_off",
        "dataset": "isic",
        "seed": 43,
        "mu": 0.0,
        "theta_ga": 1000000000,
        "pspec": 0.5
    },
    {
        "part": "PartC",
        "condition": "pspec0.3_mu0.0",
        "dataset": "ham",
        "seed": 42,
        "mu": 0.0,
        "theta_ga": 25,
        "pspec": 0.3
    },
    {
        "part": "PartC",
        "condition": "pspec0.3_mu0.04",
        "dataset": "ham",
        "seed": 43,
        "mu": 0.04,
        "theta_ga": 25,
        "pspec": 0.3
    },
    {
        "part": "PartC",
        "condition": "pspec0.3_mu0.12",
        "dataset": "ham",
        "seed": 44,
        "mu": 0.12,
        "theta_ga": 25,
        "pspec": 0.3
    },
    {
        "part": "PartC",
        "condition": "pspec0.7_mu0.04",
        "dataset": "ham",
        "seed": 42,
        "mu": 0.04,
        "theta_ga": 25,
        "pspec": 0.7
    },
    {
        "part": "PartC",
        "condition": "pspec0.7_mu0.12",
        "dataset": "ham",
        "seed": 43,
        "mu": 0.12,
        "theta_ga": 25,
        "pspec": 0.7
    },
    {
        "part": "PartC",
        "condition": "pspec0.3_mu0.0",
        "dataset": "isic",
        "seed": 44,
        "mu": 0.0,
        "theta_ga": 25,
        "pspec": 0.3
    },
    {
        "part": "PartC",
        "condition": "pspec0.3_mu0.12",
        "dataset": "isic",
        "seed": 42,
        "mu": 0.12,
        "theta_ga": 25,
        "pspec": 0.3
    },
    {
        "part": "PartC",
        "condition": "pspec0.7_mu0.0",
        "dataset": "isic",
        "seed": 43,
        "mu": 0.0,
        "theta_ga": 25,
        "pspec": 0.7
    },
    {
        "part": "PartC",
        "condition": "pspec0.7_mu0.04",
        "dataset": "isic",
        "seed": 44,
        "mu": 0.04,
        "theta_ga": 25,
        "pspec": 0.7
    },
    {
        "part": "PartB2",
        "condition": "mu_0.0",
        "dataset": "ham",
        "seed": 42,
        "mu": 0.0,
        "theta_ga": 25,
        "pspec": 0.5
    },
    {
        "part": "PartB2",
        "condition": "mu_0.04",
        "dataset": "ham",
        "seed": 43,
        "mu": 0.04,
        "theta_ga": 25,
        "pspec": 0.5
    },
    {
        "part": "PartB2",
        "condition": "mu_0.12",
        "dataset": "ham",
        "seed": 44,
        "mu": 0.12,
        "theta_ga": 25,
        "pspec": 0.5
    },
    {
        "part": "PartB2",
        "condition": "mu_0.04",
        "dataset": "isic",
        "seed": 42,
        "mu": 0.04,
        "theta_ga": 25,
        "pspec": 0.5
    },
    {
        "part": "PartB2",
        "condition": "mu_0.12",
        "dataset": "isic",
        "seed": 43,
        "mu": 0.12,
        "theta_ga": 25,
        "pspec": 0.5
    }
]

def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"completed": [], "failed": []}

def save_progress(progress):
    PROGRESS_FILE.parent.mkdir(exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

if __name__ == "__main__":
    progress = load_progress()
    exp_count = len(EXPERIMENTS)
    
    print(f"=== Batch 3/4 | {exp_count} experiments ===")
    
    for i, exp in enumerate(EXPERIMENTS, 1):
        tag = f"{exp['part']}_{exp['condition']}_{exp['dataset']}_seed{exp['seed']}"
        
        if tag in progress["completed"]:
            print(f"[SKIP] {tag}")
            continue
        
        print(f"\n[{i}/{exp_count}] Running: {tag}")
        try:
            run_mutation_run(exp['part'], exp['condition'], exp['dataset'], 
                           exp['seed'], exp['mu'], exp['theta_ga'], exp['pspec'])
            progress["completed"].append(tag)
            save_progress(progress)
        except Exception as e:
            print(f"[ERROR] {tag} failed: {e}")
            progress["failed"].append({"tag": tag, "error": str(e)})
            save_progress(progress)
    
    print(f"\n=== Batch 3 Complete ===")
    print(f"Successful: {len(progress['completed'])}/{exp_count}")
