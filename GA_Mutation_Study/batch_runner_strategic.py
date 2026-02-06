import pandas as pd
import yaml
import json
from pathlib import Path
import sys

SCRIPT_PATH = Path(__file__).resolve()
STUDY_ROOT = SCRIPT_PATH.parent
PROGRESS_FILE = STUDY_ROOT / "results" / "progress_strategic.json"

sys.path.append(str(STUDY_ROOT))
from run_mutation_study import run_mutation_run

DATASETS = ["ham", "isic"]
SEEDS_STRATEGIC = [42, 43, 44]  # Reduced from 5 to 3 seeds

# Define STRATEGIC experimental configurations (84 total)
EXPERIMENTS = []

# Part B.1: Factorial (KEEP ALL - most important)
# 3 conditions × 2 datasets × 5 seeds = 30 runs
for ds in DATASETS:
    for cond, mu, theta in [("ga_on_mut_on", 0.04, 25), ("ga_on_mut_off", 0.0, 25), ("ga_off", 0.0, 10**9)]:
        for sd in [42, 43, 44, 45, 46]:  # Keep all 5 seeds for core comparisons
            EXPERIMENTS.append({"part": "PartB1", "condition": cond, "dataset": ds, "seed": sd, "mu": mu, "theta_ga": theta, "pspec": 0.5})

# Part B.2: Dose-Response (STRATEGIC REDUCTION)
# 3 mutation rates × 2 datasets × 3 seeds = 18 runs (down from 50)
for ds in DATASETS:
    for mu in [0.0, 0.04, 0.12]:  # Low, mid, high (down from 5 rates)
        for sd in SEEDS_STRATEGIC:
            EXPERIMENTS.append({"part": "PartB2", "condition": f"mu_{mu}", "dataset": ds, "seed": sd, "mu": mu, "theta_ga": 25, "pspec": 0.5})

# Part C: Interaction (STRATEGIC REDUCTION)
# 2 p_spec × 3 mutation rates × 2 datasets × 3 seeds = 36 runs (down from 100)
for ds in DATASETS:
    for ps in [0.3, 0.7]:
        for mu in [0.0, 0.04, 0.12]:  # Low, mid, high
            for sd in SEEDS_STRATEGIC:
                EXPERIMENTS.append({"part": "PartC", "condition": f"pspec{ps}_mu{mu}", "dataset": ds, "seed": sd, "mu": mu, "theta_ga": 25, "pspec": ps})

def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"completed": [], "failed": []}

def save_progress(progress):
    PROGRESS_FILE.parent.mkdir(exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def run_all_mutation_experiments():
    progress = load_progress()
    total = len(EXPERIMENTS)
    completed_count = len(progress["completed"])
    
    print(f"=== GA Mutation Study (STRATEGIC) Batch Runner ===")
    print(f"Total Experiments: {total} (strategic reduction from 180)")
    print(f"Part B.1: 30 runs (all seeds)")
    print(f"Part B.2: 18 runs (3 mutation rates, 3 seeds)")
    print(f"Part C:   36 runs (3 mutation rates, 3 seeds)")
    print(f"Completed: {completed_count}")
    print(f"Remaining: {total - completed_count}")
    print(f"Estimated time (sequential): {(total - completed_count) * 2:.0f} hours")
    print(f"Estimated time (4-way parallel): {(total - completed_count) * 2 / 4:.0f} hours")
    
    for i, exp in enumerate(EXPERIMENTS, 1):
        tag = f"{exp['part']}_{exp['condition']}_{exp['dataset']}_seed{exp['seed']}"
        
        if tag in progress["completed"]:
            print(f"[SKIP] {tag}")
            continue
        
        print(f"\n[{i}/{total}] Running: {tag}")
        try:
            run_mutation_run(exp['part'], exp['condition'], exp['dataset'], 
                           exp['seed'], exp['mu'], exp['theta_ga'], exp['pspec'])
            progress["completed"].append(tag)
            save_progress(progress)
            completed_count += 1
        except Exception as e:
            print(f"[ERROR] {tag} failed: {e}")
            progress["failed"].append({"tag": tag, "error": str(e)})
            save_progress(progress)
    
    print(f"\n=== Batch Complete ===")
    print(f"Successful: {len(progress['completed'])}/{total}")
    print(f"Failed: {len(progress['failed'])}/{total}")

if __name__ == "__main__":
    run_all_mutation_experiments()
