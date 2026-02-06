import pandas as pd
import yaml
import json
from pathlib import Path
import sys

SCRIPT_PATH = Path(__file__).resolve()
STUDY_ROOT = SCRIPT_PATH.parent
CONFIGS_DIR = STUDY_ROOT / "configs"
PROGRESS_FILE = STUDY_ROOT / "results" / "progress.json"

# Import the main runner
sys.path.append(str(STUDY_ROOT))
from run_covering_study import run_part_a

DATASETS = ["ham", "isic"]
P_SPEC_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
SEEDS = [42, 43, 44, 45, 46]

def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"completed": [], "failed": []}

def save_progress(progress):
    PROGRESS_FILE.parent.mkdir(exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def run_all_covering_experiments():
    progress = load_progress()
    total = len(DATASETS) * len(P_SPEC_VALUES) * len(SEEDS)
    completed_count = len(progress["completed"])
    
    print(f"=== Covering Study Batch Runner ===")
    print(f"Total Experiments: {total}")
    print(f"Completed: {completed_count}")
    print(f"Remaining: {total - completed_count}")
    
    for ds in DATASETS:
        for ps in P_SPEC_VALUES:
            for sd in SEEDS:
                tag = f"{ds}_pspec{ps}_seed{sd}"
                
                if tag in progress["completed"]:
                    print(f"[SKIP] {tag} (already completed)")
                    continue
                
                print(f"\n[{completed_count+1}/{total}] Running: {tag}")
                try:
                    run_part_a(ds, ps, sd)
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
    
    if progress["failed"]:
        print("\nFailed runs:")
        for fail in progress["failed"]:
            print(f"  - {fail['tag']}: {fail['error']}")

if __name__ == "__main__":
    run_all_covering_experiments()
