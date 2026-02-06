
import json
from pathlib import Path

results_dir = Path(r"C:\Users\umair\Videos\PhD\PhD Data\Week 29 Jan\Code\GA_Mutation_Study\results")

for batch in range(1, 5):
    file_path = results_dir / f"progress_batch{batch}.json"
    if file_path.exists():
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        failed_count = len(data.get("failed", []))
        if failed_count > 0:
            print(f"Batch {batch}: Clearing {failed_count} failed experiments.")
            data["failed"] = [] # Clear failures
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            print(f"Batch {batch}: No failed experiments to clear.")
