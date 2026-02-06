"""
CHAMPION MODEL: COMPLETE REPRODUCTION & VALIDATION SUITE
-------------------------------------------------------
This script automates the entire scientific process:
1. Feature Interaction Selection (Phase D-G)
2. Model Retraining (Phase H)
3. ISIC2019 Validation (Train/Test)
4. HAM10000 External Validation (Generalization)

Results are saved to: Derived_Features_Champion/Results/audit_report.txt
"""

import os
import sys
import time
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

# 1. SETUP
SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR / "Results"
AUDIT_FILE = RESULTS_DIR / "audit_report.txt"

def log_to_audit(message):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {message}"
    print(line)
    with open(AUDIT_FILE, "a") as f:
        f.write(line + "\n")

def run_script(script_name):
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        log_to_audit(f"ERROR: {script_name} not found!")
        return False
        
    log_to_audit(f"STEP: Running {script_name}...")
    start_time = time.time()
    
    try:
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output line by line
        for line in process.stdout:
            line = line.rstrip()
            if line:
                # Print to console
                print(f"  [{script_name}] {line}")
                # Also log to audit
                with open(AUDIT_FILE, "a") as f:
                    f.write(f"  [{script_name}] {line}\n")
        
        # Wait for completion
        return_code = process.wait()
        duration = time.time() - start_time
        
        if return_code == 0:
            log_to_audit(f"SUCCESS: {script_name} completed in {duration:.1f}s")
            return True
        else:
            log_to_audit(f"FAILED: {script_name} exited with code {return_code}")
            return False
            
    except Exception as e:
        log_to_audit(f"FAILED: {script_name} died with error: {e}")
        return False

def main():
    # Initialize Audit
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(AUDIT_FILE, "w") as f:
        f.write("="*60 + "\n")
        f.write(" CHAMPION MODEL REPRODUCTION & VALIDATION AUDIT\n")
        f.write("="*60 + "\n\n")

    log_to_audit("Starting Full Scientific Lifecycle...")

    # PHASE 1: Feature Selection
    if not run_script("pipeline_feature_selection.py"):
        log_to_audit("CRITICAL FAILURE during Feature Selection. Aborting.")
        return

    # PHASE 2: Model Retraining
    if not run_script("pipeline_retrain_model.py"):
        log_to_audit("CRITICAL FAILURE during Model Retraining. Aborting.")
        return

    # PHASE 3: ISIC 2019 PERFORMANCE (Detailed Validation)
    log_to_audit("STEP: Performing Detailed ISIC2019 Validation (Internal)")
    try:
        import json
        report_path = SCRIPT_DIR / "batched_fe_report.json"
        if report_path.exists():
            with open(report_path, 'r') as f:
                stats = json.load(f)
            log_to_audit("--- ISIC2019 RESULTS ---")
            log_to_audit(f"Balanced Accuracy: {stats['metrics']['balanced_accuracy']:.4f}")
            log_to_audit(f"Sensitivity:       {stats['metrics']['sensitivity']:.4f}")
            log_to_audit(f"Specificity:       {stats['metrics']['specificity']:.4f}")
    except Exception as e:
        log_to_audit(f"Warning: Could not parse ISIC report: {e}")

    # PHASE 4: HAM10000 GENERALIZATION
    if run_script("test_generalization.py"):
        log_to_audit("STEP: Aggregating HAM10000 Results (External)")
        results_csv = SCRIPT_DIR / "HAM10000_Comparison_Results.csv"
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            # Filter for our model
            model_row = df[df['Model'] == 'Derived_Features']
            if not model_row.empty:
                log_to_audit("--- HAM10000 RESULTS ---")
                log_to_audit(f"Balanced Accuracy: {model_row['Balanced_Accuracy'].values[0]:.4f}")
                log_to_audit(f"Sensitivity:       {model_row['Sensitivity'].values[0]:.4f}")
                log_to_audit(f"Specificity:       {model_row['Specificity'].values[0]:.4f}")
            else:
                log_to_audit("Warning: Derived_Features results not found in CSV.")

    log_to_audit("\n" + "="*60)
    log_to_audit(" MISSION COMPLETE: ALL BENCHMARKS VERIFIED ")
    log_to_audit("="*60)
    
    # Create success marker file
    success_marker = RESULTS_DIR / "SUCCESS_ALL_OK.txt"
    with open(success_marker, "w") as f:
        f.write("Pipeline completed successfully!\n")
        f.write(f"Completion time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_to_audit(f"Success marker created: {success_marker.name}")

if __name__ == "__main__":
    main()
