"""
Quick test to verify covering event logging works with new hook.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import csv

# LCS Path
LCS_PATH = Path("c:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/scikit-ExSTraCS-master")
sys.path.append(str(LCS_PATH))
from skExSTraCS import ExSTraCS

SCRIPT_PATH = Path(__file__).resolve()
STUDY_ROOT = SCRIPT_PATH.parent
DATA_DIR = STUDY_ROOT / "data"
TEST_LOG = STUDY_ROOT / "test_covering_log.csv"

class CoveringStudyExSTraCS(ExSTraCS):
    """Extended ExSTraCS with covering event logging via callback hook."""
    
    def __init__(self, log_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_path = log_path
        self.covering_events = []
        
        # Set up CSV logger
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'num_specified', 'generality', 'fitness', 'accuracy', 'specified_features'])
        
        # Register covering callback
        self.covering_logger = self._log_covering_event
    
    def _log_covering_event(self, classifier, iteration):
        """Callback invoked during covering to log classifier details."""
        num_specified = len(classifier.specifiedAttList)
        d = self.env.formatData.numAttributes
        generality = 1.0 - (num_specified / d)
        
        # Log to CSV
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration,
                num_specified,
                generality,
                classifier.fitness,
                classifier.accuracy,
                ','.join(map(str, sorted(classifier.specifiedAttList)))
            ])

def test_covering_logging():
    """Quick test with minimal iterations."""
    print("Testing covering event logging...")
    print(f"Log file: {TEST_LOG}\n")
    
    # Load data
    data_file = DATA_DIR / "ham_clean.csv"
    df = pd.read_csv(data_file)
    
    train_idx, _ = train_test_split(
        range(len(df)), test_size=0.2, random_state=42, 
        stratify=df['label']
    )
    
    X_train = df.drop('label', axis=1).iloc[train_idx].values[:100]  # Only 100 instances for speed
    y_train = df['label'].iloc[train_idx].values[:100]
    
    d = X_train.shape[1]
    p_spec = 0.5
    rsl = max(1, min(d, int(round(2 * p_spec * d - 1))))
    
    print(f"Dataset: HAM (100 instances)")
    print(f"Features: {d}")
    print(f"p_spec: {p_spec}")
    print(f"RSL: {rsl}")
    print(f"Iterations: 500 (quick test)\n")
    
    # Run model
    model = CoveringStudyExSTraCS(
        log_path=TEST_LOG,
        learning_iterations=500,  # Quick test
        rule_specificity_limit=rsl,
        do_GA_subsumption=False,
        do_correct_set_subsumption=False,
        chi=0.0,
        mu=0.0
    )
    
    print("Training...", end='', flush=True)
    model.fit(X_train, y_train)
    print(" Done!\n")
    
    # Check log
    if TEST_LOG.exists():
        log_df = pd.read_csv(TEST_LOG)
        num_events = len(log_df)
        
        print("="*60)
        print("LOGGING TEST RESULTS")
        print("="*60)
        print(f"Total covering events logged: {num_events}")
        
        if num_events > 0:
            print("\n✅ SUCCESS! Covering events are being logged.\n")
            print("Sample of logged events:")
            print(log_df.head(10).to_string(index=False))
            print(f"\n... and {max(0, num_events-10)} more events")
            
            # Statistics
            print(f"\nStatistics:")
            print(f"  Mean specificity: {log_df['num_specified'].mean():.1f} features")
            print(f"  Mean generality: {log_df['generality'].mean():.3f}")
            print(f"  Events per iteration: {num_events / 500:.2f}")
        else:
            print("\n❌ FAILED: No covering events logged.")
            print("The hook may not be triggering properly.")
    else:
        print("❌ Log file not created!")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_covering_logging()
