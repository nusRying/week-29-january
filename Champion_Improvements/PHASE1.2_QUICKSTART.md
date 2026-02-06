# Phase 1.2 Threshold Optimization - Quick Start

## Overview
**Runtime**: ~30 seconds (lightweight, safe to run alongside GA batches)
**CPU Usage**: <5% (minimal impact on other processes)

## What It Does
1. Loads best model from Phase 1.1 hyperparameter tuning
2. Computes ROC curve on full dataset
3. Tests 81 different probability thresholds (0.1 to 0.9)
4. Finds optimal threshold via Youden's J statistic
5. Generates 4 visualization plots
6. Reports improvement over default 0.5 threshold

## Prerequisites
- ✅ `champion_top100_features.csv` exists
- ✅ `models/best_hyperparam_model_fast.pkl` exists (from Phase 1.1 FAST)

## Run Command
```bash
cd "C:\Users\umair\Videos\PhD\PhD Data\Week 29 Jan\Code\Champion_Improvements"
python phase1_threshold_optimization.py
```

## Expected Output
```
============================================================
PHASE 1.2: OPTIMAL THRESHOLD TUNING
============================================================

Loading model from: models/best_hyperparam_model_fast.pkl
Loading dataset from: champion_top100_features.csv
Dataset loaded: 24464 samples

Computing predicted probabilities...
Finding optimal threshold...

Optimal Threshold (Youden's J): 0.456
  Sensitivity: 0.7654
  Specificity: 0.7821
  Balanced Accuracy: 0.7738
  Youden's J: 0.5475

Default Threshold (0.5) Performance:
  Sensitivity: 0.7234
  Specificity: 0.7912
  Balanced Accuracy: 0.7573

🎯 BA Improvement from threshold tuning: +0.0165 (+1.65%)
```

## Output Files
1. `phase1_results/threshold_optimization_results.csv` - All 81 threshold results
2. `phase1_results/threshold_optimization.png` - 4-panel visualization:
   - ROC curve
   - Sensitivity/Specificity vs Threshold
   - Youden's J vs Threshold
   - Balanced Accuracy vs Threshold
3. `phase1_results/optimal_threshold.json` - Best threshold config

## Resource Impact
- **CPU**: <5% (single-threaded, computation-light)
- **Memory**: ~500 MB
- **Time**: 20-40 seconds
- **Safe to run alongside**: GA batches, other experiments

## Why It's Lightweight
✅ No model training (just inference)
✅ Single pass through data
✅ Vectorized NumPy operations
✅ Non-interactive matplotlib backend
✅ No heavy I/O operations

## Next Step
After completion, run:
```bash
python phase1_degree3_features.py
```

---

**Ready to run anytime! Won't interfere with your GA mutation batches.** ✅
