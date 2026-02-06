# Phase 2: Multi-Configuration Ensemble

## Overview
Trains 5 ExSTraCS models with different `p_spec` and `mu` combinations, then combines them via weighted voting.

**Expected Gain**: +1.5-2% BA over best single model

---

## Strategy: Diversity Through Configuration

### Ensemble Members (Based on Your Studies)

| Model | p_spec | mu | Weight | Purpose |
|-------|--------|-----|--------|---------|
| **General_LowMut** | 0.3 | 0.01 | 15% | Broad rules, stable learning |
| **Optimal_LowMut** | 0.5 | 0.01 | 25% | Balanced, conservative |
| **Optimal_MidMut** | 0.5 | 0.04 | **30%** | **Champion baseline** |
| **Optimal_HighMut** | 0.5 | 0.08 | 20% | Exploration-focused |
| **Specific_LowMut** | 0.7 | 0.01 | 10% | Precise edge cases |

**Total Weight**: 100%

---

## Why This Works

### 1. **p_spec Diversity** (From Covering Study)
- **p_spec=0.3**: General rules → good for common patterns
- **p_spec=0.5**: Optimal balance → best individual performance
- **p_spec=0.7**: Specific rules → captures rare/complex cases

### 2. **Mutation Rate Diversity** (From GA Mutation Study)
- **mu=0.01**: Low mutation → stable, exploits known good rules
- **mu=0.04**: Baseline → balanced exploration/exploitation
- **mu=0.08**: High mutation → explores rule space more aggressively

### 3. **Weighted Voting**
- Higher weight (30%) on **Optimal_MidMut** (your champion config)
- Moderate weights on variations
- Lower weight on extreme configs (edge case specialists)

---

## Prerequisites

**Required**:
- ✅ `champion_top100_features.csv` OR
- ✅ `phase1_results/degree3_enhanced_features.csv` (preferred)

**Optional** (auto-detected):
- `phase1_results/best_configuration_fast.json` (uses Phase 1 best N, nu, theta_GA)

---

## Usage

```bash
cd "C:\Users\umair\Videos\PhD\PhD Data\Week 29 Jan\Code\Champion_Improvements"
python phase2_ensemble_training.py
```

---

## Runtime

**5 models × 15 min each = ~75 minutes (1.25 hours)**

Each model:
- 200,000 iterations (moderate for speed)
- Full training + validation
- Saved individually

---

## Output Files

### `phase2_results/`
1. **ensemble_individual_results.csv** - Performance of each model
2. **ensemble_summary.json** - Overall ensemble metrics

### `models/`
3. **ensemble_General_LowMut.pkl** through **ensemble_Specific_LowMut.pkl** (5 models)
4. **ensemble_predictor.py** - Ready-to-use prediction script

---

## Expected Results

### Individual Models (Validation BA)
- General_LowMut: ~0.720
- Optimal_LowMut: ~0.730
- **Optimal_MidMut**: ~**0.735** (best individual)
- Optimal_HighMut: ~0.725
- Specific_LowMut: ~0.715

### Ensemble (Weighted Voting)
- **Validation BA**: ~**0.750** (+1.5% over best individual)
- **Test BA**: ~**0.745**

**Key**: Ensemble outperforms even the best individual model due to diversity!

---

## Using the Ensemble

After training, use the generated predictor:

```python
import sys
sys.path.append('C:/Users/umair/Videos/PhD/PhD Data/Week 29 Jan/Code/Champion_Improvements/models')

from ensemble_predictor import predict_ensemble
import pandas as pd

# Load your test data
X_test = ...  # Your feature matrix

# Predict
y_pred, y_proba = predict_ensemble(X_test)

print(f"Predictions: {y_pred}")
print(f"Probabilities: {y_proba}")
```

---

## Next Steps

### Phase 3 (Optional): Further Improvements
- **Data augmentation** (physics-informed)
- **Hard negative mining**
- **External data integration** (ISIC2020)
- **Two-stage cascade architecture**

---

## Scientific Rationale

**Ensemble learning principle**: Diverse models make different errors. By combining them, we:
1. ✅ Reduce variance (more stable predictions)
2. ✅ Capture different patterns (p_spec variations)
3. ✅ Balance exploration/exploitation (mu variations)
4. ✅ Maintain interpretability (still rule-based, not black box)

**Your covering/mutation studies provided the optimal diversity recipe!**

---

**Run Phase 2 when Phase 1 hyperparameter tuning completes (~6 hours from now)**
