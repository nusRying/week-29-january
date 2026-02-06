# GA Mutation Study: Strategic Batch Summary

## ✅ Reduction Confirmed

**Original Plan**: 180 experiments
**Strategic Plan**: 84 experiments  
**Reduction**: 96 experiments (53.3% reduction)

---

## Breakdown

### Part B.1: Factorial (30 runs)
- ✅ Kept ALL 30 (no reduction)
- 3 conditions × 2 datasets × 5 seeds
- Most critical comparisons

### Part B.2: Dose-Response (18 runs)
- ❌ Reduced from 50 → 18
- 3 mutation rates (0.0, 0.04, 0.12) instead of 5
- 3 seeds instead of 5

### Part C: Interaction (36 runs)
- ❌ Reduced from 100 → 36
- 3 mutation rates instead of 5
- 3 seeds instead of 5

---

## Currently Running

**4 parallel batches** (~21 experiments each):
- Batch 1: 21 experiments
- Batch 2: 21 experiments  
- Batch 3: 21 experiments
- Batch 4: 21 experiments

**Expected time**: ~42 hours (vs 168 hours sequential, vs 360 hours original)

---

## Files Cleaned Up

Deleted unnecessary files:
- ✅ `batch_runner.py` (original 180-run version)
- ✅ `create_parallel_batches.py` (one-time generator)
- ✅ `launch_parallel.bat` (didn't work properly)

**Kept**:
- ✅ `batch_runner_strategic.py` (84-run orchestrator)
- ✅ `batch_1_of_4.py` through `batch_4_of_4.py` (parallel runners)
- ✅ `run_mutation_study.py` (core experiment logic)
- ✅ All documentation files

---

**Status**: All 4 batches running in parallel ✅
