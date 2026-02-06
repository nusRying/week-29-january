# GA Mutation Study: Parallel Execution Guide

## ✅ Current Status: RUNNING

**4 parallel batches launched successfully!**

Each batch is running in a separate terminal window with ~21 experiments each.

---

## Timeline

- **Total Runs**: 84 (strategic sampling)
- **Parallel Jobs**: 4
- **Per-Run Time**: ~2 hours
- **Sequential Time**: 168 hours (7 days)
- **Parallel Time**: **~42 hours (~1.75 days)** ✅

---

## What's Running

### Batch 1 (Terminal 1)
- 21 experiments
- Mix of Part B.1, B.2, and C

### Batch 2 (Terminal 2)
- 21 experiments  
- Mix of Part B.1, B.2, and C

### Batch 3 (Terminal 3)
- 21 experiments
- Mix of Part B.1, B.2, and C

### Batch 4 (Terminal 4)
- 21 experiments
- Mix of Part B.1, B.2, and C

---

## Monitoring Progress

Each terminal shows:
```
=== Batch X/4 | 21 experiments ===
[1/21] Running: PartB1_ga_on_mut_on_ham_seed42
...
```

Progress is saved independently:
- `results/progress_batch1.json`
- `results/progress_batch2.json`
- `results/progress_batch3.json`
- `results/progress_batch4.json`

---

## If a Batch Fails/Stops

Simply rerun that specific batch:
```bash
cd "C:\Users\umair\Videos\PhD\PhD Data\Week 29 Jan\Code\GA_Mutation_Study"
python batch_X_of_4.py
```

It will skip completed experiments and resume from where it stopped.

---

## Expected Completion

**Start Time**: ~19:46 (31 Jan 2026)
**Expected End**: ~13:00-14:00 (2 Feb 2026)

---

## Experiments Count Breakdown

**Part B.1** (30 runs):
- 3 conditions × 2 datasets × 5 seeds
- ALL seeds kept (most important comparisons)

**Part B.2** (18 runs):
- 3 mutation rates (0.0, 0.04, 0.12) × 2 datasets × 3 seeds
- Strategic sampling from 5 rates → 3 rates

**Part C** (36 runs):
- 2 p_spec (0.3, 0.7) × 3 mutations × 2 datasets × 3 seeds
- Strategic sampling: 3 seeds instead of 5

**Total: 84 runs** (down from 180)

---

## After Completion

All results will be in:
```
runs/
  PartB1/
    ga_on_mut_on/
    ga_on_mut_off/
    ga_off/
  PartB2/
    mu_0.0/
    mu_0.04/
    mu_0.12/
  PartC/
    pspec0.3_mu0.0/
    pspec0.3_mu0.04/
    etc.
```

Run analysis:
```bash
python analyze_mutation_study.py
```

---

## Tips

1. **Don't close the terminal windows** - let them run unattended
2. **Check progress occasionally** by viewing the terminal outputs
3. **System can sleep** - processes will pause and resume
4. **If one batch completes early** - that's OK, load is balanced
5. **Each experiment saves checkpoints** at iterations: 0, 10k, 50k, 100k, 200k, 300k, 400k, 500k

---

**Let it run! Check back in ~40-45 hours for complete results.**
