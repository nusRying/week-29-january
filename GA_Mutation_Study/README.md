# GA Mutation Study: Causal Population Dynamics in ExSTraCS

## Scientific Objective
To rigorously demonstrate the causal role of mutation in preventing structural stagnation, enabling feature interaction discovery, and facilitating escape from local fitness plateaus in Learning Classifier Systems.

## Core Hypotheses

### H1: Structural Stagnation Prevention
**Mutation ON**: Structural Novelty Rate (SNR) remains > 0 throughout training  
**Mutation OFF**: SNR rapidly approaches 0 (population freezes structurally)

### H2: Feature Interaction Discovery
**Mutation ON**: Continuous discovery of new feature pairs/triples  
**Mutation OFF**: Interaction set saturates early

### H3: Plateau Escape
**Mutation ON**: Escapes local optima more frequently  
**Mutation OFF**: Gets trapped in suboptimal fitness regions

## Experimental Design

### Part B.1: Factorial Design (Core Evidence)
- **Conditions**:
  - GA ON, Mutation ON (mu=0.04, baseline)
  - GA ON, Mutation OFF (mu=0.0)
  - GA OFF (control, covering-only)
- **Datasets**: HAM10000, ISIC2019
- **Seeds**: 42-46
- **Total**: 2 × 3 × 5 = **30 runs**

### Part B.2: Dose-Response Analysis (Graded Evidence)
- **Mutation Rates**: 0.0, 0.01, 0.03, 0.05, 0.10
- **GA**: ON (theta_GA=25)
- **Datasets**: HAM10000, ISIC2019
- **Seeds**: 42-46
- **Total**: 2 × 5 × 5 = **50 runs**

### Part C: Covering × Mutation Interaction
- **p_spec**: 0.3 (general), 0.7 (specific)
- **Mutation Rates**: 0.0, 0.01, 0.03, 0.05, 0.10
- **Datasets**: HAM10000, ISIC2019
- **Seeds**: 42-46
- **Total**: 2 × 2 × 5 × 5 = **100 runs**

**Grand Total**: 180 runs

## Key Metrics

### 1. Structural Novelty Rate (SNR)
```
SNR(t) = |new_masks(t)| / |total_masks(t)|
```
Primary evidence for hypothesis H1.

### 2. New Feature Interaction Discovery Rate (NFIDR)
```
NFIDR(t) = |new_pairs(t)| / |total_pairs(t)|
```
Primary evidence for hypothesis H2.

### 3. Generality Variance
```
Var(g) where g = 1 - (#specified / d)
```
Indicates exploration breadth.

### 4. Plateau Escape Frequency
Count of fitness improvements > ε after W iterations of stagnation.

## Folder Structure
```
GA_Mutation_Study/
├── data/                    # Standardized datasets
├── configs/                 # YAML configurations
├── runs/                    # Results organized hierarchically
│   └── {part}/{condition}/{dataset}/seed_{num}/
│       ├── results.json
│       └── training_log.csv
├── snapshots/               # Population snapshots at checkpoints
│   └── {part}/{condition}/{dataset}/seed_{num}/
│       └── iter_{000000}.csv
├── results/                 # Aggregated metrics
├── figures/                 # Generated plots
├── analysis/                # Metric computation scripts
└── README.md
```

## How to Run

### Quick Start (Single Condition)
```python
from run_mutation_study import run_mutation_run
run_mutation_run("PartB1", "ga_on_mut_on", "ham", seed=42, mu=0.04, theta_ga=25)
```

### Full Experimental Suite
```bash
python batch_runner.py  # Runs all 180 configurations with progress tracking
```

### Analysis
```bash
cd analysis
python compute_metrics.py  # Generates metrics_summary.csv
python plot_comparative.py  # Creates SNR, NFIDR, and variance plots
```

## Expected Results

| Condition | SNR@500k | Unique Pairs | Gen Variance |
|-----------|----------|--------------|--------------|
| mu=0.04   | > 0.05   | > 5000       | > 0.02       |
| mu=0.0    | < 0.01   | < 2000       | < 0.005      |
| GA OFF    | ≈ 0      | < 1000       | ≈ 0          |

## PhD Contribution
This study provides:
1. **Causal Evidence**: Mutation directly drives structural exploration
2. **Dose-Response**: Effect is graded, not binary
3. **Interaction Effects**: Covering biases mutation's exploration manifold

## Citation
If using this experimental framework, cite:
- ExSTraCS: Urbanowicz et al. (2017)
- LCS Mutation Analysis: [Your thesis]
