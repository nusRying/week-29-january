# Covering Study: Initial Hypothesis Geometry in ExSTraCS

## Scientific Objective
To empirically demonstrate how the covering mechanism in ExSTraCS defines the starting geometry of the search space through implicit "don't-care" encoding, and to quantify the relationship between the rule specification probability (`p_spec`) and the resulting population diversity.

## Hypothesis
Covering acts as a controlled hypothesis injection mechanism. The `rule_specificity_limit` parameter governs the initial balance between generality and specificity. **Higher specificity limits lead to more specific initial rules, potentially restricting the evolutionary exploration space.**

## Experimental Design

### Factors
- **Datasets**: HAM10000, ISIC2019 (Binary: Malignant vs Benign)
- **p_spec values**: 0.1, 0.3, 0.5, 0.7, 0.9 (directly controlling attribute specification probability during covering)
- **Seeds**: 42, 43, 44, 45, 46
- **Total Runs**: 2 datasets × 5 p_spec × 5 seeds = **50 runs**

**Note**: The rule specificity pressure parameter (ν) is held constant to isolate covering effects.

### Isolation Strategy
- **GA**: OFF (`theta_GA = 10^9`)
- **Mutation**: OFF (`mu = 0.0`)
- **Iterations**: 10,000 (sufficient for population saturation)
  - Preliminary pilot runs confirmed that covering frequency stabilizes well before 10,000 iterations across all p_spec values.

### Key Metrics
1. **Specificity Ratio**: `#specified / total_features` (should approximate `p_spec`)
2. **Feature-wise Bias**: Probability that each feature is specified during covering
3. **Covering Frequency**: Number of covering events per iteration window

## Folder Structure
```
Covering_Study/
├── data/                    # Standardized datasets
├── configs/                 # YAML configurations per condition
├── runs/                    # Organized by dataset/pspec/seed
│   └── {dataset}/pspec_{value}/seed_{num}/
│       ├── report.json
│       └── model.pkl (optional)
├── logs/                    # Covering event logs (CSV)
│   └── log_{dataset}_pspec{value}_seed{num}.csv
├── results/                 # Aggregated summaries
├── figures/                 # Generated plots
├── analysis/                # Analysis scripts
├── docs/                    # Scientific documentation
└── README.md
```

## How to Run

### 1. Verify Data Availability
```bash
ls data/  # Should show ham_clean.csv and isic_clean.csv
```

### 2. Execute All Runs
```bash
python run_covering_study.py
```

### 3. Analyze Results
```bash
python analyze_covering.py
```

### 4. View Results
- **Figures**: `figures/covering_pspec_dist_{dataset}.png`
- **Summary**: `results/summary_metrics.csv`

## Expected Outcomes
- **Low p_spec (0.1)**: Highly general rules, broad match coverage
- **High p_spec (0.9)**: Specific rules, narrow coverage, potential overfitting
- **Feature Bias**: Certain features may be preferentially specified due to data distribution
- **Null Expectation**: Under GA-OFF conditions, population diversity should converge solely as a function of p_spec, independent of dataset size.

## PhD Contribution
This study provides the first empirical characterization of how ExSTraCS covering defines the initial hypothesis manifold through implicit don't-care encoding. By isolating covering from genetic search operators, we demonstrate that rule generality, feature inclusion bias, and early population diversity are deterministically governed by p_spec. These results establish covering as a **geometric prior on the search space**, directly constraining the scope of subsequent mutation-driven exploration.

## Citations
- ExSTraCS: Urbanowicz and Browne (2017)
- LCS Covering Mechanisms: Wilson (1995, ZCS)
