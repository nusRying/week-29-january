# Experiment Design: Covering Mechanism Analysis

## Objective
To quantitatively analyze the rule initialization bias and don't-care distribution produced by the covering mechanism in isolation.

## Factors to Test
1. **Specificity Limit (p_spec)**: 
   - p_spec = 0.1, 0.3, 0.5, 0.7, 0.9 (directly controlling attribute specification probability during covering)
   - The rule specificity pressure parameter (ν) is held constant to isolate covering effects.
2. **Feature-wise Bias**: Observing if certain features are favored by covering based on their presence in the training instances.

## Metrics to Compute
1. **Specificity Ratio**: `num_specified / total_features`.
2. **Feature Coverage Map**: Probability of feature `j` being specified across all covering events.
3. **Generality Distribution**: Histogram of rule generality `g = 1 - (#spec / d)`.

## Setup
- **GA**: OFF (`theta_GA = 1e9`)
- **Mutation**: OFF (`mu = 0`)
- **Iterations**: 10,000 (enough to saturate initial population `N`)
- **Log**: `covering_log.csv` (recording every single rule creation event)
