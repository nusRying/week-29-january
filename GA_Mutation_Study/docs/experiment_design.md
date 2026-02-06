# Experiment Design: GA Mutation & Feature Discovery

## Objective
To analyze how mutation in the genetic algorithm affects rule diversity, convergence behavior, and feature interaction discovery in a Learning Classifier System (ExSTraCS).

## Experimental Factors
- **GA Control**: GA Enabled vs. Disabled.
- **Mutation Control**: Mutation Enabled vs. Disabled.
- **Replication**: Multiple fixed seeds (e.g., 42, 43, 44).
- **Dataset**: Dermatological features (CHOG + Wavelet + ABCD interactions).

## Evaluation Focus
- **Structural Diversity**: Measuring the variation in rule condition masks, not just classification accuracy.
- **Feature Co-occurrence Patterns**: Tracking how many features are co-specified in successful rules.
- **Rule Generality Evolution**: Monitoring the average number of "don't care" (*) symbols over time.
- **Niche Dynamics**: Observing how rules compete and survive within specific data niches.

## Scientific Framing
In Learning Classifier Systems, the Genetic Algorithm is not a simple optimizer of individuals; it is a **rule discovery and restructuring mechanism**. This study aims to prove that mutation is the primary operator that allows the system to "jump" across disconnected regions of the rule search space and discover higher-order feature interactions.
