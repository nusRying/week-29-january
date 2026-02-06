# Hypotheses: GA Mutation Role in Rule evolution

## Hypothesis 1: Structural Stagnation (Mutation OFF)
Disabling mutation will cause premature structural convergence. Without the ability to flip bits and change feature specification patterns, the system will be limited to recombining existing structures (Crossover only) or relying on initial covering.
- **Effect**: Failure to discover higher-order interactions.
- **Observed Behavior**: Repeated reuse of the same feature subsets; early convergence to local optima.

## Hypothesis 2: Representational Jump (Mutation ON)
Mutation is the primary mechanism for breaking "representational inertia." It allows the LCS to jump between disconnected subspaces by altering feature specification patterns (flipping between "Specify" and "Don't Care").
- **Effect**: Escaping over-specific or over-general rule traps.
- **Observed Behavior**: Injection of new feature combinations not present in parents; survival of novel explanatory hypotheses.

## Hypothesis 3: Epistatic Interaction Discovery
Higher-order interactions (e.g., $f_i$ AND $f_j$) emerge when mutation introduces co-specification of features that were never previously active together in a single rule.
- **Effect**: Clinical relevance (e.g., combining CHOG texture with ABCD geometry).
- **Observed Behavior**: Increased interaction depth in the evolved population.
