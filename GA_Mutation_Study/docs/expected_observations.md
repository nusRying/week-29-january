# Expected Observations

## 🧪 Case A: Mutation OFF
- **Population Dynamics**: Stagnant number of unique feature masks after initial iterations.
- **Generality**: Generality levels will likely lock into a narrow range or fluctuate minimally based on crossover.
- **Discovery Rate**: The rate of new condition patterns (structural novelty) will drop toward zero.
- **Outcome**: Stagnant rule population that fails to solve complex epistatic problems.

## 🧪 Case B: Mutation ON
- **Population Dynamics**: Constant injection of novel feature masks throughout the run.
- **Generality**: Dynamic changes in rule generality as mutation intelligently specifies or generalizes rules.
- **Discovery Rate**: Discovery of rules like $R^*: f_3=1, f_7=0$ even if parents only had $(f_3=1, f_7=*)$ or $(f_3=*, f_7=0)$.
- **Survival Rate**: Observe that many mutations fail, but those that succeed (the "fittest" mutations) drive the overall performance jumps.

## 🧬 Feature Interaction Jumps
- **Target Observation**: A point in time where a mutated rule significantly outperforms its parents by combining previously isolated features.
- **Metric**: Mutation-driven feature co-occurrence frequency ($f_i$ and $f_j$ specified in the same rule).
