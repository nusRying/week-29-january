# Scientific Focus: The Geometry of Covering

## 1. What is Covering in ExSTraCS?
Covering is a targeted hypothesis injection mechanism. It triggers when the current input instance is not sufficiently covered by the existing rule population. Instead of random rule creation, it creates the simplest possible rule that matches the current instance and predicts the correct class.

## 2. The "Don't-Care" Implementation
In this implementation of ExSTraCS, "Don't-Care" is **implicit**. It is represented by the **absence** of a feature index in the `specifiedAttList`. 
- **Specified**: Feature index exists in `specifiedAttList`.
- **Don't-Care**: Feature index is missing (ignored during matching).

## 3. Experimental Parameters
- **rule_specificity_limit (RSL)**: This directly controls the `p_spec` (probability of specification). It determines how many features, on average, a new covering rule will consider.
- **RSL Low**: Rules are highly general (many don't-cares). High match coverage, low precision.
- **RSL High**: Rules are specific. Low match coverage, potentially high initial precision but prone to overfitting.

## 4. PhD Insight: The Starting Geometry
Covering defines the **starting geometry** of the search space. The Genetic Algorithm and Mutation can only explore what covering makes reachable. If covering is too specific, the GA may never find general rules. If too general, the discovery process may be overwhelmed by noise.

## 5. Experiment Logic
We will freeze the Evolutionary components (GA OFF, Mutation OFF) to isolate and observe the raw output of the covering mechanism over the first 10,000 iterations.
