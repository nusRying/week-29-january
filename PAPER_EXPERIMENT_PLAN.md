# Paper Experiment Plan
**Title:** _Enhancing Dermatological Classification through Evolutionary Rule-Based Learning to Classify Skin Lesions across Domains_

---

## 1. Problem Addressed

Skin lesion classification presents unique challenges due to **domain shift, lighting variation, and acquisition artifacts (dermatoscope types)**, which significantly degrade model generalization. State-of-the-art (SOTA) approaches are often **prone to overfitting source domain artifacts** (catastrophic forgetting).

At the same time, **Learning Classifier Systems (LCS)**, while interpretable, historically struggle with **high-dimensional image inputs**, suffering from scalability issues.

**Core problem tackled by the paper:**

> How to enable LCS to operate effectively on high-dimensional skin lesion data while maintaining classification accuracy and robustness across unseen clinical domains (ISIC $\rightarrow$ HAM10000).

---

## 2. Proposed Methodology

### Proposed Framework

We propose the development of a hybrid workflow combining domain-invariant feature engineering with evolutionary rule-based classification.

### Components

#### (a) Invariant Feature Engine
-   **We will implement** a comprehensive multi-modal extraction pipeline including:
    -   **Geometric**: ABCD (Asymmetry, Border, Color, Diameter), Hu Moments.
    -   **Texture**: GLCM, Local Binary Patterns (LBP), Gabor Filters.
    -   **Frequency**: Wavelet Transforms.
    -   **Color**: Color Histograms and Correlograms.
-   **Future Work**: We will also actively research and integrate the latest feature descriptors from literature to ensure optimal representation.
-   **Objective**: Learn compact, biologically-relevant features.

#### (b) ExSTraCS (Extended Supervised Tracking and Classifying System)
-   **We will utilize ExSTraCS** to operate on the derived feature vector instead of raw pixels.
-   **Mechanism**: Learning interval-based rules (e.g., `IF Asymmetry > 0.5 AND Texture < 0.2 THEN Malignant`).
-   **Benefit**: Supports generalization while retaining **white-box interpretability**.

#### (c) Polynomial Interaction Expansion
-   **We will generate** Degree-2 interactions to capture non-linear correlations (solving the XOR problem in biological data).

---

## 3. Anticipated Contributions

1.  **To introduce** a Domain-Invariant Feature Engine that extracts spatially meaningful, robust bio-markers.
2.  **To propose** a novel end-to-end hybrid system (Features + ExSTraCS) for high-accuracy skin lesion classification.
3.  **To demonstrate** that LCS can achieve **higher generalization (Zero-Shot)** on unseen datasets compared to SOTA baselines.
4.  **To extend** LCS capabilities to **high-dimensional medical data** while preserving **rule interpretability**.
5.  **To provide evidence** that the framework is **robust against domain shift**, unlike existing SOTA models.

---

## 4. Research Questions (Implied)

1.  **Performance:**
    Does ExSTraCS outperform state-of-the-art (SOTA) methods on unseen target domains (HAM10000)?
2.  **Statistical Significance:**
    Are the observed improvements statistically significant?
3.  **Interpretability:**
    Do the learned rules generalize well and remain interpretable to clinicians?
4.  **Sensitivity Analysis:**
    How do key parameters (Population Size $N$, Mutation Rate $\mu$) affect performance/convergence?

---

## 5. Dataset and Experimental Setup

### Dataset
-   **Source**: ISIC 2019 (25,331 images) - "The Training Ground".
-   **Target**: HAM10000 (10,015 images) - "The Unseen Test".
-   **Classes**: Binary (Malignant vs Benign) [Mapped from Multi-class].

### Preprocessing
-   **Feature Extraction**: Fixed vector of 167 raw features $\rightarrow$ 100 Selected Interactions.
-   **Normalization**: MinMaxScaler (fit on Source only).
-   **Experiments repeated 30 times** (via Seed), reporting **mean $\pm$ standard deviation**.

### Baseline Models
-   State-of-the-art (SOTA) CNNs (EfficientNet / ResNet).
-   Sequential Transfer Learning (for comparison).
-   Standard ML (Random Forest, XGBoost).

---

## 6. Evaluation Metrics

The following metrics are reported:
-   **Balanced Accuracy (BA)** [Critical due to class imbalance].
-   **Sensitivity (Recall)**.
-   **Specificity**.
-   **F-measure (F1-score)**.

---

## 7. Statistical Tests Performed

To validate performance improvements, we use:
-   **One-way ANOVA** (95% confidence level).
-   **Post-hoc multiple comparison tests**: Tukey HSD / Bonferroni.

Significance thresholds: $p < 0.05$ (ANOVA), $p < 0.01$ (Post-hoc).

---

## 8. Planned Experiments (Methodology)

We will conduct experiments on **Cross-Domain Classification Tasks**.

### Test A: Source Domain Performance (ISIC 2019)
-   **Method:** Train and validate on ISIC 2019 using 5-fold Cross Validation.
-   **Goal:** Establish a performance baseline (Balanced Accuracy) against standard CNNs (ResNet/EfficientNet).

### Test B: Target Domain Generalization (HAM10000) - "Zero-Shot"
-   **Method:** Apply the ISIC-trained model directly to HAM10000 without re-training.
-   **Goal:** Quantify the "Domain Invariance" of the Derived Features compared to pixel-based CNNs.

### Test C: Transfer Learning (Fine-Tuning)
-   **Method:** Pre-train ExSTraCS on ISIC $\rightarrow$ Fine-tune on HAM10000 (Sequential Transfer).
-   **Goal:** Determine if computational Domain Adaptation improves or degrades performance (Negative Transfer check).

### 9. Parameter Sensitivity Studies

### (a) Population Size (N)
-   **We will vary** $N \in \{500, 1000, 2000, 3000, 6000\}$.
-   **Objective**: To determine if accuracy saturates at an optimal capacity (estimated $N \approx 3000$).

### (b) Evolutionary Dynamics (Mutation)
-   **We will test**: Mutation Rate $\mu$ and GA Threshold $\theta_{GA}$.
-   **Objective**: To find the balance between exploitation and exploration for stable convergence.

### (c) Feature Dimensionality
-   **We will compare**: Top 50 vs Top 100 vs Top 150 features.
-   **Objective**: To validate the scalability of the LCS on high-dimensional data.

---

## 10. Rule Interpretability and Explanation

### Generalization
-   **Objective**: To demonstrate that rules contain **interval predicates**, covering multiple disparate images that share biological traits.

### Visualization
-   **Feature Importance Heatmaps**: We will generate heatmaps showing which Degree-2 interactions drive decisions.
-   **Population Coverage Plots**: We will visualize how rules evolved to cover the instance space.

---

## 11. One-Paragraph Summary (Prosposal)

This paper proposes the **Derived Features Champion**, a hybrid framework combining **invariant feature engineering** with **ExSTraCS** to enable interpretable, robust skin lesion classification. By extracting class-aware bio-markers, the method allows learning classifier systems to operate effectively on high-dimensional dermatological data. Across large-scale domain generalization tasks (ISIC to HAM10000), we aim to demonstrate that the framework outperforms Transfer Learning strategies outcomes in terms of Balanced Accuracy and Stability. We will validate improvements using ANOVA tests, conduct extensive sensitivity analyses on Population Size and Mutation Rates, and demonstrate interpretability through rule visualization.

---

## Table 3. Feature & Model Configuration (ExSTraCS)

| Component | Configuration |
| :--- | :--- |
| **Feature Extractor** | Domain-Invariant Feature Engine (ABCD, GLCM, LBP, Wavelets, etc.) |
| **Input Feature Size** | 100 (Degree-2 Interactions) |
| **Feature Type** | Biological, Spatial, Texture |
| **Classifier** | ExSTraCS (Supervised LCS) |
| **Rule Representation** | Interval-based predicates |
| **Rule Initialization** | Covering (Data-driven) |
| **Rule Evolution** | Genetic Algorithm (Steady State) |
| **Iterations** | 500,000 |
| **Interpretability** | Rule Lists + Attribute Tracking |

---

## Table 4. Evaluation Metrics Used

| Metric | Formula | Purpose |
| :--- | :--- | :--- |
| **Balanced Accuracy** | (Sensitivity + Specificity) / 2 | Handle Class Imbalance |
| **Sensitivity** | TP / (TP + FN) | Catching Malignant Cases |
| **Specificity** | TN / (TN + FP) | Reducing False Alarms |
| **Mean $\pm$ Std** | Across 30 seeds | Statistical Stability |

---

## Table 5. Statistical Tests Performed

| Test Type | Purpose | Confidence Level |
| :--- | :--- | :--- |
| **One-way ANOVA** | N-Study & Mutation Analysis | 95% ($p < 0.05$) |
| **Tukey HSD** | Pairwise parameter comparison | $p < 0.01$ |

---

## Table 6. Parameter Sensitivity Experiments

### (A) Population Size (N)

| Parameter | Values To Be Tested |
| :--- | :--- |
| **Population (N)** | 500, 1000, 2000, 3000, 6000 |

### (B) Mutation Study

| Parameter | Values To Be Tested |
| :--- | :--- |
| **Mutation Rate** | 0.1, 0.3, 0.5 (Varied $\mu$) |

---

## Table 7. Interpretability Experiments

| Analysis | Method |
| :--- | :--- |
| **Rule Coverage** | Populating Dynamics Plots |
| **Feature Importance** | Tracking attribute usage freq |
| **Visual Explanation** | Mapping top rules to images |

---

## Table 8. Reporting Format (Target)

| Element | Plan |
| :--- | :--- |
| **Runs per condition** | 30 (Seeds) |
| **Reported stats** | Mean $\pm$ Std |
| **Figures** | BA Convergence, Pop Size, Heatmaps |
| **Statistical validation** | ANOVA on Study Results |
| **Parameter sweeps** | One factor at a time (N, Mutation) |
| **Interpretability** | Visual examples of "Champion Rules" |
