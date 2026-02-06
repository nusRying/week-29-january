# Paper Experiment Plan

**Title:**  
_Enhancing Dermatological Classification through Evolutionary Rule-Based Learning for Cross-Domain Skin Lesion Analysis_

---

## 1. Problem Addressed

Automated skin lesion classification faces persistent challenges due to **domain shift**, arising from variations in dermatoscope devices, lighting conditions, acquisition protocols, and patient demographics. These factors significantly degrade model generalization when deployed across datasets.

State-of-the-art (SOTA) deep learning approaches often **overfit source-domain artifacts**, leading to unstable performance and catastrophic degradation on unseen clinical data.

At the same time, **Learning Classifier Systems (LCS)** offer inherent interpretability but have historically struggled with **high-dimensional image inputs**, suffering from scalability and representation limitations.

**Core problem addressed by this paper:**

> How can Learning Classifier Systems be enabled to operate effectively on high-dimensional skin lesion data while maintaining classification accuracy, interpretability, and robustness across unseen clinical domains (ISIC → HAM10000)?

---

## 2. Proposed Methodology

### 2.1 Overall Framework

We propose a **hybrid evolutionary learning framework** that decouples representation learning from decision learning by combining **domain-invariant feature engineering** with **evolutionary rule-based classification**.

This design explicitly targets domain shift while preserving interpretability.

---

### 2.2 Framework Components

#### (a) Domain-Invariant Feature Engine

We will implement a comprehensive multi-modal feature extraction pipeline designed to capture biologically meaningful and spatially robust characteristics of skin lesions.

**Feature categories include:**
- **Geometric:** ABCD (Asymmetry, Border, Color, Diameter), Hu Moments
- **Texture:** GLCM, Local Binary Patterns (LBP), Gabor Filters
- **Frequency:** Wavelet Transforms
- **Color:** Color Histograms and Correlograms

**Objective:**  
To learn compact, biologically relevant representations that reduce sensitivity to dataset-specific artifacts.
 
We will actively investigate and integrate newly proposed dermatological feature descriptors from recent literature.

---

#### (b) ExSTraCS (Extended Supervised Tracking and Classifying System)

We will utilize **ExSTraCS** to operate on the derived feature vectors rather than raw pixel data.

- **Rule Representation:** Interval-based conditions  
  *(e.g., IF Asymmetry > 0.5 AND Texture < 0.2 THEN Malignant)*
- **Learning Mechanism:** Steady-state genetic algorithm with covering and mutation
- **Key Benefit:** Enables generalization while maintaining **white-box interpretability**

---

#### (c) Polynomial Interaction Expansion

To model non-linear relationships commonly present in biological data, we will generate **degree-2 polynomial feature interactions**.

**Purpose:**
- Capture synergistic relationships between features
- Address XOR-type decision boundaries not representable by linear rules

---

## 3. Anticipated Contributions

1. Introduce a **domain-invariant feature engine** that extracts spatially meaningful dermatological bio-markers.
2. Propose a novel **end-to-end hybrid system (Features + ExSTraCS)** for skin lesion classification.
3. Demonstrate that LCS can achieve **strong zero-shot generalization** on unseen datasets.
4. Extend LCS applicability to **high-dimensional medical data** while preserving interpretability.
5. Provide empirical evidence of **robustness against domain shift** compared to SOTA models.

---

## 4. Research Questions

1. **Performance:**  
   Does ExSTraCS outperform SOTA methods on unseen target domains (HAM10000)?
2. **Statistical Significance:**  
   Are observed performance improvements statistically significant?
3. **Interpretability:**  
   Do learned rules generalize well and remain clinically interpretable?
4. **Sensitivity:**  
   How do population size and mutation rate affect convergence and performance?

---

## 5. Dataset and Experimental Setup

### 5.1 Datasets

- **Source Domain:** ISIC 2019 (25,331 images)
- **Target Domain:** HAM10000 (10,015 images)
- **Task:** Binary classification (Malignant vs Benign), mapped from multi-class labels

- **Malignant:** Melanoma (MEL), Basal Cell Carcinoma (BCC), Squamous Cell Carcinoma (SCC).
- **Benign:** Melanocytic Nevi (NV), Benign Keratosis (BKL), Dermatofibroma (DF), Vascular Lesions (VASC).

---

### 5.2 Preprocessing

- Feature extraction: 167 raw features → top 100 selected interactions
- Normalization: MinMaxScaler (fit on source domain only)
- Repeated experiments: 30 independent runs using different random seeds
- Reporting: Mean ± standard deviation

---

### 5.3 Baseline Models

- SOTA CNNs (EfficientNet, ResNet)
- Sequential Transfer Learning pipelines
- Classical ML models (Random Forest, XGBoost)

---

## 6. Evaluation Metrics

The following metrics are reported:

- **Balanced Accuracy (BA)** – primary metric
- Sensitivity (Recall)
- Specificity
- F1-score

| Metric | Formula | Purpose |
| :--- | :--- | :--- |
| **Balanced Accuracy** | (Sensitivity + Specificity) / 2 | Handle Class Imbalance |
| **Sensitivity** | TP / (TP + FN) | Catching Malignant Cases |
| **Specificity** | TN / (TN + FP) | Reducing False Alarms |
| **Mean $\pm$ Std** | Across 30 seeds | Statistical Stability |

Balanced Accuracy is emphasized due to class imbalance and clinical relevance.

---

## 7. Statistical Validation

To validate performance differences, we apply:

- **One-way ANOVA** (95% confidence level)
- **Post-hoc multiple comparison tests:** Tukey HSD and Bonferroni correction

**Significance thresholds:**  
- ANOVA: p < 0.05  
- Post-hoc: p < 0.01

---

## 8. Planned Experiments

### Test A: Source-Domain Performance (ISIC 2019)

- **Method:** 5-fold cross-validation on ISIC 2019
- **Goal:** Establish baseline performance relative to CNN models

---

### Test B: Target-Domain Generalization (HAM10000 – Zero-Shot)

- **Method:** Apply ISIC-trained models directly to HAM10000 without retraining
- **Goal:** Quantify domain invariance of derived features

---

### Test C: Sequential Transfer Learning

- **Method:** Pre-train on ISIC → fine-tune on HAM10000
- **Goal:** Assess whether domain adaptation improves or degrades performance

---

## 9. Parameter Sensitivity Studies

### (a) Population Size

- Tested values: N ∈ {500, 1000, 2000, 3000, 6000}
- Objective: Identify performance saturation and optimal capacity

---

### (b) Evolutionary Dynamics

- Parameters: Mutation rate μ, GA threshold θ_GA
- Objective: Balance exploration and exploitation for stable convergence

### (c) Generalization Control (P#)

- Parameters: Don't-care probability $P_{\#}$
- Objective: Evaluate impact on rule generality and overfitting

---

### (d) Feature Dimensionality

- Configurations: Top 50 vs 100 vs 150 features
- Objective: Validate scalability on high-dimensional inputs

---

## 10. Rule Interpretability and Explanation

### Generalization Analysis

- Evaluate interval predicate coverage across diverse lesion instances

### Visualization

- Feature importance heatmaps (degree-2 interactions)
- Rule population coverage and evolution plots

---

## 11. One-Paragraph Proposal Summary

This paper proposes a hybrid framework combining **domain-invariant feature engineering** with **ExSTraCS** to enable interpretable and robust skin lesion classification across clinical domains. By transforming raw images into biologically grounded descriptors, the approach allows Learning Classifier Systems to operate effectively on high-dimensional dermatological data. Through extensive cross-domain evaluation (ISIC → HAM10000), statistical validation, and sensitivity analysis, the framework aims to demonstrate superior generalization, stability, and interpretability compared to transfer-learning-based baselines.

---

## Table 3. Feature and Model Configuration (ExSTraCS)

| Component | Configuration |
|---------|---------------|
| Feature Extractor | Domain-Invariant Feature Engine |
| Input Feature Size | 100 (Degree-2 Interactions) |
| Feature Type | Biological, Spatial, Texture |
| Classifier | ExSTraCS |
| Rule Representation | Interval-based |
| Rule Initialization | Covering |
| Rule Evolution | Steady-State Genetic Algorithm |
| Iterations | 500,000 |
| Interpretability | Rule Lists and Attribute Tracking |

---

## Table 4. Evaluation Metrics

| Metric | Purpose |
|------|--------|
| Balanced Accuracy | Handle class imbalance |
| Sensitivity | Detect malignant cases |
| Specificity | Reduce false alarms |
| Mean ± Std | Stability across seeds |

---

## Table 5. Statistical Tests

| Test | Purpose | Confidence |
|-----|--------|-----------|
| One-way ANOVA | Parameter studies | 95% |
| Tukey HSD | Pairwise comparisons | p < 0.01 |

---

## Table 6. Parameter Sensitivity Experiments

### Population Size

| N Values |
|---------|
| 500, 1000, 2000, 3000, 6000 |

### Mutation Rate

| μ Values |
|---------|
| 0.1, 0.3, 0.5 |

### Feature Dimensionality

| Top K Features |
|---------|
| 50, 100, 150 |

### Generalization (P#)

| $P_{\#}$ Values |
|---------|
| 0.2, 0.33, 0.5, 0.8 |

---

## Table 7. Interpretability Experiments

| Analysis | Method |
|--------|--------|
| Rule Coverage | Population dynamics plots |
| Feature Importance | Attribute usage frequency |
| Visual Explanation | Mapping rules to images |

---

## Table 8. Reporting Protocol

| Element | Plan |
|-------|------|
| Runs per condition | 30 |
| Reported statistics | Mean ± Std |
| Figures | BA convergence, heatmaps |
| Statistical validation | ANOVA |
| Parameter sweeps | One-factor-at-a-time |
| Interpretability | Champion rule examples |

