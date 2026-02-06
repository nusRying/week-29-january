# Evolutionary Derma-Classifier: Cross-Domain Skin Lesion Analysis

**Repository for PhD Feature Engineering & Evolutionary Learning Experiments**

This project implements a hybrid **Domain-Invariant Feature Learning + Learning Classifier System (ExSTraCS)** framework designed to solve the problem of **domain shift** in dermatological imaging (ISIC 2019 $\rightarrow$ HAM10000).

---

## 🏗️ Project Architecture

The codebase is organized into modular experiments as defined in the research proposal:

```
├── Derived_Features_Champion/       # [CORE] The Final Proposed Model (Zero-Shot)
│   ├── feature_engine.py            # Extracts ABCD, GLCM, Wavelet Features
│   ├── pipeline_feature_selection.py# Selects Top 100 Co-adapted Interactions
│   ├── pipeline_retrain_model.py    # Trains the ExSTraCS LCS Model
│   └── CHAMPION_TECHNICAL_REPORT.md # Full performance methodology
│
├── GA_Mutation_Study/               # [EXP 1] Evolutionary Dynamics Analysis
│   ├── run_mutation_study.py        # 500k iteration sensitivity sweep
│   └── snapshots/                   # Model checkpoints (LFS Tracked)
│
├── N_Multipliers_Study/             # [EXP 2] Population Sizing Analysis
│   ├── run_n_study.py               # Testing N=500 to N=6000
│   └── results/                     # Saturation curves
│
├── Transfer_Learning_ISIC_to_HAM/   # [EXP 3] Transfer Learning Baseline
│   └── run_transfer_learning.py     # Sequential fine-tuning pipeline
│
└── scikit-ExSTraCS-master/          # [LIB] Custom Extended supervised LCS
```

---

## ⚡ Quick Start

### 1. Prerequisites
The project relies on specific versions of `numpy` (for LCS rule matching) and `opencv`.

```bash
pip install -r requirements.txt
```

### 2. Running a Prediction (Champion Model)
To use the trained champion model on a new image:

```bash
cd Derived_Features_Champion
python predict_with_champion.py --image "path/to/skin_lesion.jpg"
```

### 3. Reproducing Experiments
To reproduce the transfer learning or mutation studies:

```bash
# Example: Run the transfer learning benchmark
cd Transfer_Learning_ISIC_to_HAM
python run_transfer_learning.py
```

---

## 📊 Key Results

| Metric | ISIC 2019 (Training) | HAM 10000 (Zero-Shot) | Stability |
| :--- | :---: | :---: | :---: |
| **Balanced Acc** | **72.28%** | **72.37%** | **+0.09%** |
| **Sensitivity** | 72.39% | 74.05% | High Recall |

**Finding**: The Derived Features approach (72.37%) outperforms standard Computational Transfer Learning (67.6%) on the target domain.

---

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
