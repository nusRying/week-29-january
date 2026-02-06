# Binary Classification Experiment
**Benign (0) vs. Malignant (1)**

This experiment implements a rigorous binary classification pipeline for skin lesion analysis, strictly adhering to the following class mappings.

## 1. Class Definitions

| Class | Type | Mapping |
| :--- | :--- | :--- |
| **NV** (Melanocytic Nevus) | Benign | `0` |
| **BKL** (Benign Keratosis) | Benign | `0` |
| **DF** (Dermatofibroma) | Benign | `0` |
| **VASC** (Vascular Lesion) | Benign | `0` |
| **MEL** (Melanoma) | Malignant | `1` |
| **BCC** (Basal Cell Carcinoma) | Malignant | `1` |
| **SCC** (Squamous Cell Carcinoma) | Malignant | `1` |

*Note: Other classes (e.g., AK, UNK) are excluded.*

## 2. Pipeline Overview

### Step 1: Preprocessing & Labeling (`create_binary_dataset.py`)
-   **Input**: `ISIC_2019_Training_GroundTruth.csv`
-   **Output**: `binary_experiment_metadata.csv`
-   **Function**: Filters dataset to the 7 target classes and applies the `0/1` labels.

### Step 2: Feature Extraction (`extract_binary_features.py`)
-   **Input**: Raw Images (`images_train/`) + `binary_experiment_metadata.csv`
-   **Output**: `binary_features_all.csv`
-   **Process**:
    1.  **Hair Removal**: DullRazor-like BlackHat morphology + Inpainting.
    2.  **Color Normalization**: Gray World assumption.
    3.  **Feature Extraction**: ABCD, Texture (GLCM/LBP), Color Moments, Wavelets (approx).
-   **Resumable**: This script saves progress incrementally. If stopped, run again to continue.

### Step 3: Model Training (`train_binary_experiment.py`)
-   **Input**: `binary_features_all.csv`
-   **Output**: `Models/Binary_Exp_Model_YYYYMMDD.pkl`
-   **Model**: ExSTraCS (Learning Classifier System).
-   **Configuration**:
    -   `PopSize (N)`: 3000
    -   `Iterations`: 200,000
    -   `nu`: 5 (Power parameter)
-   **Evaluation**: Uses Stratified Train/Test split (80/20) and Train-set oversampling to handle class imbalance.

## 3. How to Run

**Full Auto-Pipeline**:
```bash
python train_binary_experiment.py
```
*(This will automatically trigger extraction if features are missing/incomplete, then train)*.
