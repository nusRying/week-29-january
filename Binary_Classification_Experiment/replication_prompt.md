# Replication Prompt

**User Request**:
"I want to create a strict Binary Classification Experiment (Benign vs. Malignant) for the ISIC2019 dataset using the ExSTraCS algorithm. Please set up a self-contained environment."

**Instructions for the AI**:

1.  **Project Structure**:
    Create a new folder named `Binary_Classification_Experiment`.
    Ensure `preprocessing_utils.py` and `feature_engine.py` are present in this folder (I will provide these files).

2.  **Step 1: Dataset Preparation (`create_binary_dataset.py`)**:
    Create a script that reads `ISIC_2019_Training_GroundTruth.csv` and generates a new metadata file `binary_experiment_metadata.csv`.
    **Strict Class Mapping**:
    *   **Benign (0)**: NV, BKL, DF, VASC
    *   **Malignant (1)**: MEL, BCC, SCC
    *   Exclude all other classes (e.g., AK, UNK).

3.  **Step 2: Feature Extraction (`extract_binary_features.py`)**:
    Create a script that:
    *   Imports the *local* `preprocessing_utils` and `feature_engine`.
    *   Reads `binary_experiment_metadata.csv`.
    *   Iterates through images, applying:
        1.  Hair Removal (BlackHat + Inpainting)
        2.  Color Normalization (Gray World)
        3.  Feature Extraction (ABCD, Texture, etc. from `feature_engine`)
    *   **Crucial**: Implement **Resume Capability**. It must check for an existing output file and skip already processed images, saving progress incrementally (e.g., every 100 images).

4.  **Step 3: Training (`train_binary_experiment.py`)**:
    Create a script that:
    *   Checks if the features file exists. If not, *automatically* runs `extract_binary_features.py`.
    *   Loads the features.
    *   Splits the data (80% Train / 20% Test, Stratified).
    *   Apply **RandomOversampling** to the Training set ONLY.
    *   Scales features (StandardScaler).
    *   Trains an **ExSTraCS** model (N=3000, Iterations=200,000, nu=5).
    *   Saves the model as `.pkl` and prints sensitivity/specificity metrics.

**Goal**: A fully automated pipeline where running `python train_binary_experiment.py` handles extraction and training in one go.
