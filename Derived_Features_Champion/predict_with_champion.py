"""
Derived Features Champion - Standalone Prediction Interface
Use this script to predict if a skin lesion is Malignant or Benign using the champion model.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures

# 1. SETUP
SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_PATH = SCRIPT_DIR / "batched_fe_model.pkl"
METADATA_PATH = SCRIPT_DIR / "feature_metadata.json"

# Add local directory to path for feature_engine and ExSTraCS
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

# Add Local ExSTraCS to path for model loading
EXSTRACS_LOCAL_PATH = SCRIPT_DIR / "scikit-ExSTraCS-master"
if EXSTRACS_LOCAL_PATH.exists() and str(EXSTRACS_LOCAL_PATH) not in sys.path:
    sys.path.append(str(EXSTRACS_LOCAL_PATH))

try:
    import feature_engine
except ImportError:
    print("Error: feature_engine.py must be in the same folder as this script.")
    sys.exit(1)

def load_champion():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        print("Error: Model or Metadata files missing in current folder.")
        sys.exit(1)
        
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(METADATA_PATH, 'r') as f:
        meta = json.load(f)
        
    return model, meta

def predict_image(image_path, model, meta):
    # 1. Extract Base Features
    try:
        raw_feats = feature_engine.extract_raw_features_only(str(image_path))
    except Exception as e:
        return {"error": f"Feature extraction failed: {e}"}
    
    # 2. Align features to training order
    base_cols = meta['base_features']
    X_base = np.array([[raw_feats.get(col, 0) for col in base_cols]])
    X_base = np.nan_to_num(X_base)
    
    # 3. Generate Interactions
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_base) # Fits to X_base columns
    
    # 4. Select Top 100
    X_derived = X_poly[:, meta['top_indices']]
    
    # 5. Predict
    prediction = model.predict(X_derived)[0]
    result_name = "MALIGNANT" if prediction == 1 else "BENIGN"
    
    return {
        "prediction": int(prediction),
        "label": result_name,
        "image": os.path.basename(image_path)
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Predict with Derived Features Champion Model")
    parser.add_argument("--image", type=str, help="Path to a single image file")
    parser.add_argument("--folder", type=str, help="Path to a folder of images")
    args = parser.parse_args()
    
    if not args.image and not args.folder:
        parser.print_help()
        return

    print("Loading Champion Model...")
    model, meta = load_champion()
    
    if args.image:
        result = predict_image(args.image, model, meta)
        print(f"\nRESULT for {result['image']}:")
        print(f"Classification: {result['label']}")
        
    if args.folder:
        input_dir = Path(args.folder)
        images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
        print(f"Found {len(images)} images. Processing...")
        
        results = []
        for img_p in images:
            results.append(predict_image(img_p, model, meta))
            
        df = pd.DataFrame(results)
        output_csv = SCRIPT_DIR / "predictions_result.csv"
        df.to_csv(output_csv, index=False)
        print(f"Done! Results saved to {output_csv}")
        print(df['label'].value_counts())

if __name__ == "__main__":
    main()
