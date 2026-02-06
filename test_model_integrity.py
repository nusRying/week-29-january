import pickle
import os

MODEL_PATH = r"c:\Users\umair\Videos\PhD\PhD Data\Week 29 Jan\Code\Transfer_Learning_ISIC_to_HAM\Results\source_isic_normalized.pkl"

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
else:
    try:
        with open(MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded model. Data length: {len(data)}")
        # Check if population is present (last element)
        pop = data[-1]
        print(f"Population type: {type(pop)}")
        print(f"Population size: {len(pop) if isinstance(pop, list) else 'N/A'}")
    except Exception as e:
        print(f"Failed to load model: {e}")
