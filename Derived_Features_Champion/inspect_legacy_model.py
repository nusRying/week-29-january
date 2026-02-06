
import pickle
import sys
import os

# Add ExSTraCS to path - assumption based on previous context
sys.path.append(r"C:\Users\umair\Videos\PhD\PhD Data\Week 29 Jan\Code\Derived_Features_Champion\scikit-ExSTraCS-master")

try:
    from skExSTraCS.ExSTraCS import ExSTraCS
except ImportError:
    print("Could not import ExSTraCS directly, proceeding with pickle load which might fail if class is not in path")

model_path = r"C:\Users\umair\Videos\PhD\PhD Data\Week 29 Jan\Code\Derived_Features_Champion\batched_fe_model.pkl"

print(f"Inspecting model: {model_path}")

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    with open("legacy_model_info.txt", "w") as out:
        out.write(f"=== Model Attributes ===\n")
        attrs_to_check = [
            'N', 'nu', 'mu', 'rule_specificity_limit', 'learning_iterations', 
            'theta_GA', 'random_state', 'run_time', 'best_params'
        ]
        
        for attr in attrs_to_check:
            if hasattr(model, attr):
                out.write(f"{attr}: {getattr(model, attr)}\n")
            else:
                out.write(f"{attr}: Not found\n")

        out.write("\n=== Feature Info ===\n")
        if hasattr(model, 'env') and hasattr(model.env, 'formatData'):
            out.write(f"Num Attributes: {model.env.formatData.numAttributes}\n")
        
        if hasattr(model, 'metrics'):
            out.write(f"Metrics: {model.metrics}\n")
            
    print("Info written to legacy_model_info.txt")
    
except Exception as e:
    print(f"Error inspecting model: {e}")
