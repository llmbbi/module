
import sys
import os
sys.path.append(os.getcwd())
try:
    from interpretability_lib_childguard.run_pipeline_childguard import load_childguard_data
    print("Import successful")
    
    data_path = "interpretability_lib_childguard/data/ChildGuard/ChildGuard.csv"
    if os.path.exists(data_path):
        print(f"Data file found at {data_path}")
        dataset = load_childguard_data(data_path)
        print("Data loaded successfully")
        print(f"Train size: {len(dataset['train'])}")
        print(f"Val size: {len(dataset['validation'])}")
        print(f"Test size: {len(dataset['test'])}")
        print(f"Sample: {dataset['train'][0]}")
    else:
        print(f"Data file NOT found at {data_path}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
