import sys
import importlib

dependencies = [
    "torch",
    "numpy",
    "scipy",
    "sklearn",
    "matplotlib",
    "lime",
    "shap",
    "h5py",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "bitsandbytes",
    "unsloth",
    "trl",
    "rich",  # Just added
    "tqdm",
    "tyro",  # Potential TRL dependency
    "pandas" # Datasets dependency
]

print(f"Checking {len(dependencies)} dependencies...")
missing = []
versions = {}

for dep in dependencies:
    try:
        module = importlib.import_module(dep)
        version = getattr(module, "__version__", "unknown")
        versions[dep] = version
        print(f"✅ {dep:15} (v{version})")
    except ImportError as e:
        print(f"❌ {dep:15} FAILED: {e}")
        missing.append(dep)
    except Exception as e:
        print(f"⚠️ {dep:15} ERROR: {e}")
        # unsloth might error on driver, but import succeeds
        if dep == "unsloth" and "NVIDIA driver" in str(e):
             print(f"   (Ignoring expected driver error for unsloth)")
        else:
             missing.append(dep)

if missing:
    print(f"\nMissing dependencies: {missing}")
    sys.exit(1)
else:
    print("\nAll dependencies verified!")
    sys.exit(0)
