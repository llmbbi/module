
import sys
import os

print("Checking environment...")
try:
    import unsloth
    print(f"Unsloth version: {unsloth.__version__ if hasattr(unsloth, '__version__') else 'unknown'}")
except ImportError as e:
    print(f"Error importing unsloth: {e}")
    sys.exit(1)

from interpretability_lib_mabsa.fine_tuning import LoRAFineTunerMABSA

model_name = "unsloth/gemma-3-1b-it"
print(f"Initializing LoRAFineTunerMABSA with {model_name}...")

tuner = LoRAFineTunerMABSA(model_name, "test_output")

print("Loading model (mocking if possible or real load)...")
# We'll try to load it. If it fails, we know why.
# Note: This might take time/memory. We'll rely on the user having a GPU if they are running this.
# If no GPU, it might fail or run on CPU slowly.

try:
    tuner.load_model(load_in_4bit=True)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

print("Detecting target modules...")
try:
    modules = tuner._detect_target_modules()
    print(f"Detected modules: {modules}")
except Exception as e:
    print(f"Error detecting modules: {e}")
    sys.exit(1)

if not modules:
    print("WARNING: No modules detected!")

print("Configuring LoRA...")
try:
    tuner.configure_lora()
    print("LoRA configured successfully.")
except Exception as e:
    print(f"Error configuring LoRA: {e}")
    sys.exit(1)

print("Reproduction script finished.")
