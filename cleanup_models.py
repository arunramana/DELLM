"""Script to close all running llama models."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.llm_client import cleanup_all_models, get_loaded_models

if __name__ == "__main__":
    print("Checking for loaded models...")
    loaded = get_loaded_models()
    
    if loaded:
        print(f"Found {len(loaded)} loaded model(s):")
        for model_path in loaded:
            print(f"  - {model_path}")
        print("\nClosing all models...")
        cleanup_all_models()
    else:
        print("No models currently loaded.")

