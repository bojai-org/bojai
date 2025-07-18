#!/usr/bin/env python3
"""
Script to inspect the contents of a PyTorch .bin file
"""

import torch
import sys
from pathlib import Path

def inspect_model(model_path):
    """Inspect the contents of a PyTorch model file"""
    print(f"=== Inspecting: {model_path} ===\n")
    
    try:
        # Load the model
        model_data = torch.load(model_path, map_location='cpu')
        
        print(f"Type: {type(model_data)}")
        print(f"Type name: {type(model_data).__name__}")
        print(f"Is callable: {callable(model_data)}")
        
        if hasattr(model_data, '__len__'):
            print(f"Length: {len(model_data)}")
        
        if hasattr(model_data, 'keys'):
            print(f"Keys: {list(model_data.keys())}")
        
        if hasattr(model_data, 'state_dict'):
            print(f"Has state_dict: True")
            state_dict = model_data.state_dict()
            print(f"State dict keys: {list(state_dict.keys())}")
        
        if hasattr(model_data, 'parameters'):
            print(f"Has parameters: True")
            param_count = sum(p.numel() for p in model_data.parameters())
            print(f"Total parameters: {param_count}")
        
        # Try to get more details about the structure
        print(f"\n=== Detailed Structure ===")
        if isinstance(model_data, dict):
            for key, value in model_data.items():
                print(f"Key: {key}")
                print(f"  Type: {type(value)}")
                if hasattr(value, 'shape'):
                    print(f"  Shape: {value.shape}")
                if hasattr(value, 'dtype'):
                    print(f"  Dtype: {value.dtype}")
                print()
        
        # Check if it's a state dict vs full model
        if isinstance(model_data, dict) and all(isinstance(v, torch.Tensor) for v in model_data.values()):
            print("This appears to be a state_dict (model weights only)")
            print("You need to load it into a model architecture first")
        elif callable(model_data):
            print("This appears to be a complete model")
        else:
            print("This is neither a state_dict nor a complete model")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    model_path = "../KNN.bin"  # Adjust path as needed
    
    if not Path(model_path).exists():
        print(f"Model file not found: {model_path}")
        print("Available files in current directory:")
        for file in Path(".").glob("*.bin"):
            print(f"  {file}")
        sys.exit(1)
    
    inspect_model(model_path) 