#!/usr/bin/env python3
"""
Debug Model Output Format
========================

Investigate what the MMSegBiSeNetV2 model actually returns to fix the training.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.mmseg_bisenetv2 import MMSegBiSeNetV2

def debug_model_output():
    """Debug what the model actually outputs."""
    print("Creating model...")
    model = MMSegBiSeNetV2(num_classes=24)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    height, width = 256, 256
    dummy_input = torch.randn(batch_size, 3, height, width)
    
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nModel output type: {type(output)}")
    
    if isinstance(output, dict):
        print(f"Output is a dictionary with keys: {list(output.keys())}")
        for key, value in output.items():
            print(f"  {key}: {type(value)} - {value.shape if hasattr(value, 'shape') else 'No shape'}")
            if hasattr(value, 'shape') and len(value.shape) == 4:
                print(f"    Shape: {value.shape} (batch={value.shape[0]}, channels={value.shape[1]}, h={value.shape[2]}, w={value.shape[3]})")
    elif isinstance(output, torch.Tensor):
        print(f"Output is a tensor with shape: {output.shape}")
    elif isinstance(output, (list, tuple)):
        print(f"Output is a {type(output)} with {len(output)} elements:")
        for i, item in enumerate(output):
            print(f"  [{i}]: {type(item)} - {item.shape if hasattr(item, 'shape') else 'No shape'}")
    else:
        print(f"Output is of type: {type(output)}")
        print(f"Output: {output}")

if __name__ == "__main__":
    debug_model_output() 