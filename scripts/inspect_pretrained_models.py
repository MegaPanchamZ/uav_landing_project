#!/usr/bin/env python3
"""
Inspect Pretrained BiSeNetV2 Models
===================================

This script inspects the structure and weights of the available
BiSeNetV2 models to understand how to properly load them.
"""

import torch
from pathlib import Path
import sys

def inspect_model(model_path: str):
    """Inspect a single model file."""
    print(f"\n🔍 Inspecting: {Path(model_path).name}")
    print("=" * 60)
    
    try:
        # Load the model
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f" Model loaded successfully")
        
        # Check what's in the checkpoint
        print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
        
        # Get the state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("📦 Using 'state_dict' key")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("📦 Using 'model' key")
        else:
            state_dict = checkpoint
            print("📦 Using root checkpoint as state_dict")
        
        # Analyze layer names and shapes
        print(f"\n📊 Model structure ({len(state_dict)} layers):")
        
        layer_categories = {
            'backbone': [],
            'decoder': [],
            'classifier': [],
            'aux': [],
            'other': []
        }
        
        for name, param in state_dict.items():
            if any(x in name.lower() for x in ['backbone', 'resnet', 'layer']):
                layer_categories['backbone'].append((name, param.shape))
            elif any(x in name.lower() for x in ['decode', 'up', 'conv_last']):
                layer_categories['decoder'].append((name, param.shape))
            elif any(x in name.lower() for x in ['classifier', 'cls', 'head']):
                layer_categories['classifier'].append((name, param.shape))
            elif 'aux' in name.lower():
                layer_categories['aux'].append((name, param.shape))
            else:
                layer_categories['other'].append((name, param.shape))
        
        for category, layers in layer_categories.items():
            if layers:
                print(f"\n🏗️ {category.upper()} layers ({len(layers)}):")
                for name, shape in layers[:5]:  # Show first 5
                    print(f"   {name}: {shape}")
                if len(layers) > 5:
                    print(f"   ... and {len(layers) - 5} more")
        
        # Check output layers specifically (these need adaptation)
        print(f"\n Output/Classifier layers:")
        output_layers = [
            (name, param.shape) for name, param in state_dict.items()
            if any(x in name.lower() for x in ['classifier', 'cls', 'head', 'conv_last'])
        ]
        
        for name, shape in output_layers:
            print(f"   {name}: {shape}")
            if len(shape) >= 1:
                num_classes = shape[0]
                print(f"     → Output classes: {num_classes}")
        
        # Total parameters
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"\n📈 Total parameters: {total_params:,}")
        
        # Check for common issues
        print(f"\n⚠️ Potential issues:")
        has_module_prefix = any(name.startswith('module.') for name in state_dict.keys())
        if has_module_prefix:
            print("   - Has 'module.' prefix (DataParallel)")
        
        cityscapes_layers = [name for name, param in state_dict.items() 
                           if any(x in name.lower() for x in ['classifier', 'cls']) and param.shape[0] == 19]
        if cityscapes_layers:
            print(f"   - Cityscapes format (19 classes): {cityscapes_layers}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False


def main():
    """Inspect all available BiSeNetV2 models."""
    print("🚀 BiSeNetV2 Model Inspector")
    print("=" * 50)
    
    model_pths_dir = Path("../../model_pths")
    
    if not model_pths_dir.exists():
        print(f"❌ Model directory not found: {model_pths_dir}")
        return 1
    
    # Find all .pth files
    pth_files = list(model_pths_dir.glob("*.pth"))
    
    if not pth_files:
        print(f"❌ No .pth files found in {model_pths_dir}")
        return 1
    
    print(f"📁 Found {len(pth_files)} model files:")
    for pth_file in pth_files:
        print(f"   {pth_file.name}")
    
    # Inspect each model
    success_count = 0
    for pth_file in pth_files:
        if inspect_model(str(pth_file)):
            success_count += 1
    
    print(f"\n📊 Summary: {success_count}/{len(pth_files)} models inspected successfully")
    
    return 0 if success_count == len(pth_files) else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 