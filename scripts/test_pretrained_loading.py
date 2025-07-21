#!/usr/bin/env python3
"""
Test Pretrained Model Loading
=============================

Test script to verify that pretrained BiSeNetV2 models can be properly
loaded and adapted for UAV landing detection.
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.enhanced_architectures import create_enhanced_model
from models.pretrained_loader import PretrainedModelLoader, load_cityscapes_bisenetv2


def test_model_creation():
    """Test basic model creation."""
    print("🧪 Testing model creation...")
    
    model = create_enhanced_model(
        model_type="enhanced_bisenetv2",
        num_classes=4,
        input_resolution=(512, 512),
        uncertainty_estimation=True
    )
    
    # Test forward pass
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        outputs = model(x)
        
    print(f"✅ Model created successfully")
    print(f"   Output keys: {list(outputs.keys())}")
    print(f"   Main output shape: {outputs['main'].shape}")
    if 'uncertainty' in outputs:
        print(f"   Uncertainty shape: {outputs['uncertainty'].shape}")


def test_pretrained_loader():
    """Test pretrained model loader."""
    print("\n🧪 Testing pretrained model loader...")
    
    # Check model_pths directory
    model_pths = Path("../../model_pths")
    if not model_pths.exists():
        print(f"❌ Model directory not found: {model_pths}")
        return False
    
    # Initialize loader
    loader = PretrainedModelLoader("../../model_pths", verbose=True)
    
    # List available models
    print("\n📋 Available models:")
    loader.list_available_models()
    
    if not loader.available_models:
        print("❌ No pretrained models found")
        return False
    
    return True


def test_weight_adaptation():
    """Test weight adaptation from Cityscapes to landing detection."""
    print("\n🧪 Testing weight adaptation...")
    
    # Create target model
    model = create_enhanced_model(
        model_type="enhanced_bisenetv2",
        num_classes=4,  # Landing classes
        uncertainty_estimation=False  # Simpler for testing
    )
    
    try:
        # Try auto-loading Cityscapes weights
        adapted_model, adaptation_info = load_cityscapes_bisenetv2(
            model=model,
            model_paths_root="../../model_pths",
            freeze_backbone=False
        )
        
        print("✅ Weight adaptation successful!")
        
        # Test forward pass with adapted model
        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            outputs = adapted_model(x)
            
        print(f"✅ Forward pass with adapted weights successful")
        print(f"   Output shape: {outputs['main'].shape}")
        print(f"   Expected: [2, 4, 512, 512] (batch, classes, height, width)")
        
        # Verify output shape is correct
        expected_shape = (2, 4, 512, 512)
        if outputs['main'].shape == expected_shape:
            print("✅ Output shape verification passed")
        else:
            print(f"❌ Output shape mismatch: {outputs['main'].shape} vs {expected_shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ Weight adaptation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_integration():
    """Test integration with training pipeline."""
    print("\n🧪 Testing training pipeline integration...")
    
    try:
        from training.enhanced_training_pipeline import create_training_config
        
        # Create training config with auto-detection
        config = create_training_config(
            model_type="enhanced_bisenetv2",
            training_mode="fast"
        )
        
        # Set auto-detection
        config['model']['pretrained_path'] = "auto_detect_cityscapes_bisenetv2"
        
        print("✅ Training config with auto-detection created")
        print(f"   Model type: {config['model']['type']}")
        print(f"   Pretrained path: {config['model']['pretrained_path']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Pretrained Model Loading System")
    print("=" * 50)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Pretrained Loader", test_pretrained_loader),
        ("Weight Adaptation", test_weight_adaptation),
        ("Training Integration", test_training_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pretrained model loading is ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 