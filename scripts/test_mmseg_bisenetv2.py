#!/usr/bin/env python3
"""
Test MMSeg BiSeNetV2 Pretrained Loading
=======================================

Test the MMSeg-compatible BiSeNetV2 model with real pretrained weights.
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.mmseg_bisenetv2 import create_mmseg_bisenetv2


def test_mmseg_model_creation():
    """Test basic MMSeg model creation."""
    print("🧪 Testing MMSeg BiSeNetV2 creation...")
    
    model = create_mmseg_bisenetv2(
        num_classes=4,
        uncertainty_estimation=True
    )
    
    # Test forward pass
    x = torch.randn(1, 3, 512, 512)
    model.eval()
    
    with torch.no_grad():
        outputs = model(x)
        
    print(f"✅ Model created and tested successfully")
    print(f"   Output keys: {list(outputs.keys())}")
    print(f"   Main output shape: {outputs['main'].shape}")
    if 'uncertainty' in outputs:
        print(f"   Uncertainty shape: {outputs['uncertainty'].shape}")
    
    return True


def test_pretrained_loading():
    """Test loading pretrained Cityscapes weights."""
    print("\n🧪 Testing pretrained weight loading...")
    
    # Find available pretrained models
    model_pths = Path("../../model_pths")
    pth_files = list(model_pths.glob("*cityscapes*.pth"))
    
    if not pth_files:
        print("❌ No Cityscapes BiSeNetV2 models found")
        return False
    
    # Use the first available model
    pretrained_path = pth_files[0]
    print(f"🔍 Using pretrained model: {pretrained_path.name}")
    
    try:
        # Create model and load pretrained weights
        model = create_mmseg_bisenetv2(
            num_classes=4,
            uncertainty_estimation=True,
            pretrained_path=str(pretrained_path)
        )
        
        print("✅ Pretrained weights loaded successfully!")
        
        # Test forward pass with pretrained weights
        x = torch.randn(2, 3, 512, 512)
        model.eval()
        
        with torch.no_grad():
            outputs = model(x)
            
        print(f"✅ Forward pass with pretrained weights successful")
        print(f"   Output shape: {outputs['main'].shape}")
        print(f"   Expected: [2, 4, 512, 512] (batch, classes, height, width)")
        
        # Verify output shape
        expected_shape = (2, 4, 512, 512)
        if outputs['main'].shape == expected_shape:
            print("✅ Output shape verification passed")
        else:
            print(f"❌ Output shape mismatch: {outputs['main'].shape} vs {expected_shape}")
            return False
            
        # Test uncertainty output
        if 'uncertainty' in outputs:
            uncertainty_shape = outputs['uncertainty'].shape
            expected_uncertainty_shape = (2, 1, 512, 512)
            if uncertainty_shape == expected_uncertainty_shape:
                print("✅ Uncertainty shape verification passed")
            else:
                print(f"❌ Uncertainty shape mismatch: {uncertainty_shape} vs {expected_uncertainty_shape}")
                return False
        
        # Check if weights are actually loaded (not all zeros)
        main_output_mean = outputs['main'].mean().item()
        if abs(main_output_mean) > 1e-6:  # Should not be exactly zero
            print(f"✅ Weights appear to be loaded (output mean: {main_output_mean:.6f})")
        else:
            print(f"⚠️ Output appears to be zeros (possible weight loading issue)")
        
        return True
        
    except Exception as e:
        print(f"❌ Pretrained loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_mode():
    """Test auxiliary outputs in training mode."""
    print("\n🧪 Testing training mode with auxiliary outputs...")
    
    try:
        model = create_mmseg_bisenetv2(
            num_classes=4,
            uncertainty_estimation=True
        )
        
        # Set to training mode
        model.train()
        
        # Use batch size > 1 to avoid BatchNorm issues
        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            outputs = model(x)
        
        print(f"✅ Training mode test successful")
        print(f"   Output keys: {list(outputs.keys())}")
        
        if 'aux' in outputs:
            aux_outputs = outputs['aux']
            print(f"   Auxiliary outputs: {len(aux_outputs)}")
            for i, aux in enumerate(aux_outputs):
                print(f"     Aux {i}: {aux.shape}")
            
            # Verify all aux outputs have correct shape
            expected_aux_shape = (2, 4, 512, 512)
            for i, aux in enumerate(aux_outputs):
                if aux.shape != expected_aux_shape:
                    print(f"❌ Aux output {i} shape mismatch: {aux.shape} vs {expected_aux_shape}")
                    return False
            
            print("✅ All auxiliary outputs have correct shape")
        else:
            print("⚠️ No auxiliary outputs found in training mode")
        
        return True
        
    except Exception as e:
        print(f"❌ Training mode test failed: {e}")
        return False


def compare_with_enhanced_model():
    """Compare MMSeg model with our enhanced model."""
    print("\n🧪 Comparing MMSeg vs Enhanced models...")
    
    try:
        from models.enhanced_architectures import create_enhanced_model
        
        # Create both models
        mmseg_model = create_mmseg_bisenetv2(num_classes=4, uncertainty_estimation=False)
        enhanced_model = create_enhanced_model("enhanced_bisenetv2", num_classes=4, uncertainty_estimation=False)
        
        # Compare parameter counts
        mmseg_params = sum(p.numel() for p in mmseg_model.parameters())
        enhanced_params = sum(p.numel() for p in enhanced_model.parameters())
        
        print(f"📊 Parameter comparison:")
        print(f"   MMSeg BiSeNetV2: {mmseg_params:,} parameters")
        print(f"   Enhanced BiSeNetV2: {enhanced_params:,} parameters")
        print(f"   Ratio: {enhanced_params/mmseg_params:.2f}x")
        
        # Test inference speed
        import time
        
        x = torch.randn(1, 3, 512, 512)
        
        # MMSeg model timing
        mmseg_model.eval()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = mmseg_model(x)
        mmseg_time = (time.time() - start_time) / 10
        
        # Enhanced model timing
        enhanced_model.eval()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = enhanced_model(x)
        enhanced_time = (time.time() - start_time) / 10
        
        print(f"⏱️ Inference timing (average of 10 runs):")
        print(f"   MMSeg BiSeNetV2: {mmseg_time*1000:.2f}ms")
        print(f"   Enhanced BiSeNetV2: {enhanced_time*1000:.2f}ms")
        print(f"   Speed ratio: {enhanced_time/mmseg_time:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ Model comparison failed: {e}")
        return False


def main():
    """Run all MMSeg BiSeNetV2 tests."""
    print("🚀 Testing MMSeg-Compatible BiSeNetV2")
    print("=" * 60)
    
    tests = [
        ("Model Creation", test_mmseg_model_creation),
        ("Pretrained Loading", test_pretrained_loading),
        ("Training Mode", test_training_mode),
        ("Model Comparison", compare_with_enhanced_model),
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
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MMSeg BiSeNetV2 is ready for training.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 