#!/usr/bin/env python3
"""
Test script for the organized UAV Landing Detection system
"""
import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_imports():
    """Test that all modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(" PyTorch available")
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    try:
        import onnxruntime as ort
        print(" ONNX Runtime available")
    except ImportError:
        print("❌ ONNX Runtime not available")
        return False
    
    try:
        import cv2
        print(" OpenCV available")
    except ImportError:
        print("❌ OpenCV not available")
        return False
    
    return True

def test_model_files():
    """Test that required model files exist."""
    print("\n📁 Testing model files...")
    
    required_files = [
        'trained_models/ultra_fast_uav_landing.onnx',
        'trained_models/ultra_stage1_best.pth',
        'trained_models/ultra_stage2_best.pth'
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            size_mb = Path(file_path).stat().st_size / 1024**2
            print(f" {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {file_path} - Missing!")
            all_exist = False
    
    return all_exist

def test_onnx_inference():
    """Test ONNX model inference."""
    print("\n🧠 Testing ONNX inference...")
    
    try:
        import onnxruntime as ort
        
        model_path = 'trained_models/ultra_fast_uav_landing.onnx'
        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            return False
        
        # Load model
        session = ort.InferenceSession(model_path)
        print(" Model loaded successfully")
        
        # Test inference
        dummy_input = np.random.rand(1, 3, 256, 256).astype(np.float32)
        
        # Warmup
        for _ in range(5):
            _ = session.run(None, {'input': dummy_input})
        
        # Benchmark
        times = []
        for _ in range(50):
            start = time.time()
            result = session.run(None, {'input': dummy_input})
            times.append((time.time() - start) * 1000)
        
        avg_time = np.mean(times)
        fps = 1000 / avg_time
        
        print(f" Inference successful")
        print(f"📊 Average time: {avg_time:.1f}ms")
        print(f"📊 FPS: {fps:.1f}")
        print(f"📊 Output shape: {result[0].shape}")
        
        # Verify output
        if result[0].shape == (1, 4, 256, 256):
            print(" Output shape correct")
        else:
            print(f"❌ Unexpected output shape: {result[0].shape}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX inference failed: {e}")
        return False

def test_classical_detector():
    """Test classical detector fallback."""
    print("\n🔍 Testing classical detector...")
    
    try:
        from classical_detector import ClassicalLandingDetector
        
        detector = ClassicalLandingDetector()
        print(" Classical detector initialized")
        
        # Test with synthetic image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        start = time.time()
        result = detector.detect_landing_zones(test_image)  # Correct method name
        inference_time = (time.time() - start) * 1000
        
        print(f" Classical inference: {inference_time:.1f}ms")
        print(f"📊 Found {len(result.get('zones', []))} zones")
        
        return True
        
    except Exception as e:
        print(f"❌ Classical detector failed: {e}")
        return False

def test_training_scripts():
    """Test that training scripts are accessible."""
    print("\n🏋️ Testing training scripts...")
    
    scripts = [
        'scripts/ultra_fast_training.py',
        'scripts/analyze_dataset.py', 
        'scripts/convert_to_onnx.py'
    ]
    
    all_exist = True
    for script in scripts:
        if Path(script).exists():
            print(f" {script}")
        else:
            print(f"❌ {script} - Missing!")
            all_exist = False
    
    return all_exist

def test_documentation():
    """Test that documentation exists."""
    print("\n📚 Testing documentation...")
    
    docs = [
        'README.md',
        'docs/TRAINING.md',
        'docs/API.md',
        'docs/DATASETS.md'
    ]
    
    all_exist = True
    for doc in docs:
        if Path(doc).exists():
            print(f" {doc}")
        else:
            print(f"❌ {doc} - Missing!")
            all_exist = False
    
    return all_exist

def test_visualizations():
    """Test that visualizations exist."""
    print("\n🎨 Testing visualizations...")
    
    viz_files = [
        'visualizations/model_architecture.png',
        'visualizations/training_pipeline.png'
    ]
    
    all_exist = True
    for viz_file in viz_files:
        if Path(viz_file).exists():
            print(f" {viz_file}")
        else:
            print(f"❌ {viz_file} - Missing!")
            all_exist = False
    
    return all_exist

def run_full_test():
    """Run complete test suite."""
    print("🚁 UAV Landing Detection - System Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Model Files", test_model_files), 
        ("ONNX Inference", test_onnx_inference),
        ("Classical Detector", test_classical_detector),
        ("Training Scripts", test_training_scripts),
        ("Documentation", test_documentation),
        ("Visualizations", test_visualizations)
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
    print("\n" + "=" * 50)
    print("🏁 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = " PASS" if result else "❌ FAIL"
        print(f"{status:<10} {test_name}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED! System ready for deployment!")
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1)
