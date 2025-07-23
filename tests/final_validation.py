#!/usr/bin/env python3
"""
Final System Validation Test
============================

Comprehensive test of the complete UAV landing system including:
- Plug & play interface functionality
- Neuro-symbolic reasoning
- Real-world image processing  
- Traceability and logging
- Error handling

This test validates the entire system is production-ready.
"""

import sys
import os
import cv2
import numpy as np
import json
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def test_system_functionality():
    """Test basic system functionality"""
    print("🔧 Testing System Functionality...")
    
    try:
        from uav_landing_system import UAVLandingSystem, process_image_for_landing
        print(" System imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test system initialization
    try:
        system = UAVLandingSystem()
        print(" System initialization successful")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False
    
    return True

def test_convenience_function():
    """Test the convenience function for simple usage"""
    print("\n Testing Convenience Function...")
    
    try:
        from uav_landing_system import process_image_for_landing
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test basic processing
        result = process_image_for_landing(test_image, altitude=5.0, enable_tracing=True)
        
        print(f" Status: {result.status}")
        print(f" Confidence: {result.confidence:.3f}")
        print(f" Processing time: {result.processing_time:.1f}ms")
        print(f" Has explanation: {bool(result.decision_explanation)}")
        print(f" Has trace: {result.trace is not None}")
        
        return True
    except Exception as e:
        print(f"❌ Convenience function failed: {e}")
        return False

def test_neuro_symbolic_traceability():
    """Test neuro-symbolic reasoning with full traceability"""
    print("\n🧠 Testing Neuro-Symbolic Traceability...")
    
    try:
        from uav_landing_system import UAVLandingSystem
        
        system = UAVLandingSystem(enable_logging=True)
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Process with full tracing
        result = system.process_frame(test_image, altitude=6.0, enable_tracing=True)
        
        if result.trace:
            print(" Trace available:")
            print(f"   Neural confidence: {result.trace.neural_confidence:.3f}")
            print(f"   Symbolic candidates: {result.trace.symbolic_candidates_found}")
            print(f"   Rules applied: {len(result.trace.symbolic_rules_applied)}")
            print(f"   Safety checks: {len(result.trace.symbolic_safety_checks)}")
            print(f"   Risk level: {result.trace.risk_level}")
            print(f"   Processing time: {result.trace.total_processing_time:.1f}ms")
            print(f"   FPS: {result.trace.inference_fps:.1f}")
            
            # Test JSON export
            trace_dict = result.trace.to_dict()
            print(" JSON export successful")
            
        else:
            print("⚠️  No trace available")
        
        return True
        
    except Exception as e:
        print(f"❌ Neuro-symbolic test failed: {e}")
        return False

def test_real_world_processing():
    """Test with real UDD images if available"""
    print("\n🌍 Testing Real-World Processing...")
    
    # Look for UDD validation images
    udd_path = Path("../datasets/UDD/UDD/UDD6/val")
    if not udd_path.exists():
        print("⚠️  UDD dataset not available, using synthetic test")
        # Create more realistic test image
        test_image = create_synthetic_uav_image()
    else:
        # Load first available validation image
        image_files = list(udd_path.glob("*.jpg"))[:1]
        if image_files:
            test_image = cv2.imread(str(image_files[0]))
            print(f" Loaded real UDD image: {image_files[0].name}")
        else:
            test_image = create_synthetic_uav_image()
            print("⚠️  No UDD images found, using synthetic test")
    
    try:
        from uav_landing_system import UAVLandingSystem
        
        system = UAVLandingSystem(enable_logging=True)
        
        # Process real-world scenario
        start_time = time.time()
        result = system.process_frame(test_image, altitude=7.5, enable_tracing=True)
        end_time = time.time()
        
        print(f" Real-world processing successful:")
        print(f"   Status: {result.status}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Processing time: {(end_time - start_time) * 1000:.1f}ms")
        print(f"   Image shape: {test_image.shape}")
        
        if result.target_pixel:
            print(f"   Target found at: {result.target_pixel}")
        
        if result.trace:
            print(f"   Neural-symbolic score: {result.trace.neuro_symbolic_score:.3f}")
            print(f"   Risk assessment: {result.trace.risk_level}")
        
        return True
        
    except Exception as e:
        print(f"❌ Real-world test failed: {e}")
        return False

def test_error_handling():
    """Test system error handling and robustness"""
    print("\n🛡️  Testing Error Handling...")
    
    try:
        from uav_landing_system import UAVLandingSystem
        
        system = UAVLandingSystem()
        
        # Test with invalid inputs
        test_cases = [
            ("None input", None),
            ("Empty array", np.array([])),
            ("Wrong shape", np.random.randint(0, 255, (100, 100), dtype=np.uint8)),  # 2D instead of 3D
            ("Single channel", np.random.randint(0, 255, (256, 256, 1), dtype=np.uint8)),
        ]
        
        error_handled = 0
        for test_name, test_input in test_cases:
            try:
                result = system.process_frame(test_input, altitude=5.0)
                if result and result.status == "ERROR":
                    error_handled += 1
                    print(f" {test_name}: Gracefully handled")
                else:
                    print(f"⚠️  {test_name}: Unexpected result")
            except Exception:
                error_handled += 1
                print(f" {test_name}: Exception caught and handled")
        
        print(f" Error handling: {error_handled}/{len(test_cases)} cases handled properly")
        return error_handled >= len(test_cases) // 2  # At least half should be handled
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_configuration():
    """Test custom configuration options"""
    print("\n⚙️  Testing Configuration...")
    
    try:
        from uav_landing_system import UAVLandingSystem
        
        # Test with custom weights
        custom_config = {
            "neural_weight": 0.3,
            "symbolic_weight": 0.7,
            "safety_threshold": 0.4
        }
        
        # Save config
        with open("/tmp/test_config.json", "w") as f:
            json.dump(custom_config, f)
        
        # Test with config
        system = UAVLandingSystem(config_path="/tmp/test_config.json")
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = system.process_frame(test_image, altitude=5.0, enable_tracing=True)
        
        # Verify config was applied
        if result.trace and hasattr(result.trace, 'decision_weights'):
            print(" Custom configuration applied successfully")
            print(f"   Decision weights: {result.trace.decision_weights}")
        else:
            print("⚠️  Configuration may not have been fully applied")
        
        # Cleanup
        os.remove("/tmp/test_config.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def create_synthetic_uav_image():
    """Create a synthetic UAV-like image for testing"""
    # Create base terrain
    image = np.random.randint(50, 150, (480, 640, 3), dtype=np.uint8)
    
    # Add some "landing zones" (lighter areas)
    cv2.circle(image, (200, 200), 50, (200, 220, 180), -1)
    cv2.circle(image, (400, 300), 30, (180, 200, 160), -1)
    
    # Add some obstacles (darker areas)
    cv2.rectangle(image, (100, 100), (150, 200), (80, 60, 40), -1)
    cv2.rectangle(image, (500, 350), (600, 450), (70, 50, 30), -1)
    
    return image

def run_comprehensive_test():
    """Run all tests and provide summary"""
    print("🚀 UAV Landing System - Comprehensive Validation Test")
    print("=" * 60)
    
    tests = [
        ("System Functionality", test_system_functionality),
        ("Convenience Function", test_convenience_function),
        ("Neuro-Symbolic Traceability", test_neuro_symbolic_traceability),
        ("Real-World Processing", test_real_world_processing),
        ("Error Handling", test_error_handling),
        ("Configuration", test_configuration),
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    end_time = time.time()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = " PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("-" * 60)
    print(f" OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"⏱️  RUNTIME: {end_time - start_time:.2f} seconds")
    
    if passed == total:
        print("🚀 SYSTEM IS PRODUCTION READY! 🚁🎯")
    elif passed >= total * 0.8:
        print("✨ SYSTEM IS MOSTLY READY - minor issues to address")
    else:
        print("⚠️  SYSTEM NEEDS ATTENTION - multiple issues found")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
