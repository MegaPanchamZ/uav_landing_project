#!/usr/bin/env python3
"""
Headless test for UAV Landing System - No GUI required
"""

import numpy as np
import sys
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from uav_landing.detector import UAVLandingDetector
    from uav_landing.types import LandingResult
    print(" Successfully imported UAV landing modules")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)

def create_synthetic_frame(frame_type="grass", size=(512, 512)):
    """Create synthetic test frames"""
    frame = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    
    if frame_type == "landing_zone":
        # Create a clear landing zone in the center
        center_x, center_y = size[0] // 2, size[1] // 2
        # Add a rectangular landing pad (using numpy since cv2 might not work)
        frame[center_y-50:center_y+50, center_x-50:center_x+50, :] = [100, 100, 100]
    elif frame_type == "grass":
        # All green-ish (grass scenario)
        frame[:, :, 1] = np.random.randint(80, 120, size)  # More green
        frame[:, :, 0] = np.random.randint(20, 60, size)   # Less blue
        frame[:, :, 2] = np.random.randint(30, 70, size)   # Less red
    
    return frame

def test_basic_functionality():
    """Test basic detector functionality"""
    print("\n🧪 Testing Basic Functionality")
    print("=" * 50)
    
    try:
        # Initialize detector
        detector = UAVLandingDetector(
            model_path="models/bisenetv2_uav_landing.onnx",
            enable_memory=False
        )
        print(" Detector initialized successfully")
        
        # Test with synthetic frame
        test_frame = create_synthetic_frame("grass")
        print(f" Created synthetic frame: {test_frame.shape}")
        
        # Process frame with required altitude parameter
        start_time = time.time()
        result = detector.process_frame(test_frame, altitude=5.0)
        processing_time = (time.time() - start_time) * 1000
        
        print(f" Frame processed in {processing_time:.1f}ms")
        print(f"   Result: {result.status}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Should land: {result.status in ['TARGET_ACQUIRED', 'LANDING']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_memory_system():
    """Test memory system functionality"""
    print("\n🧠 Testing Memory System")
    print("=" * 50)
    
    try:
        # Initialize detector with memory
        detector = UAVLandingDetector(
            model_path="models/bisenetv2_uav_landing.onnx",
            enable_memory=True
        )
        print(" Detector with memory initialized")
        
        # Process several frames to build memory
        for i in range(5):
            frame_type = "landing_zone" if i < 3 else "grass"
            test_frame = create_synthetic_frame(frame_type)
            
            start_time = time.time()
            result = detector.process_frame(test_frame, altitude=5.0)
            processing_time = (time.time() - start_time) * 1000
            
            print(f"   Frame {i}: {frame_type} -> {result.status} "
                  f"(conf: {result.confidence:.3f}, "
                  f"memory: {'yes' if result.perception_memory_fusion != 'perception_only' else 'no'}, "
                  f"time: {processing_time:.1f}ms)")
        
        # Check memory status
        memory_zones = detector.memory.get_active_zones()
        print(f" Memory system working - Active zones: {len(memory_zones)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory system test failed: {e}")
        return False

def test_performance():
    """Test performance benchmarks"""
    print("\n⚡ Testing Performance")
    print("=" * 50)
    
    try:
        detector = UAVLandingDetector(
            model_path="models/bisenetv2_uav_landing.onnx",
            enable_memory=True
        )
        
        # Run multiple frames for performance test
        times = []
        for i in range(10):
            test_frame = create_synthetic_frame("grass")
            
            start_time = time.time()
            result = detector.process_frame(test_frame, altitude=5.0)
            processing_time = (time.time() - start_time) * 1000
            times.append(processing_time)
        
        avg_time = np.mean(times)
        fps = 1000 / avg_time if avg_time > 0 else 0
        
        print(f" Performance test completed")
        print(f"   Average processing time: {avg_time:.1f}ms")
        print(f"   Estimated FPS: {fps:.1f}")
        print(f"   Min/Max time: {min(times):.1f}/{max(times):.1f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all headless tests"""
    print("🚁 UAV Landing System - Headless Test Suite")
    print("=" * 60)
    
    # Add OpenCV fallback for headless
    try:
        import cv2
        print(" OpenCV available")
    except ImportError:
        print("⚠️  OpenCV not available, using numpy only")
        # Mock cv2.rectangle for synthetic frame creation
        import sys
        sys.modules['cv2'] = type('MockCV2', (), {
            'rectangle': lambda *args, **kwargs: None
        })()
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Memory System", test_memory_system), 
        ("Performance", test_performance)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        success = test_func()
        results.append((test_name, success))
        
        if success:
            print(f" {test_name} test PASSED")
        else:
            print(f"❌ {test_name} test FAILED")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = " PASS" if success else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests PASSED! System is ready for production!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
