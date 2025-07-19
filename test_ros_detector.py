#!/usr/bin/env python3
"""
Test script for ROS Landing Detector without webcam.
Tests the complete pipeline with synthetic images.
"""

import cv2
import numpy as np
from ros_landing_detector import ROSLandingDetector
import time

def generate_test_image(size=(640, 480)):
    """Generate a synthetic aerial image for testing."""
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create ground base
    base_color = [80, 120, 60]  # Greenish ground
    image[:] = base_color
    
    # Add texture
    noise = np.random.randint(-15, 15, (height, width, 3))
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add landing zone
    center_x, center_y = width // 2, height // 2
    zone_w, zone_h = 100, 80
    
    x1 = center_x - zone_w // 2
    y1 = center_y - zone_h // 2
    x2 = x1 + zone_w
    y2 = y1 + zone_h
    
    # Clear landing zone
    cv2.rectangle(image, (x1, y1), (x2, y2), [120, 140, 100], -1)
    
    # Add some obstacles
    cv2.rectangle(image, (50, 50), (120, 150), [60, 60, 60], -1)  # Building
    cv2.circle(image, (width - 100, 100), 40, [40, 80, 40], -1)   # Tree
    
    return image

def main():
    print("🧪 Testing ROS Landing Detector (No Webcam)")
    print("=" * 50)
    
    # Initialize detector
    try:
        detector = ROSLandingDetector(
            model_path="models/bisenetv2_udd6_final.onnx",  # Will use placeholder if not found
            enable_visualization=True,
            safety_mode=True
        )
        print("✅ ROSLandingDetector initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return False
        
    # Test with multiple synthetic images
    num_tests = 10
    successful_tests = 0
    total_processing_time = 0
    
    print(f"\n🎯 Running {num_tests} test iterations...")
    
    for i in range(num_tests):
        try:
            # Generate test image
            test_image = generate_test_image()
            
            # Simulate varying altitude
            altitude = 5.0 + np.random.uniform(-2.0, 3.0)
            
            # Process image
            start_time = time.time()
            result = detector.process_frame(test_image, altitude=altitude)
            processing_time = (time.time() - start_time) * 1000
            
            total_processing_time += processing_time
            
            # Validate result
            if hasattr(result, 'status') and result.status in ['TARGET_ACQUIRED', 'NO_TARGET', 'UNSAFE']:
                successful_tests += 1
                
                print(f"Test {i+1:2d}: {result.status:15s} | "
                      f"Conf: {result.confidence:.3f} | "
                      f"Time: {processing_time:5.1f}ms", end="")
                
                if result.status == 'TARGET_ACQUIRED' and result.distance_to_target:
                    print(f" | Dist: {result.distance_to_target:.1f}m")
                else:
                    print()
                    
            else:
                print(f"Test {i+1:2d}: ❌ Invalid result: {result}")
                
        except Exception as e:
            print(f"Test {i+1:2d}: ❌ Exception: {e}")
            
    # Performance summary
    print(f"\n📊 Test Results:")
    print(f"   Success rate: {successful_tests}/{num_tests} ({successful_tests/num_tests*100:.1f}%)")
    print(f"   Avg processing time: {total_processing_time/num_tests:.1f}ms")
    print(f"   Processing rate: {1000/(total_processing_time/num_tests):.1f} FPS")
    
    # Get detector performance stats
    stats = detector.get_performance_stats()
    print(f"   Detector stats: {stats['frame_rate']:.1f} FPS over {stats.get('total_frames', 'N/A')} frames")
    print(f"   Current phase: {stats['current_phase']}")
    
    # Test different scenarios
    print(f"\n🎮 Testing Different Scenarios:")
    
    scenarios = [
        ("High altitude", generate_test_image(), 15.0),
        ("Low altitude", generate_test_image(), 2.0),
        ("Very low altitude", generate_test_image(), 0.8),
    ]
    
    for scenario_name, image, alt in scenarios:
        result = detector.process_frame(image, altitude=alt)
        print(f"   {scenario_name:15s} ({alt:4.1f}m): {result.status} "
              f"(conf: {result.confidence:.3f}, phase: {detector.current_phase})")
        
    # Test state reset
    print(f"\n🔄 Testing State Reset:")
    print(f"   Before reset: Phase={detector.current_phase}, Frames={detector.target_lock_frames}")
    detector.reset_state()
    print(f"   After reset:  Phase={detector.current_phase}, Frames={detector.target_lock_frames}")
    
    # Final status
    success_rate = successful_tests / num_tests
    if success_rate >= 0.8:
        print(f"\n✅ ROS Landing Detector test PASSED ({success_rate*100:.1f}% success)")
        return True
    else:
        print(f"\n❌ ROS Landing Detector test FAILED ({success_rate*100:.1f}% success)")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
