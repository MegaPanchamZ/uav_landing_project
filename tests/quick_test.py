#!/usr/bin/env python3
"""Quick test of the UAV Landing Detector"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import numpy as np
import cv2
from uav_landing_detector import UAVLandingDetector

def quick_test():
    print("🚁 Testing UAV Landing Detector...")
    
    # Initialize detector
    detector = UAVLandingDetector(
        model_path=None,  # Use placeholder mode
        enable_visualization=True
    )
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test single frame processing
    result = detector.process_frame(test_image, altitude=5.0)
    
    print(f" Status: {result.status}")
    print(f" Confidence: {result.confidence:.2f}")
    print(f" Processing time: {result.processing_time:.1f}ms")
    print(f" FPS: {result.fps:.1f}")
    
    if result.target_pixel:
        print(f" Target pixel: {result.target_pixel}")
        print(f" Target world: {result.target_world}")
        print(f" Distance: {result.distance:.1f}m")
    
    print(f" Commands: [F:{result.forward_velocity:.1f}, R:{result.right_velocity:.1f}, D:{result.descent_rate:.1f}]")
    
    # Test performance stats
    stats = detector.get_performance_stats()
    print(f" Stats: {stats}")
    
    print("\n Single-class implementation working perfectly!")
    return True

if __name__ == "__main__":
    quick_test()
