#!/usr/bin/env python3
"""Test UAV Landing Detector with real ONNX model"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import numpy as np
import cv2
from uav_landing_detector import UAVLandingDetector

def test_real_model():
    print("🚁 Testing UAV Landing Detector with REAL ONNX model...")
    
    # Initialize detector with real model
    detector = UAVLandingDetector(
        model_path="models/bisenetv2_uav_landing.onnx",  # Real ONNX model
        enable_visualization=True
    )
    
    # Create test image (more realistic)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some ground texture
    test_image[:, :] = [120, 140, 100]  # Grass-like color
    
    # Add a landing area
    cv2.rectangle(test_image, (280, 200), (360, 280), (100, 150, 100), -1)
    
    # Add some obstacles  
    cv2.rectangle(test_image, (100, 100), (180, 200), (80, 80, 80), -1)  # Building
    cv2.circle(test_image, (500, 150), 40, (60, 80, 60), -1)  # Tree
    
    # Test processing with different altitudes
    altitudes = [8.0, 4.0, 2.0, 1.0]
    
    for altitude in altitudes:
        result = detector.process_frame(test_image, altitude=altitude)
        
        print(f"\n📏 Altitude: {altitude:.1f}m")
        print(f"   Status: {result.status}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Processing time: {result.processing_time:.1f}ms")
        print(f"   FPS: {result.fps:.1f}")
        print(f"   Landing phase: {detector.landing_phase}")
        
        if result.target_pixel:
            print(f"   Target: pixel={result.target_pixel}, world={result.target_world}")
            print(f"   Commands: F={result.forward_velocity:.2f}, R={result.right_velocity:.2f}, D={result.descent_rate:.2f}")
    
    print("\n Real ONNX model test completed!")
    print(f"📊 Final stats: {detector.get_performance_stats()}")
    
    # Save a test visualization if available
    if result.annotated_image is not None:
        cv2.imwrite("test_output.jpg", result.annotated_image)
        print("💾 Saved visualization: test_output.jpg")
    
    return True

if __name__ == "__main__":
    test_real_model()
