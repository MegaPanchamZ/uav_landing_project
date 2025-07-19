#!/usr/bin/env python3
"""
Simple debug test for ROS Landing Detector
"""

import traceback
import numpy as np
from ros_landing_detector import ROSLandingDetector

def generate_simple_image():
    """Generate minimal test image."""
    return np.zeros((480, 640, 3), dtype=np.uint8)

def main():
    print("🔍 Debug test for ROS Landing Detector")
    
    try:
        print("1. Initializing detector...")
        detector = ROSLandingDetector()
        print("   ✅ Detector initialized")
        
        print("2. Generating test image...")
        test_image = generate_simple_image()
        print(f"   ✅ Image shape: {test_image.shape}")
        
        print("3. Processing frame...")
        # Get the neural engine result first 
        neural_result = detector.neural_engine.process_frame(test_image)
        print(f"   Neural result: {neural_result}")
        
        # Then try symbolic processing
        if neural_result.get('success', False):
            mask = neural_result.get('mask', np.zeros((test_image.shape[0], test_image.shape[1]), dtype=np.uint8))
            print(f"   Mask shape: {mask.shape}, dtype: {mask.dtype}")
            
            symbolic_result = detector.symbolic_engine.run(mask)
            print(f"   Symbolic result: {symbolic_result}")
        
        result = detector.process_frame(test_image, altitude=5.0)
        print(f"   ✅ Result: {result.status}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
