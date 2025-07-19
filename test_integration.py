#!/usr/bin/env python3
"""
Simple test to verify the fine-tuned model integration
"""

import sys
from pathlib import Path
sys.path.append('.')

from uav_landing_detector import UAVLandingDetector
import cv2
import numpy as np

def test_fine_tuned_model():
    """Test the integrated fine-tuned model."""
    
    print("🔬 Testing Fine-Tuned Model Integration")
    print("=" * 40)
    
    # Initialize detector with explicit model path
    model_path = "models/bisenetv2_uav_landing.onnx"
    
    try:
        detector = UAVLandingDetector(model_path=model_path, enable_viz=False)
        print("✅ Detector initialized successfully")
        
        # Test with a sample image
        test_image_path = "../datasets/drone_deploy_dataset_intermediate/dataset-medium/images/107f24d6e9_F1BE1D4184INSPIRE-ortho.tif"
        
        if not Path(test_image_path).exists():
            print(f"⚠️  Test image not found: {test_image_path}")
            print("Creating synthetic test image...")
            
            # Create synthetic test image
            test_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            
            # Add some "suitable" areas (green-ish patches)
            test_img[100:200, 100:200] = [50, 150, 50]  # Green-ish area
            test_img[300:400, 300:400] = [40, 140, 40]  # Another green area
            
            result = detector.detect_landing_zones(test_img)
            
        else:
            print(f"📸 Testing with: {test_image_path}")
            result = detector.detect_landing_zones(test_image_path)
        
        # Analyze results
        if result:
            print("\\n🎯 Detection Results:")
            print(f"   Landing zones found: {len(result.get('landing_zones', []))}")
            print(f"   Safety score: {result.get('safety_score', 0):.1f}%")
            print(f"   Recommendation: {result.get('recommendation', 'Unknown')}")
            
            if result.get('landing_zones'):
                for i, zone in enumerate(result['landing_zones'], 1):
                    print(f"   Zone {i}: {zone.get('area', 0):.0f} pixels, "
                          f"confidence: {zone.get('confidence', 0):.2f}")
        else:
            print("❌ No results returned")
            
        print("\\n✅ Fine-tuned model test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fine_tuned_model()
