#!/usr/bin/env python3
"""
Simple Resolution Configuration Test
Test the resolution parameter without model loading
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path  
sys.path.append(str(Path(__file__).parent.parent))

from uav_landing_system import UAVLandingSystem

def test_resolution_parameter():
    """Test that the resolution parameter is accepted and configured correctly"""
    
    print("🔍 Testing Resolution Parameter Configuration")
    print("=" * 50)
    
    resolutions = [
        (256, 256, "Ultra-Fast"),
        (512, 512, "Balanced"),
        (768, 768, "High-Quality"),
        (1024, 1024, "Maximum")
    ]
    
    for width, height, name in resolutions:
        print(f"\n🔧 Testing {width}×{height} ({name})")
        
        try:
            # Initialize with custom resolution
            system = UAVLandingSystem(
                input_resolution=(width, height),
                enable_logging=False
            )
            
            # Check if detector has correct input size
            actual_size = system.detector.input_size
            expected_size = (width, height)
            
            if actual_size == expected_size:
                print(f"    Resolution configured correctly: {actual_size}")
                print(f"   📐 Camera center adjusted to: ({system.detector.cx}, {system.detector.cy})")
            else:
                print(f"   ❌ Resolution mismatch: expected {expected_size}, got {actual_size}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n Resolution Configuration Test Complete!")

if __name__ == "__main__":
    test_resolution_parameter()
