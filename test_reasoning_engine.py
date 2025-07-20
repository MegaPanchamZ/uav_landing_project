#!/usr/bin/env python3
"""
Simple test to validate our Scallop Reasoning Engine directly
"""

import numpy as np
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from scallop_reasoning_engine import ScallopReasoningEngine

def test_scallop_reasoning_engine():
    """Test the Scallop reasoning engine directly"""
    
    print("🧠 Testing Scallop Reasoning Engine")
    print("=" * 40)
    
    try:
        # Initialize reasoning engine
        print("\n📋 Initializing reasoning engine...")
        engine = ScallopReasoningEngine(context="commercial")
        print(f"   ✅ Engine initialized")
        print(f"   ✅ Context: {engine.context}")
        print(f"   ✅ Scallop Available: {engine.scallop_available}")
        
        # Create synthetic data
        print("\n🔧 Creating synthetic data...")
        seg_output = np.zeros((256, 256), dtype=np.uint8)
        seg_output[64:192, 64:192] = 0  # Large grass area (class 0)
        seg_output[100:116, 100:116] = 1  # Small building (class 1) 
        
        confidence_map = np.random.uniform(0.7, 0.9, (256, 256)).astype(np.float32)
        image_shape = (256, 256)
        altitude = 10.0
        
        print(f"   ✅ Segmentation shape: {seg_output.shape}")
        print(f"   ✅ Confidence shape: {confidence_map.shape}")
        print(f"   ✅ Altitude: {altitude}m")
        
        # Test reasoning
        print("\n🎯 Running reasoning...")
        result = engine.reason(
            segmentation_output=seg_output,
            confidence_map=confidence_map,
            image_shape=image_shape,
            altitude=altitude
        )
        
        print(f"   ✅ Reasoning complete")
        print(f"   ✅ Status: {result.status}")
        print(f"   ✅ Confidence: {result.confidence:.3f}")
        print(f"   ✅ Context: {result.context}")
        
        if result.target_pixel:
            print(f"   ✅ Target: {result.target_pixel}")
        
        if result.reasoning_trace:
            print(f"   ✅ Reasoning trace length: {len(result.reasoning_trace)}")
        
        # Test different contexts
        print("\n🔄 Testing context switching...")
        contexts = ["emergency", "precision", "delivery"]
        
        for ctx in contexts:
            engine.set_context(ctx)
            result = engine.reason(
                segmentation_output=seg_output,
                confidence_map=confidence_map,
                image_shape=image_shape,
                altitude=altitude
            )
            print(f"   ✅ Context {ctx}: {result.status} (conf: {result.confidence:.3f})")
        
        # Test performance
        print("\n📊 Performance testing...")
        import time
        
        times = []
        for i in range(10):
            start = time.time()
            result = engine.reason(
                segmentation_output=seg_output,
                confidence_map=confidence_map,
                image_shape=image_shape,
                altitude=altitude
            )
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        print(f"   ✅ Average reasoning time: {avg_time:.1f}ms")
        print(f"   ✅ Min/Max times: {np.min(times):.1f}/{np.max(times):.1f}ms")
        
        # Get performance stats
        stats = engine.get_performance_stats()
        print(f"   ✅ Total reasonings: {stats.get('total_reasonings', 0)}")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_scallop_reasoning_engine()
    sys.exit(0 if success else 1)
