#!/usr/bin/env python3
"""
Demo script for the Enhanced UAV Landing System with Scallop Integration

This script demonstrates the complete working system with neuro-symbolic reasoning.
"""

import numpy as np
import cv2
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from enhanced_uav_detector import EnhancedUAVDetector

def create_synthetic_image():
    """Create a synthetic aerial view image for testing"""
    
    # Create a 512x512 image
    image = np.ones((512, 512, 3), dtype=np.uint8) * 100  # Gray background
    
    # Add grass areas (good landing zones)
    cv2.rectangle(image, (100, 100), (200, 200), (50, 150, 50), -1)  # Large grass area
    cv2.rectangle(image, (300, 300), (400, 400), (60, 140, 60), -1)  # Another grass area
    
    # Add obstacles
    cv2.rectangle(image, (250, 250), (280, 280), (120, 120, 120), -1)  # Building
    cv2.circle(image, (150, 350), 20, (100, 100, 100), -1)  # Tree
    
    # Add some texture/noise
    noise = np.random.randint(-30, 30, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image

def demo_context_switching(detector, test_image):
    """Demo context switching capabilities"""
    
    print("\n🔄 Testing Context Switching:")
    contexts = ["commercial", "emergency", "precision", "delivery"]
    
    for context in contexts:
        print(f"\n📋 Context: {context.upper()}")
        detector.set_context(context)
        
        result = detector.process_frame(test_image, altitude=10.0)
        print(f"   Status: {result.status}")
        print(f"   Confidence: {result.confidence:.3f}")
        if hasattr(result, 'processing_time'):
            print(f"   Processing Time: {result.processing_time:.1f}ms")

def demo_performance_analysis(detector, test_image):
    """Demo performance analysis"""
    
    print("\n📊 Performance Analysis:")
    processing_times = []
    
    # Process multiple frames
    for i in range(10):
        result = detector.process_frame(test_image, altitude=10.0 + i)
        if hasattr(result, 'processing_time'):
            processing_times.append(result.processing_time)
    
    if processing_times:
        avg_time = np.mean(processing_times)
        min_time = np.min(processing_times)
        max_time = np.max(processing_times)
        
        print(f"   Average Processing Time: {avg_time:.1f}ms")
        print(f"   Min/Max Processing Time: {min_time:.1f}/{max_time:.1f}ms")
        print(f"   Estimated FPS: {1000/avg_time:.1f}")
    
    # Get detailed performance stats
    stats = detector.get_enhanced_performance_stats()
    print(f"   Scallop Available: {stats.get('scallop_available', 'Unknown')}")
    print(f"   Using Scallop: {stats.get('use_scallop', 'Unknown')}")
    if 'scallop_reasoning_count' in stats:
        print(f"   Scallop Reasoning Count: {stats['scallop_reasoning_count']}")

def demo_reasoning_explanation(detector):
    """Demo reasoning explanation capabilities"""
    
    print("\n🧠 Reasoning Explanation:")
    explanation = detector.get_reasoning_explanation()
    
    print(f"   Reasoning Engine: {explanation.get('reasoning_engine', 'Unknown')}")
    print(f"   Context: {explanation.get('context', 'Unknown')}")
    print(f"   Scallop Available: {explanation.get('scallop_available', 'Unknown')}")
    print(f"   Fallback Count: {explanation.get('fallback_count', 0)}")

def main():
    """Main demo function"""
    
    print("🚁 Enhanced UAV Landing System Demo")
    print("=" * 50)
    
    try:
        # Test Scallop availability
        print("\n🔧 Testing Scallop Installation:")
        try:
            import scallopy
            print("    Scallop imported successfully")
            
            # Test basic Scallop functionality  
            ctx = scallopy.Context()
            ctx.add_program("type test(i32, i32)")
            ctx.add_facts("test", [(1, 2)])
            ctx.run()
            result = list(ctx.relation("test"))
            if result == [(1, 2)]:
                print("    Scallop basic functionality verified")
            else:
                print("   ⚠️  Scallop functionality issue")
                
        except ImportError as e:
            print(f"   ❌ Scallop import failed: {e}")
            print("   📝 System will use mock implementation")
        
        # Create enhanced detector
        print("\n🤖 Initializing Enhanced UAV Detector:")
        detector = EnhancedUAVDetector(
            context="commercial",
            use_scallop=True,
            enable_visualization=False
        )
        print("    Enhanced detector initialized")
        
        # Create test image
        print("\n🖼️  Creating synthetic test image...")
        test_image = create_synthetic_image()
        print("    Test image created (512x512)")
        
        # Basic detection test
        print("\n Basic Detection Test:")
        result = detector.process_frame(test_image, altitude=15.0)
        
        print(f"   Status: {result.status}")
        print(f"   Confidence: {result.confidence:.3f}")
        
        if hasattr(result, 'target_pixel') and result.target_pixel:
            print(f"   Target Pixel: {result.target_pixel}")
            
        if hasattr(result, 'target_world') and result.target_world:
            print(f"   Target World: {result.target_world}")
            
        if hasattr(result, 'distance') and result.distance:
            print(f"   Distance: {result.distance:.2f}m")
            
        if hasattr(result, 'processing_time'):
            print(f"   Processing Time: {result.processing_time:.1f}ms")
        
        # Context switching demo
        demo_context_switching(detector, test_image)
        
        # Performance analysis
        demo_performance_analysis(detector, test_image)
        
        # Reasoning explanation
        demo_reasoning_explanation(detector)
        
        # Test with different altitudes
        print("\n🛫 Altitude Sensitivity Test:")
        altitudes = [5.0, 10.0, 15.0, 20.0, 25.0]
        
        for alt in altitudes:
            result = detector.process_frame(test_image, altitude=alt)
            print(f"   Altitude {alt:4.1f}m: {result.status} (conf: {result.confidence:.3f})")
        
        print("\n Demo completed successfully!")
        print("\n📋 System Summary:")
        print(f"   - Scallop Integration: {'✅' if hasattr(detector, 'scallop_available') and detector.scallop_available else '⚠️  (Mock)'}")
        print(f"   - Context Awareness: ✅")
        print(f"   - Performance Tracking: ✅")
        print(f"   - Reasoning Explanation: ✅")
        print(f"   - Neuro-Symbolic Architecture: ✅")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
