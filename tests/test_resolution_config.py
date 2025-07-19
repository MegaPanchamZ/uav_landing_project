#!/usr/bin/env python3
"""
Resolution Configuration Test
Test different input resolutions and compare performance vs quality
"""

import sys
import cv2
import numpy as np
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from uav_landing_system import UAVLandingSystem, process_image_for_landing

def create_test_image():
    """Create a realistic UAV landing zone test image"""
    # Create base terrain (brownish)
    image = np.random.randint(80, 120, (600, 800, 3), dtype=np.uint8)
    
    # Add green vegetation areas
    cv2.circle(image, (200, 150), 80, (60, 120, 50), -1)
    cv2.circle(image, (600, 400), 60, (70, 130, 60), -1)
    
    # Add suitable landing zones (lighter, flatter areas)
    cv2.ellipse(image, (400, 300), (120, 80), 0, 0, 360, (150, 160, 140), -1)
    cv2.ellipse(image, (150, 450), (90, 90), 0, 0, 360, (140, 150, 130), -1)
    
    # Add obstacles (darker areas)
    cv2.rectangle(image, (300, 100), (380, 180), (40, 30, 20), -1)  # Building
    cv2.rectangle(image, (500, 450), (580, 520), (30, 25, 15), -1)  # Building
    
    # Add some texture and noise for realism
    noise = np.random.normal(0, 10, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image

def test_resolutions():
    """Test different input resolutions and compare results"""
    
    print("🔍 UAV Landing System - Resolution Configuration Test")
    print("=" * 60)
    
    # Create test image
    test_image = create_test_image()
    print(f"📸 Created test image: {test_image.shape}")
    
    # Test different resolutions
    resolutions = [
        (256, 256, "Ultra-Fast", "Racing/Real-time"),
        (512, 512, "Balanced", "General/Commercial"), 
        (768, 768, "High-Quality", "Precision/Mapping"),
        (1024, 1024, "Maximum", "Research/Analysis")
    ]
    
    results = []
    
    print("\n🚀 Testing Different Resolutions:")
    print("-" * 60)
    
    for width, height, quality_name, use_case in resolutions:
        print(f"\n🔧 Testing {width}×{height} ({quality_name}) - {use_case}")
        
        try:
            # Time the processing
            start_time = time.time()
            
            # Test using convenience function
            result = process_image_for_landing(
                test_image, 
                altitude=5.0,
                input_resolution=(width, height),
                enable_tracing=True
            )
            
            end_time = time.time()
            total_time = (end_time - start_time) * 1000  # Convert to ms
            
            # Calculate FPS
            fps = 1000 / total_time if total_time > 0 else 0
            
            print(f"   ✅ Status: {result.status}")
            print(f"   📊 Confidence: {result.confidence:.3f}")
            print(f"   ⏱️  Processing Time: {total_time:.1f}ms")
            print(f"   🎯 Estimated FPS: {fps:.1f}")
            
            if result.target_pixel:
                print(f"   📍 Target Location: {result.target_pixel}")
            
            if result.trace:
                print(f"   🧠 Neural Confidence: {result.trace.neural_confidence:.3f}")
                print(f"   🔬 Symbolic Candidates: {result.trace.symbolic_candidates_found}")
                print(f"   ⚠️  Risk Level: {result.trace.risk_level}")
            
            results.append({
                'resolution': f"{width}×{height}",
                'quality': quality_name,
                'use_case': use_case,
                'time_ms': total_time,
                'fps': fps,
                'status': result.status,
                'confidence': result.confidence,
                'target_found': result.target_pixel is not None
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'resolution': f"{width}×{height}",
                'quality': quality_name,
                'use_case': use_case,
                'time_ms': float('inf'),
                'fps': 0,
                'status': 'ERROR',
                'confidence': 0.0,
                'target_found': False
            })
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("📊 RESOLUTION COMPARISON SUMMARY")
    print("=" * 60)
    
    print(f"{'Resolution':<12} {'Quality':<12} {'Time(ms)':<10} {'FPS':<8} {'Status':<15} {'Confidence':<12}")
    print("-" * 70)
    
    for result in results:
        time_str = f"{result['time_ms']:.1f}" if result['time_ms'] != float('inf') else "FAIL"
        fps_str = f"{result['fps']:.1f}" if result['fps'] > 0 else "0"
        
        print(f"{result['resolution']:<12} {result['quality']:<12} {time_str:<10} {fps_str:<8} "
              f"{result['status']:<15} {result['confidence']:<12.3f}")
    
    # Recommendations
    print("\n🎯 RECOMMENDATIONS BY USE CASE:")
    print("-" * 40)
    
    recommendations = {
        "🏎️ Racing/Real-time": "256×256 (Ultra-Fast) for maximum speed",
        "🏢 Commercial UAV": "512×512 (Balanced) for optimal quality/speed", 
        "🔬 Research/Mapping": "768×768 (High-Quality) for detailed analysis",
        "🎓 Development/Testing": "512×512 (Balanced) for general development"
    }
    
    for use_case, recommendation in recommendations.items():
        print(f"{use_case}: {recommendation}")
    
    # Performance insights
    print("\n💡 PERFORMANCE INSIGHTS:")
    print("-" * 30)
    
    if len([r for r in results if r['fps'] > 0]) >= 2:
        fastest = max(results, key=lambda x: x['fps'] if x['fps'] > 0 else 0)
        slowest = min(results, key=lambda x: x['fps'] if x['fps'] > 0 else float('inf'))
        
        if fastest['fps'] > 0 and slowest['fps'] > 0:
            speed_diff = fastest['fps'] / slowest['fps']
            print(f"• Speed difference: {speed_diff:.1f}x faster ({fastest['resolution']} vs {slowest['resolution']})")
        
        high_conf_results = [r for r in results if r['confidence'] > 0.5]
        if high_conf_results:
            best_quality = max(high_conf_results, key=lambda x: x['confidence'])
            print(f"• Best confidence: {best_quality['confidence']:.3f} at {best_quality['resolution']}")
    
    print(f"\n✅ Test completed! {len([r for r in results if r['status'] != 'ERROR'])}/{len(results)} resolutions working")
    
    return results

def test_dynamic_resolution():
    """Test dynamic resolution switching based on conditions"""
    
    print("\n🔄 Dynamic Resolution Test")
    print("=" * 30)
    
    test_image = create_test_image()
    
    # Test different scenarios
    scenarios = [
        {"name": "High-speed flight", "altitude": 10.0, "velocity": 8.0, "recommended": (256, 256)},
        {"name": "Normal flight", "altitude": 5.0, "velocity": 3.0, "recommended": (512, 512)},
        {"name": "Precision landing", "altitude": 1.5, "velocity": 0.5, "recommended": (768, 768)},
        {"name": "Research analysis", "altitude": 8.0, "velocity": 0.0, "recommended": (1024, 1024)}
    ]
    
    for scenario in scenarios:
        print(f"\n📍 Scenario: {scenario['name']}")
        print(f"   Altitude: {scenario['altitude']}m, Velocity: {scenario['velocity']}m/s")
        
        # Select resolution based on scenario
        resolution = scenario['recommended']
        
        try:
            result = process_image_for_landing(
                test_image,
                altitude=scenario['altitude'],
                input_resolution=resolution,
                enable_tracing=True
            )
            
            print(f"   🔍 Using resolution: {resolution}")
            print(f"   📊 Result: {result.status} (confidence: {result.confidence:.3f})")
            print(f"   ⏱️  Processing time: {result.processing_time:.1f}ms")
            
        except Exception as e:
            print(f"   ❌ Error in scenario: {e}")

if __name__ == "__main__":
    print("🚁 Starting UAV Landing System Resolution Test")
    
    # Main resolution test
    results = test_resolutions()
    
    # Dynamic resolution test
    test_dynamic_resolution()
    
    print("\n🎯 Test Complete! Check the results above to choose your optimal resolution.")
    print("📖 For more details, see: docs/RESOLUTION_UPGRADE_GUIDE.md")
