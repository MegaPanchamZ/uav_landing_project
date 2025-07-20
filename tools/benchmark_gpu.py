#!/usr/bin/env python3
"""
GPU Performance Benchmarking for UAV Landing System

Compare TensorRT vs CUDA vs CPU performance
"""

import time
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from uav_landing_detector import UAVLandingDetector

def benchmark_device(device_name, runs=10):
    """Benchmark a specific device"""
    print(f"\n🔧 Testing {device_name}...")
    
    try:
        detector = UAVLandingDetector(device=device_name, input_resolution=(512, 512))
        print(f"   Device: {detector.actual_device}")
        print(f"   Provider: {detector.session.get_providers()[0] if detector.session else 'None'}")
        
        if not detector.session:
            print(f"   ❌ Model not loaded for {device_name}")
            return None, None
            
        # Create test image
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Warmup run
        _ = detector.process_frame(test_image, altitude=10.0)
        
        # Benchmark runs
        times = []
        for i in range(runs):
            start_time = time.time()
            result = detector.process_frame(test_image, altitude=10.0)
            end_time = time.time()
            times.append(end_time - start_time)
            
        avg_time = np.mean(times) * 1000  # Convert to ms
        avg_fps = 1.0 / np.mean(times)
        
        print(f"   📊 Avg time: {avg_time:.1f}ms")
        print(f"   🚀 Avg FPS: {avg_fps:.1f}")
        
        return avg_time, avg_fps
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None, None

def main():
    """Run performance comparison"""
    print("🏎️  UAV Landing System - GPU Performance Benchmark")
    print("=" * 55)
    
    results = {}
    
    # Test all available devices
    devices_to_test = ['cpu', 'cuda', 'tensorrt', 'auto']
    
    for device in devices_to_test:
        avg_time, avg_fps = benchmark_device(device)
        if avg_time is not None:
            results[device] = {'time': avg_time, 'fps': avg_fps}
    
    # Print summary
    print(f"\n📊 Performance Summary:")
    print("=" * 40)
    
    if results:
        sorted_results = sorted(results.items(), key=lambda x: x[1]['fps'], reverse=True)
        
        for i, (device, metrics) in enumerate(sorted_results):
            rank_emoji = ["🏆", "🥈", "🥉"][i] if i < 3 else "📋"
            print(f"{rank_emoji} {device.upper()}: {metrics['fps']:.1f} FPS ({metrics['time']:.1f}ms)")
        
        best_device = sorted_results[0][0]
        best_fps = sorted_results[0][1]['fps']
        
        print(f"\n🎯 Recommendation: Use '{best_device}' for {best_fps:.1f} FPS")
        
        # Performance improvement calculation
        if 'cpu' in results and len(sorted_results) > 1:
            cpu_fps = results['cpu']['fps']
            improvement = (best_fps / cpu_fps - 1) * 100
            print(f"   Performance gain over CPU: +{improvement:.0f}%")
    else:
        print("No devices could be benchmarked.")
    
    print(f"\n🔧 System Status:")
    print(f"   Resolution tested: 512x512")
    print(f"   Benchmark runs: 10 per device")
    print(f"   Includes Scallop reasoning overhead")

if __name__ == "__main__":
    main()
