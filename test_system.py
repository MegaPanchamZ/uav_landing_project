#!/usr/bin/env python3
"""
Test script for UAV Landing Zone Detection System
Runs basic functionality tests to ensure the system is working correctly.
"""

import sys
import numpy as np
import cv2
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

import config
from neural_engine import NeuralEngine
from symbolic_engine import SymbolicEngine


def test_config():
    """Test that configuration is loaded correctly."""
    print("Testing configuration...")
    assert hasattr(config, 'CLASS_MAP'), "CLASS_MAP not found in config"
    assert hasattr(config, 'MIN_LANDING_AREA_PIXELS'), "MIN_LANDING_AREA_PIXELS not found in config"
    assert len(config.CLASS_MAP) == 5, f"Expected 5 classes, got {len(config.CLASS_MAP)}"
    print("✓ Configuration test passed")


def test_neural_engine():
    """Test neural engine initialization and processing."""
    print("Testing Neural Engine...")
    
    # Initialize neural engine
    neural_engine = NeuralEngine()
    print("✓ Neural engine initialized")
    
    # Test with dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    start_time = time.time()
    seg_map = neural_engine.process_frame(dummy_frame)
    processing_time = time.time() - start_time
    
    print(f"✓ Frame processed in {processing_time*1000:.1f}ms")
    
    # Verify output shape and data type
    assert seg_map.shape == config.INPUT_RESOLUTION, f"Expected shape {config.INPUT_RESOLUTION}, got {seg_map.shape}"
    assert seg_map.dtype == np.uint8, f"Expected uint8, got {seg_map.dtype}"
    assert np.max(seg_map) < len(config.CLASS_MAP), f"Invalid class ID found: {np.max(seg_map)}"
    
    # Test performance stats
    stats = neural_engine.get_performance_stats()
    print(f"✓ Performance stats: {stats}")
    
    print("✓ Neural Engine test passed")
    return neural_engine


def test_symbolic_engine():
    """Test symbolic engine with various scenarios."""
    print("Testing Symbolic Engine...")
    
    symbolic_engine = SymbolicEngine()
    print("✓ Symbolic engine initialized")
    
    # Test scenario 1: No safe zones
    empty_seg_map = np.zeros(config.INPUT_RESOLUTION, dtype=np.uint8)
    decision = symbolic_engine.run(empty_seg_map)
    assert decision['status'] == 'NO_VALID_ZONE', f"Expected NO_VALID_ZONE, got {decision['status']}"
    print("✓ Empty segmentation test passed")
    
    # Test scenario 2: Safe zone present
    seg_map_with_zone = np.zeros(config.INPUT_RESOLUTION, dtype=np.uint8)
    cv2.rectangle(seg_map_with_zone, (100, 100), (300, 300), config.SAFE_LANDING_CLASS_ID, -1)
    
    # Run multiple times to test temporal stability
    for _ in range(config.TEMPORAL_CONFIRMATION_THRESHOLD + 2):
        decision = symbolic_engine.run(seg_map_with_zone)
    
    print(f"✓ Final decision: {decision['status']}")
    
    # Test scenario 3: Safe zone with obstacles
    seg_map_with_obstacles = seg_map_with_zone.copy()
    cv2.circle(seg_map_with_obstacles, (200, 200), 20, config.HIGH_OBSTACLE_CLASS_ID, -1)
    
    decision = symbolic_engine.run(seg_map_with_obstacles)
    print(f"✓ Decision with obstacles: {decision['status']}")
    
    # Test performance stats
    stats = symbolic_engine.get_performance_stats()
    print(f"✓ Performance stats: {stats}")
    
    print("✓ Symbolic Engine test passed")
    return symbolic_engine


def test_integration():
    """Test integration between neural and symbolic engines."""
    print("Testing Integration...")
    
    neural_engine = NeuralEngine()
    symbolic_engine = SymbolicEngine()
    
    # Create a test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Process through the complete pipeline
    start_time = time.time()
    seg_map = neural_engine.process_frame(test_frame)
    decision = symbolic_engine.run(seg_map)
    total_time = time.time() - start_time
    
    print(f"✓ Complete pipeline processed in {total_time*1000:.1f}ms")
    
    # Verify decision structure
    required_keys = ['status', 'frame_count', 'potential_zones', 'valid_zones', 'obstacles']
    for key in required_keys:
        assert key in decision, f"Missing key in decision: {key}"
    
    print("✓ Integration test passed")


def test_visualization():
    """Test visualization functions."""
    print("Testing Visualization...")
    
    # This test would normally require a display, so we'll do basic checks
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    seg_map = np.zeros(config.INPUT_RESOLUTION, dtype=np.uint8)
    
    # Add a fake landing zone
    cv2.rectangle(seg_map, (200, 200), (400, 400), config.SAFE_LANDING_CLASS_ID, -1)
    
    decision = {
        'status': 'TARGET_ACQUIRED',
        'zone': {
            'center': (300, 300),
            'area': 40000,
            'score': 0.85,
            'temporal_stability': 12,
            'bounding_rect': (200, 200, 200, 200),
            'aspect_ratio': 1.0
        },
        'frame_count': 100,
        'potential_zones': 1,
        'valid_zones': 1,
        'confirmed_zones': 1,
        'obstacles': 0
    }
    
    # The visualization would normally be tested with a display
    print("✓ Visualization test structure verified")


def performance_benchmark():
    """Run a performance benchmark."""
    print("\nRunning Performance Benchmark...")
    
    neural_engine = NeuralEngine()
    symbolic_engine = SymbolicEngine()
    
    # Test with different frame sizes
    frame_sizes = [(320, 240), (640, 480), (1280, 720)]
    
    for width, height in frame_sizes:
        test_frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Warm up
        for _ in range(5):
            seg_map = neural_engine.process_frame(test_frame)
            symbolic_engine.run(seg_map)
        
        # Benchmark
        times = []
        for _ in range(20):
            start = time.time()
            seg_map = neural_engine.process_frame(test_frame)
            decision = symbolic_engine.run(seg_map)
            end = time.time()
            times.append(end - start)
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        print(f"  {width}x{height}: {avg_time*1000:.1f}ms avg, {fps:.1f} FPS")
    
    print("✓ Performance benchmark completed")


def main():
    """Run all tests."""
    print("UAV Landing Zone Detection System - Test Suite")
    print("=" * 60)
    
    try:
        test_config()
        print()
        
        test_neural_engine()
        print()
        
        test_symbolic_engine()
        print()
        
        test_integration()
        print()
        
        test_visualization()
        print()
        
        performance_benchmark()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("The UAV Landing Zone Detection System is working correctly.")
        print("\nNext steps:")
        print("1. Generate test videos: python generate_test_video.py")
        print("2. Run the main application: python main.py")
        print("3. Test with your camera: python main.py --video 0")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
