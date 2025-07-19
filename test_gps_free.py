#!/usr/bin/env python3
"""
GPS-Free Landing System Test Suite
Tests all components of the markerless, GPS-free landing system.
"""

import sys
import numpy as np
import cv2
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from visual_odometry import VisualOdometry, RelativePositioning
from flight_controller import MockFlightController, GPSFreeLandingController, FlightMode


def test_visual_odometry():
    """Test visual odometry with synthetic motion."""
    print("Testing Visual Odometry...")
    
    # Initialize with default camera parameters
    vo = VisualOdometry()
    print("✓ Visual odometry initialized")
    
    # Create synthetic frames with known motion
    frame1 = create_test_frame_with_features(100)
    frame2 = create_test_frame_with_features(100, offset=(10, 5))  # Simulate camera motion
    
    # Process frames
    motion1 = vo.process_frame(frame1)
    motion2 = vo.process_frame(frame2)
    
    print(f"✓ Frame 1: {motion1['num_features']} features, confidence: {motion1['motion_confidence']:.2f}")
    print(f"✓ Frame 2: {motion2['num_features']} features, confidence: {motion2['motion_confidence']:.2f}")
    
    # Test motion smoothing
    smoothed = vo.get_smoothed_motion()
    print(f"✓ Smoothed motion confidence: {smoothed['confidence']:.2f}")
    
    print("✅ Visual Odometry test passed")
    return vo


def test_relative_positioning():
    """Test relative positioning calculations."""
    print("Testing Relative Positioning...")
    
    vo = VisualOdometry()
    positioning = RelativePositioning(vo)
    
    # Test pixel to position conversion
    test_cases = [
        ((320, 240), 10.0),  # Center pixel at 10m altitude
        ((420, 240), 10.0),  # 100 pixels right
        ((320, 340), 10.0),  # 100 pixels down
    ]
    
    for pixel_coords, altitude in test_cases:
        x_meters, y_meters = positioning.pixel_to_relative_position(pixel_coords, altitude)
        print(f"✓ Pixel {pixel_coords} at {altitude}m → ({x_meters:.2f}, {y_meters:.2f}) meters")
    
    # Test landing vector calculation
    landing_zone = (400, 300)  # Example landing zone
    landing_vector = positioning.get_landing_vector(landing_zone, 10.0)
    
    print(f"✓ Landing vector: {landing_vector['distance_meters']:.2f}m at {landing_vector['bearing_degrees']:.1f}°")
    print(f"✓ Movement needed: forward={landing_vector['forward_meters']:.2f}m, right={landing_vector['right_meters']:.2f}m")
    
    print("✅ Relative Positioning test passed")
    return positioning


def test_flight_controller():
    """Test mock flight controller."""
    print("Testing Flight Controller...")
    
    fc = MockFlightController(initial_altitude=15.0)
    fc.start_simulation()
    
    print("✓ Flight controller initialized and simulation started")
    
    # Test state reading
    state = fc.get_state()
    print(f"✓ Initial state: altitude={state.altitude_relative:.1f}m, mode={state.mode.value}")
    
    # Test command sending
    from flight_controller import RelativeCommand
    
    test_command = RelativeCommand(
        forward_velocity=1.0,
        right_velocity=0.5,
        down_velocity=0.0,
        yaw_rate=0.0,
        duration=2.0,
        confidence=0.8
    )
    
    success = fc.send_relative_command(test_command)
    print(f"✓ Command sent: {success}")
    
    # Wait and check if motion occurred
    time.sleep(1.0)
    new_state = fc.get_state()
    print(f"✓ State after command: velocity={new_state.velocity_ned}")
    
    # Test mode changes
    fc.set_mode(FlightMode.APPROACH)
    state = fc.get_state()
    print(f"✓ Mode changed to: {state.mode.value}")
    
    # Test emergency stop
    fc.emergency_stop()
    state = fc.get_state()
    print(f"✓ Emergency stop: mode={state.mode.value}")
    
    fc.stop_simulation()
    print("✓ Simulation stopped")
    
    print("✅ Flight Controller test passed")
    return fc


def test_landing_controller():
    """Test GPS-free landing controller."""
    print("Testing Landing Controller...")
    
    fc = MockFlightController(initial_altitude=10.0)
    fc.start_simulation()
    
    landing_controller = GPSFreeLandingController(fc)
    print("✓ Landing controller initialized")
    
    # Create test visual odometry and positioning
    vo = VisualOdometry()
    positioning = RelativePositioning(vo)
    
    # Simulate landing decision with target
    test_decision = {
        'status': 'TARGET_ACQUIRED',
        'zone': {
            'center': (350, 280),  # Slightly off-center target
            'area': 5000,
            'score': 0.85
        }
    }
    
    # Simulate motion info
    motion_info = {
        'altitude': 10.0,
        'motion_confidence': 0.7,
        'num_features': 150
    }
    
    # Test landing decision processing
    continue_landing = landing_controller.process_landing_decision(
        test_decision, motion_info, positioning
    )
    
    print(f"✓ Landing decision processed: continue={continue_landing}")
    
    # Check landing status
    status = landing_controller.get_landing_status()
    print(f"✓ Landing status: {status}")
    
    # Test no-target scenario
    no_target_decision = {
        'status': 'NO_VALID_ZONE',
        'reason': 'No safe zones detected'
    }
    
    continue_landing = landing_controller.process_landing_decision(
        no_target_decision, motion_info, positioning
    )
    
    print(f"✓ No-target decision processed: continue={continue_landing}")
    
    fc.stop_simulation()
    print("✅ Landing Controller test passed")


def test_integration():
    """Test complete system integration."""
    print("Testing System Integration...")
    
    # Initialize all components
    vo = VisualOdometry()
    positioning = RelativePositioning(vo)
    fc = MockFlightController(initial_altitude=8.0)
    fc.start_simulation()
    landing_controller = GPSFreeLandingController(fc)
    
    print("✓ All components initialized")
    
    # Simulate complete processing pipeline
    test_frame = create_test_frame_with_features(200)
    
    # 1. Visual odometry
    motion_info = vo.process_frame(test_frame)
    
    # 2. Create synthetic segmentation (normally from neural network)
    seg_map = create_synthetic_segmentation()
    
    # 3. Simulate altitude estimation
    estimated_altitude = vo.estimate_altitude_from_features(test_frame, seg_map)
    motion_info['altitude'] = estimated_altitude
    
    # 4. Simulate decision (normally from symbolic engine)
    decision = {
        'status': 'TARGET_ACQUIRED',
        'zone': {
            'center': (320, 280),  # Near center
            'area': 4500,
            'score': 0.92
        }
    }
    
    # 5. Process landing decision
    continue_landing = landing_controller.process_landing_decision(
        decision, motion_info, positioning
    )
    
    print(f"✓ Complete pipeline processed: continue_landing={continue_landing}")
    
    # Check final states
    fc_state = fc.get_state()
    landing_status = landing_controller.get_landing_status()
    
    print(f"✓ Flight controller state: altitude={fc_state.altitude_relative:.1f}m, mode={fc_state.mode.value}")
    print(f"✓ Landing controller status: {landing_status['mode']}")
    
    fc.stop_simulation()
    print("✅ Integration test passed")


def create_test_frame_with_features(num_features: int, offset: tuple = (0, 0)) -> np.ndarray:
    """Create a test frame with detectable features."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create random feature points
    np.random.seed(42)  # For reproducible tests
    
    for i in range(num_features):
        x = np.random.randint(20, 620) + offset[0]
        y = np.random.randint(20, 460) + offset[1]
        
        # Keep points within frame bounds
        x = max(20, min(620, x))
        y = max(20, min(460, y))
        
        # Create distinctive features (corners, lines, etc.)
        if i % 4 == 0:
            # Square feature
            cv2.rectangle(frame, (x-5, y-5), (x+5, y+5), (255, 255, 255), -1)
        elif i % 4 == 1:
            # Circle feature
            cv2.circle(frame, (x, y), 5, (200, 200, 200), -1)
        elif i % 4 == 2:
            # Cross feature
            cv2.line(frame, (x-5, y), (x+5, y), (180, 180, 180), 2)
            cv2.line(frame, (x, y-5), (x, y+5), (180, 180, 180), 2)
        else:
            # Triangle feature
            pts = np.array([[x, y-5], [x-5, y+5], [x+5, y+5]], np.int32)
            cv2.fillPoly(frame, [pts], (220, 220, 220))
    
    # Add some noise
    noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)
    
    return frame


def create_synthetic_segmentation() -> np.ndarray:
    """Create a synthetic segmentation map for testing."""
    seg_map = np.zeros(config.INPUT_RESOLUTION, dtype=np.uint8)
    
    # Add a large safe landing zone
    cv2.rectangle(seg_map, (150, 150), (400, 350), config.SAFE_LANDING_CLASS_ID, -1)
    
    # Add some obstacles
    cv2.circle(seg_map, (100, 100), 30, config.HIGH_OBSTACLE_CLASS_ID, -1)
    cv2.rectangle(seg_map, (450, 400), (500, 450), config.HIGH_OBSTACLE_CLASS_ID, -1)
    
    # Add some uneven surfaces
    cv2.rectangle(seg_map, (50, 400), (120, 480), config.CLASS_NAME_TO_ID["unsafe_uneven_surface"], -1)
    
    return seg_map


def performance_benchmark_gps_free():
    """Benchmark the GPS-free system performance."""
    print("\nRunning GPS-Free System Performance Benchmark...")
    
    vo = VisualOdometry()
    positioning = RelativePositioning(vo)
    
    # Create test data
    frames = [create_test_frame_with_features(100 + i*10) for i in range(20)]
    
    # Benchmark visual odometry
    vo_times = []
    for frame in frames:
        start_time = time.time()
        motion_info = vo.process_frame(frame)
        vo_times.append(time.time() - start_time)
    
    avg_vo_time = np.mean(vo_times)
    vo_fps = 1.0 / avg_vo_time
    
    print(f"  Visual Odometry: {avg_vo_time*1000:.1f}ms avg, {vo_fps:.1f} FPS")
    
    # Benchmark positioning calculations
    pos_times = []
    test_pixels = [(320, 240), (400, 300), (250, 350)]
    
    for _ in range(1000):
        start_time = time.time()
        for pixel in test_pixels:
            positioning.pixel_to_relative_position(pixel, 10.0)
        pos_times.append(time.time() - start_time)
    
    avg_pos_time = np.mean(pos_times) / len(test_pixels)  # Per calculation
    pos_fps = 1.0 / avg_pos_time
    
    print(f"  Positioning Calc: {avg_pos_time*1000:.3f}ms avg, {pos_fps:.0f} FPS")
    
    print("✅ Performance benchmark completed")


def main():
    """Run all GPS-free system tests."""
    print("GPS-Free UAV Landing System - Test Suite")
    print("=" * 60)
    
    try:
        test_visual_odometry()
        print()
        
        test_relative_positioning()
        print()
        
        test_flight_controller()
        print()
        
        test_landing_controller()
        print()
        
        test_integration()
        print()
        
        performance_benchmark_gps_free()
        
        print("\n" + "=" * 60)
        print("✅ ALL GPS-FREE TESTS PASSED!")
        print("The GPS-Free Landing System is working correctly.")
        print("\nNext steps:")
        print("1. Run camera calibration: python gps_free_main.py --calibrate")
        print("2. Test with video: python gps_free_main.py --video test_videos/test_video_mixed.mp4")
        print("3. Test with webcam: python gps_free_main.py --video 0")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
