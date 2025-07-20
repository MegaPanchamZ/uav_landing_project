#!/usr/bin/env python3
"""
Test script for Neurosymbolic Memory System
Demonstrates memory-based landing zone detection in challenging scenarios
"""

import sys
import numpy as np
import cv2
import time
import math
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from memory_enhanced_detector import MemoryEnhancedUAVDetector
from neurosymbolic_memory import NeuroSymbolicMemory


def create_test_scenario_frame(scenario_type: str, frame_count: int, noise_level: float = 0.1) -> np.ndarray:
    """Create synthetic test frames for different scenarios"""
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    if scenario_type == "clear_landing_pad":
        # Clear landing pad scenario
        frame[:] = [120, 180, 100]  # Grass background
        
        # Landing pad
        center_x, center_y = 320, 240
        pad_size = 80
        cv2.rectangle(frame, 
                     (center_x - pad_size, center_y - pad_size),
                     (center_x + pad_size, center_y + pad_size),
                     (200, 200, 200), -1)  # Concrete pad
        
        # Add some variation
        if frame_count % 10 < 3:
            # Occasionally add some obstacles
            cv2.circle(frame, (center_x + 60, center_y - 40), 20, (100, 50, 0), -1)
    
    elif scenario_type == "grass_only":
        # Uniform grass - challenging for visual detection
        frame[:] = [60, 120, 60]  # Grass color
        
        # Add subtle texture variation
        for i in range(0, frame.shape[0], 20):
            for j in range(0, frame.shape[1], 20):
                variation = np.random.randint(-10, 10, 3)
                end_i = min(i + 20, frame.shape[0])
                end_j = min(j + 20, frame.shape[1])
                frame[i:end_i, j:end_j] = np.clip(
                    frame[i:end_i, j:end_j].astype(np.int16) + variation, 0, 255
                ).astype(np.uint8)
    
    elif scenario_type == "partially_occluded":
        # Landing pad partially visible
        frame[:] = [120, 180, 100]  # Grass background
        
        # Landing pad
        center_x, center_y = 320, 240
        pad_size = 80
        
        # Only show part of the landing pad
        visible_fraction = 0.3 + 0.4 * math.sin(frame_count * 0.1)  # Varies over time
        visible_width = int(pad_size * 2 * visible_fraction)
        
        cv2.rectangle(frame,
                     (center_x - visible_width // 2, center_y - pad_size),
                     (center_x + visible_width // 2, center_y + pad_size),
                     (200, 200, 200), -1)
    
    elif scenario_type == "moving_target":
        # Landing pad that moves (simulates drone movement)
        frame[:] = [120, 180, 100]  # Grass background
        
        # Moving landing pad
        center_x = 320 + int(50 * math.sin(frame_count * 0.05))
        center_y = 240 + int(30 * math.cos(frame_count * 0.07))
        pad_size = 60
        
        cv2.rectangle(frame,
                     (center_x - pad_size, center_y - pad_size),
                     (center_x + pad_size, center_y + pad_size),
                     (200, 200, 200), -1)
    
    elif scenario_type == "multiple_zones":
        # Multiple potential landing zones
        frame[:] = [120, 180, 100]  # Grass background
        
        zones = [
            (200, 180, 40),  # (x, y, size)
            (450, 200, 35),
            (320, 320, 50)
        ]
        
        for i, (x, y, size) in enumerate(zones):
            # Vary quality of zones over time
            quality = 0.5 + 0.5 * math.sin(frame_count * 0.1 + i)
            color_intensity = int(150 + 50 * quality)
            cv2.rectangle(frame,
                         (x - size, y - size),
                         (x + size, y + size),
                         (color_intensity, color_intensity, color_intensity), -1)
    
    else:  # random noise
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Add noise
    if noise_level > 0:
        noise = np.random.randint(
            int(-255 * noise_level), 
            int(255 * noise_level), 
            frame.shape, dtype=np.int16
        )
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return frame


def run_memory_test_scenario(detector, scenario_name: str, duration_frames: int = 100):
    """Run a specific test scenario and analyze memory performance"""
    
    print(f"\n🧪 Running scenario: {scenario_name}")
    print("=" * 50)
    
    # Scenario parameters
    scenarios = {
        "learning_phase": {
            "frames": ["clear_landing_pad"] * 30,
            "altitude_profile": [10.0] * 30,
            "description": "Clear conditions - build memory"
        },
        "grass_challenge": {
            "frames": ["grass_only"] * 50,
            "altitude_profile": [5.0] * 50,
            "description": "Uniform grass - memory only"
        },
        "partial_occlusion": {
            "frames": ["partially_occluded"] * 40,
            "altitude_profile": [3.0] * 40,
            "description": "Partial visibility - memory fusion"
        },
        "moving_target": {
            "frames": ["moving_target"] * 60,
            "altitude_profile": [4.0] * 60,
            "description": "Moving target - memory tracking"
        },
        "multiple_zones": {
            "frames": ["multiple_zones"] * 45,
            "altitude_profile": [6.0] * 45,
            "description": "Multiple zones - memory selection"
        }
    }
    
    if scenario_name not in scenarios:
        print(f"❌ Unknown scenario: {scenario_name}")
        return
    
    scenario = scenarios[scenario_name]
    frame_types = scenario["frames"]
    altitudes = scenario["altitude_profile"]
    
    print(f"Description: {scenario['description']}")
    
    # Statistics tracking
    results = {
        "frame_count": 0,
        "target_acquired_count": 0,
        "memory_used_count": 0,
        "fusion_used_count": 0,
        "recovery_mode_count": 0,
        "avg_confidence": 0.0,
        "avg_memory_confidence": 0.0,
        "visual_confidence_history": [],
        "memory_zones_history": []
    }
    
    # Simulate drone position
    drone_pos = [0.0, 0.0]
    
    for frame_idx in range(len(frame_types)):
        frame_type = frame_types[frame_idx]
        altitude = altitudes[frame_idx]
        
        # Create test frame
        frame = create_test_scenario_frame(frame_type, frame_idx)
        
        # Simulate drone movement
        drone_pos[0] += (np.random.random() - 0.5) * 0.2
        drone_pos[1] += (np.random.random() - 0.5) * 0.2
        
        # Process frame
        result = detector.process_frame(
            frame,
            altitude=altitude,
            drone_position=tuple(drone_pos),
            drone_heading=0.0
        )
        
        # Collect statistics
        results["frame_count"] += 1
        
        if result.status == "TARGET_ACQUIRED":
            results["target_acquired_count"] += 1
            results["avg_confidence"] += result.confidence
        
        if result.perception_memory_fusion == "memory_only":
            results["memory_used_count"] += 1
        elif result.perception_memory_fusion == "fused":
            results["fusion_used_count"] += 1
        
        if result.recovery_mode:
            results["recovery_mode_count"] += 1
        
        results["avg_memory_confidence"] += result.memory_confidence
        results["memory_zones_history"].append(len(result.memory_zones))
        
        # Show progress occasionally
        if frame_idx % 20 == 0:
            print(f"  Frame {frame_idx}: {result.status} "
                  f"(conf: {result.confidence:.2f}, "
                  f"mem_conf: {result.memory_confidence:.2f}, "
                  f"fusion: {result.perception_memory_fusion})")
    
    # Calculate final statistics
    if results["target_acquired_count"] > 0:
        results["avg_confidence"] /= results["target_acquired_count"]
    
    results["avg_memory_confidence"] /= results["frame_count"]
    
    # Print summary
    print(f"\n📊 Scenario Results:")
    print(f"  Total frames: {results['frame_count']}")
    print(f"  Targets acquired: {results['target_acquired_count']} ({100*results['target_acquired_count']/results['frame_count']:.1f}%)")
    print(f"  Memory-only used: {results['memory_used_count']} ({100*results['memory_used_count']/results['frame_count']:.1f}%)")
    print(f"  Fusion used: {results['fusion_used_count']} ({100*results['fusion_used_count']/results['frame_count']:.1f}%)")
    print(f"  Recovery mode: {results['recovery_mode_count']} ({100*results['recovery_mode_count']/results['frame_count']:.1f}%)")
    print(f"  Avg confidence: {results['avg_confidence']:.3f}")
    print(f"  Avg memory confidence: {results['avg_memory_confidence']:.3f}")
    print(f"  Max memory zones: {max(results['memory_zones_history']) if results['memory_zones_history'] else 0}")
    
    return results


def run_interactive_demo():
    """Run interactive demo with visualization"""
    
    print("🚁 Interactive Memory-Enhanced UAV Detector Demo")
    print("=" * 55)
    
    # Initialize detector
    detector = MemoryEnhancedUAVDetector(
        enable_visualization=True,
        enable_memory=True,
        memory_config={
            'memory_horizon': 300.0,
            'spatial_resolution': 0.5,
            'confidence_decay_rate': 0.985,
            'min_observations': 2
        }
    )
    
    print("🎮 Controls:")
    print("  1-5: Switch scenarios")
    print("  'r': Reset memory")
    print("  's': Show memory stats")
    print("  'q': Quit")
    print("\nScenarios:")
    print("  1: Clear landing pad")
    print("  2: Grass only (memory challenge)")
    print("  3: Partially occluded target")
    print("  4: Moving target")
    print("  5: Multiple zones")
    
    current_scenario = "clear_landing_pad"
    frame_count = 0
    altitude = 8.0
    drone_pos = [0.0, 0.0]
    
    try:
        while True:
            # Create frame based on current scenario
            frame = create_test_scenario_frame(current_scenario, frame_count)
            
            # Update drone position (simulate movement)
            drone_pos[0] += (np.random.random() - 0.5) * 0.1
            drone_pos[1] += (np.random.random() - 0.5) * 0.1
            
            # Process frame
            result = detector.process_frame(
                frame,
                altitude=altitude,
                drone_position=tuple(drone_pos),
                drone_heading=0.0
            )
            
            # Create display frame
            display_frame = result.annotated_image if result.annotated_image is not None else frame
            
            # Add scenario and memory info overlay
            info_text = [
                f"Scenario: {current_scenario}",
                f"Status: {result.status}",
                f"Confidence: {result.confidence:.3f}",
                f"Memory conf: {result.memory_confidence:.3f}",
                f"Fusion: {result.perception_memory_fusion}",
                f"Memory zones: {len(result.memory_zones)}",
                f"Recovery: {'YES' if result.recovery_mode else 'NO'}"
            ]
            
            for i, text in enumerate(info_text):
                color = (0, 255, 0) if result.status == "TARGET_ACQUIRED" else (0, 0, 255)
                cv2.putText(display_frame, text, (10, 150 + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Show main display
            cv2.imshow("Memory-Enhanced UAV Detector Demo", display_frame)
            
            # Show memory visualization
            memory_viz = detector.get_memory_visualization()
            if memory_viz is not None:
                # Add title
                cv2.putText(memory_viz, "Memory Map", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Memory Visualization", memory_viz)
            
            # Handle input
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('1'):
                current_scenario = "clear_landing_pad"
                frame_count = 0
                print("📍 Switched to: Clear landing pad")
            elif key == ord('2'):
                current_scenario = "grass_only"
                frame_count = 0
                print("🌱 Switched to: Grass only (memory challenge)")
            elif key == ord('3'):
                current_scenario = "partially_occluded"
                frame_count = 0
                print("🔍 Switched to: Partially occluded")
            elif key == ord('4'):
                current_scenario = "moving_target"
                frame_count = 0
                print("🎯 Switched to: Moving target")
            elif key == ord('5'):
                current_scenario = "multiple_zones"
                frame_count = 0
                print("🎪 Switched to: Multiple zones")
            elif key == ord('r'):
                detector.reset_memory()
                drone_pos = [0.0, 0.0]
                altitude = 8.0
                frame_count = 0
                print("🔄 Memory reset")
            elif key == ord('s'):
                if hasattr(result, 'memory_status'):
                    print(f"\n📊 Memory Status:")
                    for k, v in result.memory_status.items():
                        print(f"  {k}: {v}")
            
            frame_count += 1
            
            # Simulate descent when target acquired
            if result.status == "TARGET_ACQUIRED" and result.descent_rate > 0:
                altitude = max(0.5, altitude - 0.05)
            elif result.recovery_mode:
                altitude = max(2.0, altitude)  # Don't descend while searching
    
    except KeyboardInterrupt:
        pass
    
    # Save memory
    detector.save_memory()
    cv2.destroyAllWindows()
    print("\n✅ Interactive demo completed")


def main():
    """Main test function"""
    
    print("🧠 Neurosymbolic Memory System Test Suite")
    print("=" * 50)
    
    choice = input("\nSelect test mode:\n"
                  "1. Automated scenario tests\n"
                  "2. Interactive demo\n"
                  "Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Run automated tests
        print("\n🤖 Running automated scenario tests...")
        
        # Initialize detector
        detector = MemoryEnhancedUAVDetector(
            enable_visualization=False,
            enable_memory=True,
            memory_config={
                'memory_horizon': 300.0,
                'spatial_resolution': 0.5,
                'confidence_decay_rate': 0.985,
                'min_observations': 2
            }
        )
        
        # Run test scenarios in sequence
        scenarios = [
            "learning_phase",
            "grass_challenge", 
            "partial_occlusion",
            "moving_target",
            "multiple_zones"
        ]
        
        all_results = {}
        
        for scenario in scenarios:
            all_results[scenario] = run_memory_test_scenario(detector, scenario)
            time.sleep(1)  # Brief pause between scenarios
        
        # Overall summary
        print("\n" + "=" * 60)
        print("🏆 OVERALL TEST SUMMARY")
        print("=" * 60)
        
        for scenario, results in all_results.items():
            success_rate = 100 * results["target_acquired_count"] / results["frame_count"]
            memory_usage = 100 * (results["memory_used_count"] + results["fusion_used_count"]) / results["frame_count"]
            print(f"{scenario:20}: Success {success_rate:5.1f}%, Memory {memory_usage:5.1f}%")
        
        # Save memory for future use
        detector.save_memory()
        print(f"\n💾 Memory saved to: {detector.memory_persistence_file}")
        
    elif choice == "2":
        # Run interactive demo
        run_interactive_demo()
        
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
