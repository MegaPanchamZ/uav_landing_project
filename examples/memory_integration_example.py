#!/usr/bin/env python3
"""
Simple integration example for the neurosymbolic memory system.
Shows how to upgrade existing UAV landing code to use memory.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import time

# Add src directory for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from memory_enhanced_detector import MemoryEnhancedUAVDetector


def example_basic_usage():
    """Basic usage example - minimal changes to existing code"""
    
    print("🚁 Basic Memory Integration Example")
    print("=" * 40)
    
    # Initialize detector with memory enabled
    # This is the main change - use MemoryEnhancedUAVDetector instead of UAVLandingDetector
    detector = MemoryEnhancedUAVDetector(
        model_path="models/bisenetv2_uav_landing.onnx",  # Your existing model
        input_resolution=(512, 512),  # Your preferred resolution
        enable_visualization=True,
        enable_memory=True,  # Enable memory system
        memory_persistence_file="flight_memory.json"  # Persistent across flights
    )
    
    # Your existing flight parameters
    altitude = 10.0
    current_velocity = (0.0, 0.0, 0.0)
    
    # Simulate position information (in real use, get from GPS/odometry)
    drone_position = [0.0, 0.0]  # [x, y] in world coordinates
    drone_heading = 0.0  # radians
    
    print("Processing frames with memory enhancement...")
    
    # Your existing frame processing loop
    for frame_count in range(50):  # Simulate 50 frames
        
        # Get frame from your camera (this part stays the same)
        frame = create_demo_frame(frame_count)  # Simulated for demo
        
        # Update position (in real use, get from navigation system)
        drone_position[0] += (np.random.random() - 0.5) * 0.1
        drone_position[1] += (np.random.random() - 0.5) * 0.1
        
        # Process frame - enhanced call with position info
        result = detector.process_frame(
            image=frame,
            altitude=altitude,
            current_velocity=current_velocity,
            drone_position=tuple(drone_position),  # NEW: position for memory
            drone_heading=drone_heading  # NEW: heading for memory
        )
        
        # Your existing result processing (unchanged)
        if result.status == "TARGET_ACQUIRED":
            print(f"Frame {frame_count}: Target at {result.distance:.1f}m "
                  f"(confidence: {result.confidence:.2f})")
            
            # Use navigation commands as before
            forward_cmd = result.forward_velocity
            right_cmd = result.right_velocity  
            descent_cmd = result.descent_rate
            
            # NEW: Check if memory was used
            if result.perception_memory_fusion != "perception_only":
                print(f"  💭 Memory assist: {result.perception_memory_fusion}")
                print(f"  🧠 Memory zones available: {len(result.memory_zones)}")
        
        elif result.status == "NO_TARGET":
            # NEW: Memory might still provide guidance
            if result.memory_zones:
                print(f"Frame {frame_count}: No visual target, but {len(result.memory_zones)} memory zones available")
            
            # NEW: Check if in recovery mode
            if result.recovery_mode:
                print(f"  🔄 Recovery mode: {result.search_pattern}")
        
        # Simulate drone movement and descent
        if result.descent_rate > 0:
            altitude = max(0.5, altitude - 0.1)
        
        time.sleep(0.1)  # Simulate real-time processing
    
    # NEW: Save memory for next flight
    detector.save_memory()
    print(f"\n💾 Memory saved for next flight")


def create_demo_frame(frame_count: int) -> np.ndarray:
    """Create demo frames showing different scenarios"""
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    if frame_count < 15:
        # Clear conditions - landing pad visible
        frame[:] = [120, 180, 100]  # Grass background
        cv2.rectangle(frame, (270, 190), (370, 290), (200, 200, 200), -1)  # Landing pad
        
    elif frame_count < 35:
        # Challenging conditions - mostly grass (memory should help here)
        frame[:] = [60, 120, 60]  # Uniform grass
        # Add subtle texture
        noise = np.random.randint(-20, 20, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
    else:
        # Mixed conditions - partial visibility
        frame[:] = [120, 180, 100]  # Grass background
        # Partially visible landing pad
        visibility = 0.3 + 0.4 * np.sin((frame_count - 35) * 0.2)
        pad_width = int(50 * visibility)
        cv2.rectangle(frame, (320 - pad_width, 190), (320 + pad_width, 290), (200, 200, 200), -1)
    
    return frame


def example_advanced_usage():
    """Advanced usage showing memory configuration and analysis"""
    
    print("\n🧠 Advanced Memory Usage Example")
    print("=" * 40)
    
    # Configure memory system for specific use case
    memory_config = {
        'memory_horizon': 600.0,        # 10 minutes of memory
        'spatial_resolution': 0.3,      # High resolution (30cm grid)
        'confidence_decay_rate': 0.99,  # Slow decay for long flights
        'min_observations': 3           # Conservative - need 3 observations
    }
    
    detector = MemoryEnhancedUAVDetector(
        enable_memory=True,
        memory_config=memory_config,
        memory_persistence_file="advanced_memory.json"
    )
    
    print("Memory configuration:")
    for key, value in memory_config.items():
        print(f"  {key}: {value}")
    
    # Simulate a flight with memory analysis
    positions = [(0, 0), (2, 1), (1, 3), (-1, 2), (0, 0)]  # Return to start
    
    for i, pos in enumerate(positions):
        frame = create_demo_frame(i * 10)
        
        result = detector.process_frame(
            frame,
            altitude=5.0,
            drone_position=pos
        )
        
        # Analyze memory state
        memory_status = result.memory_status
        print(f"\nPosition {pos}: Status={result.status}")
        print(f"  Memory zones: {memory_status.get('active_memory_zones', 0)}")
        print(f"  Grid coverage: {memory_status.get('grid_coverage', 0):.2%}")
        print(f"  Fusion mode: {result.perception_memory_fusion}")
        
        # Get memory visualization if available
        memory_viz = detector.get_memory_visualization()
        if memory_viz is not None:
            cv2.imwrite(f"memory_viz_{i}.jpg", memory_viz)
            print(f"  Memory visualization saved to memory_viz_{i}.jpg")
    
    # Final memory analysis
    final_status = detector.memory.get_memory_status()
    print(f"\n📊 Final Memory Status:")
    print(f"  Total zones: {final_status['total_memory_zones']}")
    print(f"  Memory age: {final_status['memory_age']:.1f}s")
    print(f"  Frame count: {final_status['frame_count']}")
    
    detector.save_memory()


def example_recovery_behavior():
    """Example showing recovery behavior when target is lost"""
    
    print("\n🔄 Recovery Behavior Example")  
    print("=" * 40)
    
    detector = MemoryEnhancedUAVDetector(
        enable_memory=True,
        enable_visualization=True
    )
    
    # Simulate scenario: good detection -> lost target -> recovery
    scenarios = [
        ("good", 10),      # 10 frames of good detection
        ("lost", 15),      # 15 frames of lost target
        ("recovery", 10)   # 10 frames of recovery
    ]
    
    frame_count = 0
    for scenario_name, duration in scenarios:
        print(f"\n--- {scenario_name.upper()} PHASE ---")
        
        for i in range(duration):
            if scenario_name == "good":
                # Clear landing pad
                frame = np.full((480, 640, 3), [120, 180, 100], dtype=np.uint8)
                cv2.rectangle(frame, (270, 190), (370, 290), (200, 200, 200), -1)
                
            elif scenario_name == "lost":
                # Uniform grass - no clear target
                frame = np.full((480, 640, 3), [60, 120, 60], dtype=np.uint8)
                
            else:  # recovery
                # Gradually reveal target
                frame = np.full((480, 640, 3), [120, 180, 100], dtype=np.uint8)
                reveal_factor = i / duration
                pad_size = int(50 * reveal_factor)
                if pad_size > 10:
                    cv2.rectangle(frame, (320-pad_size, 240-pad_size), 
                                 (320+pad_size, 240+pad_size), (200, 200, 200), -1)
            
            # Process frame
            result = detector.process_frame(
                frame,
                altitude=5.0,
                drone_position=(frame_count * 0.1, 0)
            )
            
            # Show recovery behavior
            if result.recovery_mode:
                print(f"  Frame {frame_count}: Recovery mode - {result.search_pattern}")
                print(f"    Commands: fwd={result.forward_velocity:.2f}, "
                      f"right={result.right_velocity:.2f}")
            
            elif result.perception_memory_fusion != "perception_only":
                print(f"  Frame {frame_count}: Using memory - {result.perception_memory_fusion}")
            
            elif result.status == "TARGET_ACQUIRED":
                print(f"  Frame {frame_count}: Target acquired (conf: {result.confidence:.2f})")
            
            frame_count += 1
    
    print(f"\n✅ Recovery behavior demonstration complete")


def main():
    """Run example based on user choice"""
    
    print("🧠 Neurosymbolic Memory Integration Examples")
    print("=" * 50)
    
    examples = {
        "1": ("Basic Usage", example_basic_usage),
        "2": ("Advanced Configuration", example_advanced_usage), 
        "3": ("Recovery Behavior", example_recovery_behavior),
        "4": ("All Examples", None)
    }
    
    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}: {name}")
    
    choice = input("\nSelect example (1-4): ").strip()
    
    if choice == "4":
        # Run all examples
        for key, (name, func) in examples.items():
            if func:  # Skip "All Examples" entry
                print(f"\n{'='*60}")
                print(f"Running: {name}")
                print(f"{'='*60}")
                func()
                time.sleep(2)
    elif choice in examples and examples[choice][1]:
        examples[choice][1]()
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
