#!/usr/bin/env python3
"""
UAV Landing System - Production Entry Point

A complete, clean system for autonomous UAV landing using neural segmentation
and symbolic reasoning with persistent memory capabilities.

Usage:
    python uav_landing_main.py [options]
    
Examples:
    # Basic usage with webcam
    python uav_landing_main.py --camera 0
    
    # High resolution for precision landing
    python uav_landing_main.py --resolution 768x768 --enable-memory
    
    # Test with synthetic data
    python uav_landing_main.py --test-mode
"""

import argparse
import cv2
import numpy as np
import time
import sys
from pathlib import Path

# Add the uav_landing module to path
sys.path.insert(0, str(Path(__file__).parent))

from uav_landing import UAVLandingDetector


def parse_args():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(
        description="UAV Landing System with Neurosymbolic Memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --camera 0                    # Use webcam
  %(prog)s --resolution 768x768          # High precision
  %(prog)s --test-mode                   # Synthetic test data
  %(prog)s --disable-memory              # Disable memory system
        """
    )
    
    # Input options
    parser.add_argument('--camera', type=int, default=None,
                       help='Camera device index (default: synthetic test)')
    parser.add_argument('--video', type=str, default=None,
                       help='Input video file path')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run with synthetic test data')
    
    # Model options
    parser.add_argument('--model-path', type=str, 
                       default='models/bisenetv2_uav_landing.onnx',
                       help='Path to ONNX model file')
    parser.add_argument('--resolution', type=str, default='512x512',
                       help='Model input resolution (WxH): 256x256, 512x512, 768x768, 1024x1024')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Inference device')
    
    # Memory options  
    parser.add_argument('--disable-memory', action='store_true',
                       help='Disable neurosymbolic memory system')
    parser.add_argument('--memory-file', type=str, default='uav_memory.json',
                       help='Memory persistence file')
    parser.add_argument('--memory-horizon', type=float, default=300.0,
                       help='Memory horizon in seconds')
    
    # Camera parameters
    parser.add_argument('--camera-fx', type=float, default=800.0,
                       help='Camera focal length X')
    parser.add_argument('--camera-fy', type=float, default=800.0,
                       help='Camera focal length Y')
    
    # Display options
    parser.add_argument('--no-display', action='store_true',
                       help='Disable visualization display')
    parser.add_argument('--save-video', type=str, default=None,
                       help='Save output video to file')
    parser.add_argument('--fps-limit', type=float, default=30.0,
                       help='Limit processing FPS')
    
    # Flight simulation
    parser.add_argument('--altitude', type=float, default=10.0,
                       help='Initial altitude for simulation (meters)')
    parser.add_argument('--simulate-position', action='store_true',
                       help='Simulate drone position movement')
    
    return parser.parse_args()


def create_test_frame(frame_count: int, scenario: str = "mixed") -> np.ndarray:
    """Create synthetic test frames"""
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    if scenario == "clear":
        # Clear landing pad
        frame[:] = [120, 180, 100]  # Grass background
        cv2.rectangle(frame, (270, 190), (370, 290), (200, 200, 200), -1)
        
    elif scenario == "grass":
        # Challenging grass-only scenario
        frame[:] = [60, 120, 60]
        noise = np.random.randint(-20, 20, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
    elif scenario == "mixed":
        # Cycle through different scenarios
        cycle = (frame_count // 30) % 4
        if cycle == 0:
            return create_test_frame(frame_count, "clear")
        elif cycle == 1:
            return create_test_frame(frame_count, "grass")
        elif cycle == 2:
            # Partially visible
            frame[:] = [120, 180, 100]
            visibility = 0.3 + 0.4 * np.sin(frame_count * 0.1)
            pad_width = int(50 * visibility)
            cv2.rectangle(frame, (320 - pad_width, 190), (320 + pad_width, 290), (200, 200, 200), -1)
        else:
            # Moving target
            frame[:] = [120, 180, 100]
            center_x = 320 + int(50 * np.sin(frame_count * 0.05))
            center_y = 240 + int(30 * np.cos(frame_count * 0.07))
            cv2.rectangle(frame, (center_x - 40, center_y - 40), (center_x + 40, center_y + 40), (200, 200, 200), -1)
    
    return frame


def main():
    """Main function"""
    
    args = parse_args()
    
    print("🚁 UAV Landing System - Production Version")
    print("=" * 50)
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        input_resolution = (width, height)
    except ValueError:
        print(f"❌ Invalid resolution format: {args.resolution}")
        sys.exit(1)
    
    # Memory configuration
    memory_config = {
        'memory_horizon': args.memory_horizon,
        'spatial_resolution': 0.5,
        'confidence_decay_rate': 0.985,
        'min_observations': 2
    }
    
    # Initialize detector
    print(f"🔧 Initializing detector...")
    print(f"   Model: {args.model_path}")
    print(f"   Resolution: {input_resolution}")
    print(f"   Memory: {'enabled' if not args.disable_memory else 'disabled'}")
    print(f"   Device: {args.device}")
    
    detector = UAVLandingDetector(
        model_path=args.model_path,
        input_resolution=input_resolution,
        camera_fx=args.camera_fx,
        camera_fy=args.camera_fy,
        enable_memory=not args.disable_memory,
        enable_visualization=not args.no_display,
        memory_config=memory_config,
        memory_persistence_file=args.memory_file,
        device=args.device
    )
    
    # Initialize input source
    cap = None
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"❌ Failed to open camera {args.camera}")
            sys.exit(1)
        print(f"📹 Using camera {args.camera}")
        
    elif args.video is not None:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"❌ Failed to open video file {args.video}")
            sys.exit(1)
        print(f"📹 Using video file {args.video}")
        
    elif args.test_mode:
        print("🧪 Using synthetic test data")
        
    else:
        print("⚠️  No input specified, using synthetic test data")
        args.test_mode = True
    
    # Initialize video writer if needed
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.save_video, fourcc, 20.0, (640, 480))
        print(f"💾 Saving output to {args.save_video}")
    
    # Flight simulation parameters
    altitude = args.altitude
    drone_position = [0.0, 0.0] if args.simulate_position else None
    frame_count = 0
    last_time = time.time()
    
    print("\n🎮 Controls:")
    print("  'q': Quit")
    print("  'r': Reset memory and state")
    print("  's': Show performance stats")
    print("  'm': Show memory status")
    print("  'SPACE': Pause/Resume")
    print()
    
    paused = False
    
    try:
        while True:
            current_time = time.time()
            
            # FPS limiting
            if args.fps_limit > 0:
                min_frame_time = 1.0 / args.fps_limit
                if current_time - last_time < min_frame_time:
                    time.sleep(min_frame_time - (current_time - last_time))
                    current_time = time.time()
            
            # Get frame
            if args.test_mode:
                frame = create_test_frame(frame_count, "mixed")
                time.sleep(0.033)  # Simulate ~30 FPS
                
            else:
                ret, frame = cap.read()
                if not ret:
                    if args.video:
                        print("📹 Video ended")
                        break
                    else:
                        print("❌ Failed to read frame")
                        continue
            
            if not paused:
                # Simulate drone position movement
                if args.simulate_position and drone_position:
                    drone_position[0] += (np.random.random() - 0.5) * 0.1
                    drone_position[1] += (np.random.random() - 0.5) * 0.1
                
                # Process frame
                result = detector.process_frame(
                    image=frame,
                    altitude=altitude,
                    drone_position=tuple(drone_position) if drone_position else None,
                    drone_heading=0.0
                )
                
                # Print status for interesting results
                if result.perception_memory_fusion != "perception_only":
                    print(f"🧠 Frame {frame_count}: Memory active ({result.perception_memory_fusion}), "
                          f"Status: {result.status}, Memory zones: {len(result.memory_zones)}")
                
                if result.recovery_mode:
                    print(f"🔄 Frame {frame_count}: Recovery mode - {result.search_pattern}")
                
                # Simulate altitude changes
                if result.status == "TARGET_ACQUIRED" and result.descent_rate > 0:
                    altitude = max(0.5, altitude - 0.05)
                elif result.recovery_mode:
                    altitude = max(2.0, altitude)  # Don't descend while searching
                
                frame_count += 1
            
            # Display
            if not args.no_display:
                display_frame = result.annotated_image if not paused and result.annotated_image is not None else frame
                
                if paused:
                    cv2.putText(display_frame, "PAUSED - Press SPACE to resume", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow("UAV Landing System", display_frame)
                
                # Save frame if recording
                if video_writer is not None and not paused:
                    video_writer.write(display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                detector.reset_state()
                altitude = args.altitude
                drone_position = [0.0, 0.0] if args.simulate_position else None
                frame_count = 0
                print("🔄 System reset")
            elif key == ord('s'):
                stats = detector.get_performance_stats()
                print(f"\n📊 Performance Stats:")
                for k, v in stats.items():
                    print(f"   {k}: {v}")
                print()
            elif key == ord('m'):
                if detector.memory:
                    status = detector.memory.get_memory_status()
                    print(f"\n🧠 Memory Status:")
                    for k, v in status.items():
                        if isinstance(v, dict):
                            print(f"   {k}:")
                            for k2, v2 in v.items():
                                print(f"     {k2}: {v2}")
                        else:
                            print(f"   {k}: {v}")
                    print()
                else:
                    print("⚠️  Memory system disabled")
            elif key == ord(' '):
                paused = not paused
                print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
            
            last_time = current_time
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    # Cleanup
    if cap:
        cap.release()
    
    if video_writer:
        video_writer.release()
        print(f"💾 Video saved to {args.save_video}")
    
    if not args.no_display:
        cv2.destroyAllWindows()
    
    # Save memory
    detector.save_memory()
    print("💾 Memory state saved")
    
    print("✅ UAV Landing System shutdown complete")


if __name__ == "__main__":
    main()
