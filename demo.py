#!/usr/bin/env python3
"""
UAV Landing Detector Demo

Simple demo script to test the single-class UAV landing detector.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from uav_landing_detector import UAVLandingDetector

def create_test_image(width=640, height=480, pattern="landing_zone"):
    """Create synthetic test images for demonstration."""
    
    image = np.random.randint(60, 180, (height, width, 3), dtype=np.uint8)
    
    if pattern == "landing_zone":
        # Add a clear landing zone in center
        center_x, center_y = width // 2, height // 2
        cv2.rectangle(image, 
                     (center_x - 80, center_y - 60), 
                     (center_x + 80, center_y + 60), 
                     (100, 150, 100), -1)
        
        # Add some obstacles
        cv2.rectangle(image, (50, 50), (120, 150), (80, 80, 80), -1)  # Building
        cv2.circle(image, (width - 100, 100), 40, (60, 60, 60), -1)  # Tree
        
    elif pattern == "obstacles":
        # Image with many obstacles
        for i in range(5):
            x = np.random.randint(50, width - 50)
            y = np.random.randint(50, height - 50)
            size = np.random.randint(30, 80)
            cv2.rectangle(image, (x, y), (x + size, y + size), (80, 80, 80), -1)
            
    elif pattern == "marginal":
        # Marginal landing areas
        cv2.rectangle(image, (200, 200), (440, 280), (120, 140, 110), -1)
        
    return image

def run_webcam_demo():
    """Run demo with webcam input."""
    
    print("🎥 Starting webcam demo...")
    
    # Initialize detector
    detector = UAVLandingDetector(
        model_path=None,  # Use placeholder mode
        enable_visualization=True
    )
    
    # Try to open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return False
    
    print("🎮 Controls:")
    print("  'q' - Quit")
    print("  'r' - Reset detector state") 
    print("  's' - Show statistics")
    print("  'h' - Increase altitude")
    print("  'l' - Decrease altitude")
    
    altitude = 5.0
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process frame
        result = detector.process_frame(frame, altitude=altitude)
        
        # Display results
        if result.annotated_image is not None:
            cv2.imshow("UAV Landing Detector", result.annotated_image)
        else:
            cv2.imshow("UAV Landing Detector", frame)
        
        # Print status every 30 frames
        if frame_count % 30 == 0:
            if result.status == "TARGET_ACQUIRED":
                print(f"🎯 Target: {result.distance:.1f}m, Alt: {altitude:.1f}m, "
                      f"Cmd: [{result.forward_velocity:.1f}, {result.right_velocity:.1f}, {result.descent_rate:.1f}]")
            else:
                print(f"🔍 Status: {result.status}, Alt: {altitude:.1f}m, FPS: {result.fps:.1f}")
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset_state()
            altitude = 5.0
            print("🔄 Reset detector state")
        elif key == ord('s'):
            stats = detector.get_performance_stats()
            print(f"📊 Stats: {stats}")
        elif key == ord('h'):
            altitude = min(20.0, altitude + 0.5)
            print(f"⬆️  Altitude: {altitude:.1f}m")
        elif key == ord('l'):
            altitude = max(0.5, altitude - 0.5)
            print(f"⬇️  Altitude: {altitude:.1f}m")
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def run_synthetic_demo():
    """Run demo with synthetic test images."""
    
    print("🔬 Starting synthetic demo...")
    
    # Initialize detector
    detector = UAVLandingDetector(
        model_path=None,  # Use placeholder mode
        enable_visualization=True
    )
    
    test_patterns = ["landing_zone", "obstacles", "marginal"]
    altitudes = [8.0, 4.0, 2.0, 1.0]
    
    print("🎮 Controls:")
    print("  'q' - Quit")
    print("  'n' - Next test pattern")
    print("  'r' - Reset detector")
    print("  SPACE - Pause/Resume")
    
    pattern_idx = 0
    altitude_idx = 0
    paused = False
    
    while True:
        if not paused:
            # Create test image
            pattern = test_patterns[pattern_idx]
            altitude = altitudes[altitude_idx]
            
            image = create_test_image(pattern=pattern)
            
            # Process frame
            result = detector.process_frame(image, altitude=altitude)
            
            # Auto-advance altitude for landing simulation
            if result.status == "TARGET_ACQUIRED" and result.descent_rate > 0:
                altitude = max(0.5, altitude - 0.1)
                altitudes[altitude_idx] = altitude
        
        # Display results
        if result.annotated_image is not None:
            display_img = result.annotated_image.copy()
        else:
            display_img = image.copy()
        
        # Add test info overlay
        cv2.putText(display_img, f"Pattern: {pattern}", (10, display_img.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_img, f"Altitude: {altitude:.1f}m", (10, display_img.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if paused:
            cv2.putText(display_img, "PAUSED", (display_img.shape[1]//2 - 50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        cv2.imshow("UAV Landing Detector - Synthetic Demo", display_img)
        
        # Print status
        if result.status == "TARGET_ACQUIRED":
            print(f"🎯 [{pattern}] Target at {result.distance:.1f}m, Alt: {altitude:.1f}m")
        else:
            print(f"🔍 [{pattern}] Status: {result.status}, Alt: {altitude:.1f}m")
        
        # Handle keys
        key = cv2.waitKey(500) & 0xFF  # Wait 500ms between frames
        if key == ord('q'):
            break
        elif key == ord('n'):
            pattern_idx = (pattern_idx + 1) % len(test_patterns)
            altitude_idx = (altitude_idx + 1) % len(altitudes)
            detector.reset_state()
            print(f"🔄 Switched to pattern: {test_patterns[pattern_idx]}")
        elif key == ord('r'):
            detector.reset_state()
            altitudes = [8.0, 4.0, 2.0, 1.0]
            altitude_idx = 0
            print("🔄 Reset detector state")
        elif key == ord(' '):
            paused = not paused
            print(f"⏯️  {'Paused' if paused else 'Resumed'}")
    
    cv2.destroyAllWindows()

def run_performance_test():
    """Run performance benchmark."""
    
    print("⚡ Starting performance test...")
    
    # Initialize detector
    detector = UAVLandingDetector(
        model_path=None,  # Use placeholder mode
        enable_visualization=False  # Disable for pure performance
    )
    
    # Test parameters
    test_frames = 100
    image_sizes = [(320, 240), (640, 480), (1024, 768)]
    
    for width, height in image_sizes:
        print(f"\n📏 Testing {width}x{height}...")
        
        # Generate test images
        images = []
        for i in range(test_frames):
            img = create_test_image(width, height)
            images.append(img)
        
        # Warm up
        for i in range(5):
            detector.process_frame(images[0], altitude=5.0)
        
        # Benchmark
        start_time = time.time()
        
        for i, image in enumerate(images):
            altitude = 5.0 - (i / test_frames) * 4.0  # Simulate descent
            result = detector.process_frame(image, altitude=altitude)
        
        end_time = time.time()
        
        # Results
        total_time = end_time - start_time
        fps = test_frames / total_time
        avg_ms = (total_time / test_frames) * 1000
        
        print(f"✅ Results: {fps:.1f} FPS, {avg_ms:.1f} ms/frame")
    
    # Final stats
    stats = detector.get_performance_stats()
    print(f"\n📊 Final Stats: {stats}")

def main():
    """Main demo function."""
    
    print("🚁 UAV Landing Detector Demo")
    print("=" * 40)
    
    while True:
        print("\nSelect demo mode:")
        print("1. Webcam Demo")
        print("2. Synthetic Demo") 
        print("3. Performance Test")
        print("4. Convert PyTorch Model to ONNX")
        print("0. Exit")
        
        choice = input("\nEnter choice (0-4): ").strip()
        
        if choice == "1":
            if not run_webcam_demo():
                run_synthetic_demo()  # Fallback to synthetic
                
        elif choice == "2":
            run_synthetic_demo()
            
        elif choice == "3":
            run_performance_test()
            
        elif choice == "4":
            # Convert model
            pth_files = list(Path("../model_pths").glob("*.pth"))
            if not pth_files:
                print("❌ No .pth files found in ../model_pths/")
                continue
                
            print("\nAvailable models:")
            for i, pth_file in enumerate(pth_files):
                print(f"{i+1}. {pth_file.name}")
            
            try:
                idx = int(input("Select model (1-{}): ".format(len(pth_files)))) - 1
                if 0 <= idx < len(pth_files):
                    pth_path = str(pth_files[idx])
                    onnx_path = pth_path.replace(".pth", "_uav_landing.onnx")
                    
                    # Import and run conversion
                    from convert_to_onnx import convert_model
                    convert_model(pth_path, onnx_path)
                    
                    print(f"\n✅ Model converted! Update detector initialization:")
                    print(f"   detector = UAVLandingDetector(model_path='models/{onnx_path.split('/')[-1]}')")
                else:
                    print("❌ Invalid selection")
            except ValueError:
                print("❌ Invalid input")
            except Exception as e:
                print(f"❌ Conversion failed: {e}")
                
        elif choice == "0":
            break
            
        else:
            print("❌ Invalid choice")
    
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
