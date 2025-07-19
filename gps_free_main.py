# gps_free_main.py
"""
GPS-Free UAV Landing System - Main Application
Integrates visual odometry, symbolic reasoning, and flight control
for markerless autonomous landing.
"""

import cv2
import numpy as np
import time
import argparse
import threading
from pathlib import Path
from typing import Dict, Optional

import config
from neural_engine import NeuralEngine
from symbolic_engine import SymbolicEngine
from visual_odometry import VisualOdometry, RelativePositioning
from flight_controller import MockFlightController, GPSFreeLandingController, FlightMode


class GPSFreeLandingSystem:
    """
    Complete GPS-free landing system that integrates all components.
    """
    
    def __init__(self, video_source=0, simulation_mode=True):
        """
        Initialize the GPS-free landing system.
        
        Args:
            video_source: Video source (0 for webcam, path for video file)
            simulation_mode: If True, use mock flight controller
        """
        self.video_source = video_source
        self.simulation_mode = simulation_mode
        
        print("Initializing GPS-Free UAV Landing System...")
        
        # Initialize core components
        print("1. Initializing Neural Engine...")
        self.neural_engine = NeuralEngine()
        
        print("2. Initializing Symbolic Engine...")
        self.symbolic_engine = SymbolicEngine()
        
        print("3. Initializing Visual Odometry...")
        camera_matrix = np.array(config.CAMERA_MATRIX)
        dist_coeffs = np.array(config.DISTORTION_COEFFICIENTS)
        self.visual_odometry = VisualOdometry(camera_matrix, dist_coeffs)
        
        print("4. Initializing Relative Positioning...")
        self.positioning = RelativePositioning(self.visual_odometry)
        
        print("5. Initializing Flight Controller...")
        if simulation_mode:
            self.flight_controller = MockFlightController(
                initial_altitude=config.INITIAL_ALTITUDE_ESTIMATE
            )
            self.flight_controller.start_simulation()
        else:
            # Here you would initialize your real flight controller
            # e.g., self.flight_controller = RealFlightController()
            raise NotImplementedError("Real flight controller not implemented yet")
        
        print("6. Initializing Landing Controller...")
        self.landing_controller = GPSFreeLandingController(self.flight_controller)
        
        # State tracking
        self.running = False
        self.landing_active = False
        self.frame_count = 0
        self.start_time = time.time()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        print("✅ GPS-Free Landing System initialized successfully!")
        print(f"🎥 Video source: {video_source}")
        print(f"🤖 Simulation mode: {simulation_mode}")
    
    def calibrate_camera(self):
        """
        Camera calibration routine for accurate visual odometry.
        This should be run once for each camera before deployment.
        """
        print("\n📷 Camera Calibration Mode")
        print("Show a checkerboard pattern to the camera from multiple angles...")
        print("Press 'c' to capture calibration images, 'q' to finish")
        
        # Calibration parameters
        CHECKERBOARD_SIZE = (9, 6)  # Inner corners
        calibration_images = []
        
        cap = cv2.VideoCapture(self.video_source)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners
            ret_corners, corners = cv2.findChessboardCorners(
                gray, CHECKERBOARD_SIZE, None
            )
            
            # Draw corners if found
            if ret_corners:
                cv2.drawChessboardCorners(frame, CHECKERBOARD_SIZE, corners, ret_corners)
                cv2.putText(frame, f"Pattern detected! Press 'c' to capture ({len(calibration_images)}/20)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"Show checkerboard pattern ({len(calibration_images)}/20)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Camera Calibration", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and ret_corners:
                calibration_images.append((gray.copy(), corners))
                print(f"Captured calibration image {len(calibration_images)}")
                
            elif key == ord('q') or len(calibration_images) >= 20:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(calibration_images) < 10:
            print("❌ Need at least 10 calibration images")
            return False
        
        # Perform calibration
        print(f"🔄 Calibrating camera with {len(calibration_images)} images...")
        
        # Prepare object points
        objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
        
        objpoints = [objp] * len(calibration_images)
        imgpoints = [corners for _, corners in calibration_images]
        
        # Calibrate
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        if ret:
            print("✅ Camera calibration successful!")
            print("Camera Matrix:")
            print(camera_matrix)
            print("Distortion Coefficients:")
            print(dist_coeffs)
            
            # Save calibration
            calib_file = "camera_calibration.npz"
            np.savez(calib_file, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
            print(f"📁 Calibration saved to {calib_file}")
            
            return True
        else:
            print("❌ Camera calibration failed")
            return False
    
    def start_landing_sequence(self):
        """Start the autonomous landing sequence."""
        if self.landing_active:
            print("⚠️ Landing sequence already active")
            return
        
        print("\n🚁 Starting GPS-Free Landing Sequence...")
        self.landing_active = True
        self.flight_controller.set_mode(FlightMode.SEARCH)
        
        # Set reference position
        state = self.flight_controller.get_state()
        print(f"📍 Reference altitude: {state.altitude_relative:.1f}m")
    
    def stop_landing_sequence(self):
        """Stop the landing sequence and hover."""
        if not self.landing_active:
            return
        
        print("\n⏹️ Stopping landing sequence - hovering in place")
        self.landing_active = False
        self.flight_controller.emergency_stop()
    
    def run(self):
        """Main application loop."""
        print(f"\n🚀 Starting GPS-Free Landing System")
        print("Controls:")
        print("  'l' - Start landing sequence")
        print("  's' - Stop landing sequence")
        print("  'r' - Reset visual odometry")
        print("  'c' - Camera calibration mode")
        print("  'p' - Print performance stats")
        print("  'q' - Quit")
        print("\n")
        
        # Initialize video capture
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video source {self.video_source}")
            return
        
        # Set camera properties if using webcam
        if isinstance(self.video_source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_RESOLUTION[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RESOLUTION[1])
            cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.running = True
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(self.video_source, int):
                        print("❌ Lost camera connection")
                        break
                    else:
                        print("📹 End of video file")
                        # For video files, loop back to start
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                
                self.frame_count += 1
                self.fps_counter += 1
                
                # Update FPS counter
                current_time = time.time()
                if current_time - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
                    self.fps_counter = 0
                    self.fps_start_time = current_time
                
                # Process frame through the complete pipeline
                start_time = time.time()
                
                # 1. Neural network processing
                seg_map = self.neural_engine.process_frame(frame)
                
                # 2. Visual odometry processing
                motion_info = self.visual_odometry.process_frame(frame)
                
                # 3. Update altitude estimate from features
                estimated_altitude = self.visual_odometry.estimate_altitude_from_features(frame, seg_map)
                motion_info['altitude'] = estimated_altitude
                
                # 4. Symbolic reasoning
                decision = self.symbolic_engine.run(seg_map)
                
                # 5. Landing control (if active)
                if self.landing_active:
                    continue_landing = self.landing_controller.process_landing_decision(
                        decision, motion_info, self.positioning
                    )
                    if not continue_landing:
                        print("🎯 Landing sequence completed!")
                        self.landing_active = False
                
                processing_time = time.time() - start_time
                
                # 6. Visualization
                display_frame = self.visualize_complete_system(
                    frame, seg_map, decision, motion_info, processing_time
                )
                
                cv2.imshow("GPS-Free UAV Landing System", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("👋 Quit requested")
                    break
                elif key == ord('l'):
                    self.start_landing_sequence()
                elif key == ord('s'):
                    self.stop_landing_sequence()
                elif key == ord('r'):
                    self.visual_odometry = VisualOdometry(
                        np.array(config.CAMERA_MATRIX), 
                        np.array(config.DISTORTION_COEFFICIENTS)
                    )
                    print("🔄 Visual odometry reset")
                elif key == ord('c'):
                    cv2.destroyAllWindows()
                    self.calibrate_camera()
                elif key == ord('p'):
                    self.print_performance_stats()
                
                # Print periodic status
                if config.DEBUG_MODE and self.frame_count % 60 == 0:
                    self.print_status_update(decision, motion_info, processing_time)
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        
        finally:
            self.cleanup()
    
    def visualize_complete_system(self, frame: np.ndarray, seg_map: np.ndarray, 
                                decision: Dict, motion_info: Dict, processing_time: float) -> np.ndarray:
        """Create comprehensive visualization of the entire system."""
        # Resize frame to match segmentation if needed
        if frame.shape[:2] != seg_map.shape:
            display_frame = cv2.resize(frame, config.INPUT_RESOLUTION)
        else:
            display_frame = frame.copy()
        
        # Create segmentation overlay
        overlay = np.zeros_like(display_frame)
        class_colors = {
            0: (0, 0, 0),        # background - black
            1: (0, 255, 0),      # safe_flat_surface - green
            2: (255, 255, 0),    # unsafe_uneven_surface - yellow
            3: (255, 165, 0),    # low_obstacle - orange
            4: (0, 0, 255),      # high_obstacle - red
        }
        
        for class_id, color in class_colors.items():
            mask = (seg_map == class_id)
            overlay[mask] = color
        
        # Blend overlay
        alpha = 0.4
        blended = cv2.addWeighted(display_frame, 1-alpha, overlay, alpha, 0)
        
        # Draw landing zone if detected
        if decision['status'] == 'TARGET_ACQUIRED':
            zone = decision['zone']
            center = zone['center']
            
            # Landing zone visualization
            cv2.circle(blended, center, 15, (0, 255, 255), 3)
            cv2.circle(blended, center, 5, (0, 255, 255), -1)
            
            # Landing vector
            if self.landing_active:
                landing_vector = self.positioning.get_landing_vector(
                    center, motion_info['altitude']
                )
                
                # Draw movement arrow
                arrow_length = min(50, int(landing_vector['distance_meters'] * 10))
                angle = landing_vector['bearing_radians']
                end_point = (
                    int(center[0] + arrow_length * np.cos(angle)),
                    int(center[1] + arrow_length * np.sin(angle))
                )
                cv2.arrowedLine(blended, center, end_point, (255, 255, 0), 3)
        
        # System status overlay
        self.draw_system_status(blended, decision, motion_info, processing_time)
        
        return blended
    
    def draw_system_status(self, frame: np.ndarray, decision: Dict, 
                          motion_info: Dict, processing_time: float):
        """Draw comprehensive system status on the frame."""
        y_offset = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Flight status
        if hasattr(self, 'flight_controller'):
            state = self.flight_controller.get_state()
            status_color = (0, 255, 0) if self.landing_active else (255, 255, 255)
            
            status_text = f"Mode: {state.mode.value.upper()}"
            if self.landing_active:
                status_text += " - LANDING ACTIVE"
            
            cv2.putText(frame, status_text, (10, y_offset), 
                       font, font_scale, status_color, thickness)
            y_offset += 25
            
            cv2.putText(frame, f"Altitude: {state.altitude_relative:.1f}m", 
                       (10, y_offset), font, font_scale, (255, 255, 255), 1)
            y_offset += 20
        
        # Visual odometry status
        vo_color = (0, 255, 0) if motion_info['motion_confidence'] > 0.5 else (0, 255, 255)
        cv2.putText(frame, f"VO Confidence: {motion_info['motion_confidence']:.2f}", 
                   (10, y_offset), font, font_scale-0.1, vo_color, 1)
        y_offset += 20
        
        cv2.putText(frame, f"Features: {motion_info['num_features']}", 
                   (10, y_offset), font, font_scale-0.1, (255, 255, 255), 1)
        y_offset += 20
        
        # Landing decision status
        decision_color = (0, 255, 0) if decision['status'] == 'TARGET_ACQUIRED' else (0, 0, 255)
        cv2.putText(frame, f"Decision: {decision['status']}", 
                   (10, y_offset), font, font_scale-0.1, decision_color, 1)
        y_offset += 20
        
        # Performance info
        cv2.putText(frame, f"FPS: {self.current_fps:.1f} | Process: {processing_time*1000:.1f}ms", 
                   (10, frame.shape[0]-10), font, 0.5, (0, 255, 255), 1)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", 
                   (frame.shape[1]-120, 25), font, 0.5, (255, 255, 255), 1)
    
    def print_status_update(self, decision: Dict, motion_info: Dict, processing_time: float):
        """Print detailed status update."""
        state = self.flight_controller.get_state()
        
        print(f"\n--- Frame {self.frame_count} Status ---")
        print(f"Flight Mode: {state.mode.value}")
        print(f"Altitude: {state.altitude_relative:.1f}m")
        print(f"Decision: {decision['status']}")
        print(f"VO Confidence: {motion_info['motion_confidence']:.2f}")
        print(f"Features: {motion_info['num_features']}")
        print(f"Processing: {processing_time*1000:.1f}ms")
        
        if decision['status'] == 'TARGET_ACQUIRED':
            zone = decision['zone']
            landing_vector = self.positioning.get_landing_vector(
                zone['center'], motion_info['altitude']
            )
            print(f"Target Distance: {landing_vector['distance_meters']:.1f}m")
            print(f"Target Bearing: {landing_vector['bearing_degrees']:.0f}°")
    
    def print_performance_stats(self):
        """Print comprehensive performance statistics."""
        neural_stats = self.neural_engine.get_performance_stats()
        symbolic_stats = self.symbolic_engine.get_performance_stats()
        vo_motion = self.visual_odometry.get_smoothed_motion()
        
        print("\n" + "="*50)
        print("PERFORMANCE STATISTICS")
        print("="*50)
        print(f"Neural Engine FPS: {neural_stats['fps']:.1f}")
        print(f"Symbolic Engine FPS: {symbolic_stats['fps']:.1f}")
        print(f"Overall System FPS: {self.current_fps:.1f}")
        print(f"Visual Odometry Confidence: {vo_motion['confidence']:.2f}")
        print(f"Total Frames Processed: {self.frame_count}")
        print(f"Runtime: {time.time() - self.start_time:.1f}s")
        print("="*50)
    
    def cleanup(self):
        """Clean up resources."""
        print("🧹 Cleaning up resources...")
        
        if hasattr(self, 'flight_controller') and self.simulation_mode:
            self.flight_controller.emergency_stop()
            self.flight_controller.stop_simulation()
        
        cv2.destroyAllWindows()
        print("✅ Cleanup completed")


def main():
    """Entry point for GPS-free landing system."""
    parser = argparse.ArgumentParser(description="GPS-Free UAV Landing System")
    parser.add_argument("--video", "-v", default=0,
                       help="Video source (0 for webcam, path for video file)")
    parser.add_argument("--calibrate", "-c", action="store_true",
                       help="Run camera calibration first")
    parser.add_argument("--real-fc", action="store_true",
                       help="Use real flight controller (not implemented)")
    
    args = parser.parse_args()
    
    # Convert video argument to int if it's a digit
    try:
        video_source = int(args.video)
    except ValueError:
        video_source = args.video
    
    # Initialize system
    simulation_mode = not args.real_fc
    system = GPSFreeLandingSystem(video_source=video_source, simulation_mode=simulation_mode)
    
    if args.calibrate:
        print("🎯 Running camera calibration...")
        if system.calibrate_camera():
            print("✅ Calibration complete. Restart without --calibrate flag.")
        else:
            print("❌ Calibration failed.")
        return
    
    # Run the system
    system.run()


if __name__ == "__main__":
    main()
