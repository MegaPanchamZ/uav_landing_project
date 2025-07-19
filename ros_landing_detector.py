#!/usr/bin/env python3
"""
ROS-Compatible UAV Landing Zone Detector
Integrates semantic segmentation with neuro-symbolic reasoning for single-frame analysis.

Usage in ROS:
    detector = ROSLandingDetector(model_path="models/bisenetv2_final.onnx")
    result = detector.process_frame(cv_image, altitude=5.0)
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Import our existing components
from neural_engine import NeuralEngine
from symbolic_engine import SymbolicEngine, LandingZone
from visual_odometry import RelativePositioning
import config

@dataclass
class ROSLandingResult:
    """Result structure for ROS compatibility"""
    # Decision info
    status: str  # 'TARGET_ACQUIRED', 'NO_TARGET', 'UNSAFE'
    confidence: float  # 0.0 - 1.0
    
    # Landing zone info (if found)
    landing_zone: Optional[LandingZone] = None
    pixel_center: Optional[Tuple[int, int]] = None
    world_position: Optional[Tuple[float, float]] = None  # meters from camera
    distance_to_target: Optional[float] = None  # meters
    bearing_to_target: Optional[float] = None  # radians
    
    # Movement commands
    forward_velocity: float = 0.0  # m/s
    right_velocity: float = 0.0   # m/s  
    yaw_rate: float = 0.0         # rad/s
    descent_rate: float = 0.0     # m/s (positive = down)
    
    # Debug info
    processing_time: float = 0.0   # milliseconds
    num_zones_detected: int = 0
    segmentation_quality: float = 0.0
    
    # Visualization data for ROS
    annotated_image: Optional[np.ndarray] = None
    segmentation_mask: Optional[np.ndarray] = None

class ROSLandingDetector:
    """
    Production-ready landing zone detector for ROS integration.
    
    Combines semantic segmentation (BiSeNetV2) with symbolic reasoning
    for robust landing zone detection from single camera frames.
    """
    
    def __init__(self, 
                 model_path: str = "models/bisenetv2_udd6_final.onnx",
                 camera_matrix: Optional[np.ndarray] = None,
                 enable_visualization: bool = True,
                 safety_mode: bool = True):
        """
        Initialize the ROS-compatible landing detector.
        
        Args:
            model_path: Path to trained ONNX model
            camera_matrix: Camera intrinsic matrix (3x3)
            enable_visualization: Generate annotated images for ROS visualization
            safety_mode: Enable conservative safety checks
        """
        self.model_path = Path(model_path)
        self.enable_visualization = enable_visualization
        self.safety_mode = safety_mode
        
        # Initialize components
        self.neural_engine = NeuralEngine(str(self.model_path))
        self.symbolic_engine = SymbolicEngine()
        
        # Set up camera parameters
        if camera_matrix is not None:
            self.camera_matrix = camera_matrix
        else:
            # Use default from config
            self.camera_matrix = np.array(config.CAMERA_MATRIX, dtype=np.float32)
        
        # Initialize visual odometry for relative positioning
        from visual_odometry import VisualOdometry
        self.visual_odometry = VisualOdometry(
            camera_matrix=self.camera_matrix,
            dist_coeffs=np.array(config.DISTORTION_COEFFICIENTS, dtype=np.float32)
        )
        self.relative_positioning = RelativePositioning(self.visual_odometry)
        
        # Performance tracking
        self.frame_count = 0
        self.total_processing_time = 0.0
        
        # Landing phase state
        self.current_phase = "SEARCH"  # SEARCH, APPROACH, PRECISION, LANDING
        self.target_lock_frames = 0
        self.last_good_target = None
        
        print(f"🚁 ROSLandingDetector initialized")
        print(f"   Model: {self.model_path}")
        print(f"   Camera Matrix: {self.camera_matrix[0,0]:.1f}fx, {self.camera_matrix[1,1]:.1f}fy")
        print(f"   Visualization: {enable_visualization}")
        print(f"   Safety Mode: {safety_mode}")
        
    def process_frame(self, 
                     image: np.ndarray, 
                     altitude: float = 5.0,
                     current_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> ROSLandingResult:
        """
        Process a single camera frame to detect landing zones.
        
        Args:
            image: Input camera frame (BGR format)
            altitude: Current altitude above ground (meters)
            current_velocity: Current velocity [vx, vy, vz] in m/s
            
        Returns:
            ROSLandingResult with detection results and movement commands
        """
        start_time = time.time()
        
        # Validate inputs
        if image is None or image.size == 0:
            return self._create_error_result("Invalid input image", start_time)
            
        if altitude <= 0:
            return self._create_error_result("Invalid altitude", start_time) 
            
        try:
            # Step 1: Semantic Segmentation
            segmentation_result = self.neural_engine.process_frame(image)
            
            if not segmentation_result['success']:
                return self._create_error_result("Neural processing failed", start_time)
                
            # Step 2: Symbolic Reasoning
            decision = self.symbolic_engine.run(
                segmentation_result.get('mask', np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8))
            )
            
            # Step 3: Generate Movement Commands
            result = self._generate_landing_commands(
                decision, image, altitude, current_velocity, start_time
            )
            
            # Step 4: Update state tracking
            self._update_phase_state(result, altitude)
            
            # Step 5: Apply safety overrides
            if self.safety_mode:
                result = self._apply_safety_checks(result, altitude)
                
            # Step 6: Generate visualization if enabled
            if self.enable_visualization:
                result.annotated_image = self._create_visualization(
                    image, segmentation_result, result
                )
                result.segmentation_mask = segmentation_result.get('mask')
                
            # Update performance metrics
            self.frame_count += 1
            processing_time_ms = (time.time() - start_time) * 1000
            self.total_processing_time += processing_time_ms
            result.processing_time = processing_time_ms
            
            return result
            
        except Exception as e:
            print(f"❌ Error in process_frame: {e}")
            return self._create_error_result(f"Processing exception: {e}", start_time)
            
    def _generate_landing_commands(self, 
                                  decision: Dict,
                                  image: np.ndarray,
                                  altitude: float,
                                  current_velocity: Tuple[float, float, float],
                                  start_time: float) -> ROSLandingResult:
        """Generate movement commands based on detection results."""
        
        result = ROSLandingResult(
            status="NO_TARGET",
            confidence=0.0,
            processing_time=0.0
        )
        
        if decision['status'] == 'TARGET_ACQUIRED' and decision.get('zone'):
            zone = decision['zone']
            result.status = 'TARGET_ACQUIRED'
            result.confidence = decision.get('confidence', 0.0)
            result.landing_zone = zone
            result.num_zones_detected = decision.get('num_zones', 0)
            
            # Get pixel coordinates of landing zone center
            center_pixel = (int(zone.center_x), int(zone.center_y))
            result.pixel_center = center_pixel
            
            # Convert to world coordinates
            try:
                world_pos = self.relative_positioning.pixel_to_relative_position(
                    center_pixel, altitude
                )
                result.world_position = world_pos
                
                # Calculate distance and bearing
                result.distance_to_target = np.sqrt(world_pos[0]**2 + world_pos[1]**2)
                result.bearing_to_target = np.arctan2(world_pos[1], world_pos[0])
                
                # Generate movement commands based on phase
                result = self._calculate_movement_commands(
                    result, world_pos, altitude, current_velocity
                )
                
            except Exception as e:
                print(f"⚠️ Position calculation error: {e}")
                result.confidence *= 0.5  # Reduce confidence on positioning error
                
        elif decision['status'] == 'UNSAFE':
            result.status = 'UNSAFE'
            result.confidence = decision.get('confidence', 0.0)
            # Emergency hover commands
            result.forward_velocity = 0.0
            result.right_velocity = 0.0
            result.descent_rate = 0.0
            result.yaw_rate = 0.0
            
        return result
        
    def _calculate_movement_commands(self,
                                   result: ROSLandingResult,
                                   world_pos: Tuple[float, float],
                                   altitude: float,
                                   current_velocity: Tuple[float, float, float]) -> ROSLandingResult:
        """Calculate movement commands based on current phase and target position."""
        
        dx, dy = world_pos  # meters from camera center
        distance = result.distance_to_target
        
        # Phase-based control gains
        if self.current_phase == "SEARCH" and altitude > config.APPROACH_ALTITUDE:
            # Search phase: larger movements allowed
            max_velocity = config.MAX_APPROACH_SPEED
            position_gain = config.POSITION_CONTROL_GAIN * 1.2
            descent_rate = config.SEARCH_DESCENT_RATE
            
        elif self.current_phase == "APPROACH" and altitude > config.PRECISION_ALTITUDE:
            # Approach phase: moderate movements
            max_velocity = config.MAX_APPROACH_SPEED * 0.8
            position_gain = config.POSITION_CONTROL_GAIN
            descent_rate = config.APPROACH_DESCENT_RATE
            
        elif self.current_phase == "PRECISION" and altitude > config.LANDING_THRESHOLD:
            # Precision phase: small, careful movements
            max_velocity = config.MAX_PRECISION_SPEED
            position_gain = config.POSITION_CONTROL_GAIN * 0.6
            descent_rate = config.PRECISION_DESCENT_RATE
            
        else:
            # Final landing phase: minimal movement
            max_velocity = config.MAX_PRECISION_SPEED * 0.3
            position_gain = config.POSITION_CONTROL_GAIN * 0.3
            descent_rate = config.FINAL_DESCENT_RATE
            
        # Calculate desired velocities
        result.forward_velocity = np.clip(dx * position_gain, -max_velocity, max_velocity)
        result.right_velocity = np.clip(dy * position_gain, -max_velocity, max_velocity)
        
        # Descent rate based on distance to target and altitude
        if distance < config.LANDING_THRESHOLD and altitude > config.MIN_SAFE_ALTITUDE:
            result.descent_rate = descent_rate
        else:
            result.descent_rate = 0.0  # Hover until positioned
            
        # Yaw correction (keep minimal for stability)
        if abs(result.bearing_to_target) > np.radians(10):  # 10 degree threshold
            result.yaw_rate = np.clip(result.bearing_to_target * 0.2, -0.5, 0.5)
        else:
            result.yaw_rate = 0.0
            
        return result
        
    def _update_phase_state(self, result: ROSLandingResult, altitude: float):
        """Update landing phase based on current state."""
        
        if result.status == 'TARGET_ACQUIRED':
            self.target_lock_frames += 1
            self.last_good_target = result.world_position
            
            # Phase transitions based on altitude and target lock
            if altitude > config.APPROACH_ALTITUDE:
                self.current_phase = "SEARCH"
            elif altitude > config.PRECISION_ALTITUDE and self.target_lock_frames > 10:
                self.current_phase = "APPROACH"  
            elif altitude > config.LANDING_THRESHOLD and self.target_lock_frames > 20:
                self.current_phase = "PRECISION"
            elif self.target_lock_frames > 30:
                self.current_phase = "LANDING"
                
        else:
            # Target lost
            if self.target_lock_frames > 0:
                self.target_lock_frames -= 1
                
            # Return to search if target lost for too long
            if self.target_lock_frames == 0:
                self.current_phase = "SEARCH"
                self.last_good_target = None
                
    def _apply_safety_checks(self, result: ROSLandingResult, altitude: float) -> ROSLandingResult:
        """Apply safety overrides and limits."""
        
        # Altitude safety check
        if altitude < config.MIN_SAFE_ALTITUDE:
            print("⚠️ Altitude safety override - stopping descent")
            result.descent_rate = 0.0
            result.status = "UNSAFE"
            result.confidence *= 0.5
            
        # Maximum altitude check  
        if altitude > config.MAX_FLIGHT_ALTITUDE:
            print("⚠️ Maximum altitude exceeded")
            result.descent_rate = max(result.descent_rate, 0.5)  # Force descent
            
        # Velocity limits
        max_safe_vel = config.MAX_APPROACH_SPEED if altitude > 2.0 else config.MAX_PRECISION_SPEED
        
        result.forward_velocity = np.clip(result.forward_velocity, -max_safe_vel, max_safe_vel)
        result.right_velocity = np.clip(result.right_velocity, -max_safe_vel, max_safe_vel)
        result.yaw_rate = np.clip(result.yaw_rate, -1.0, 1.0)  # Max 1 rad/s yaw
        
        # Confidence-based speed reduction
        confidence_factor = max(result.confidence, 0.3)  # Minimum 30% speed
        result.forward_velocity *= confidence_factor
        result.right_velocity *= confidence_factor
        result.descent_rate *= confidence_factor
        
        return result
        
    def _create_visualization(self,
                            image: np.ndarray,
                            segmentation_result: Dict,
                            result: ROSLandingResult) -> np.ndarray:
        """Create annotated visualization for ROS display."""
        
        vis_image = image.copy()
        
        # Draw segmentation overlay if available
        if segmentation_result.get('mask') is not None:
            mask = segmentation_result['mask']
            colored_mask = self._colorize_mask(mask)
            vis_image = cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0)
            
        # Draw landing zone if detected
        if result.landing_zone and result.pixel_center:
            zone = result.landing_zone
            center = result.pixel_center
            
            # Draw zone boundaries
            cv2.rectangle(vis_image, 
                         (int(zone.min_x), int(zone.min_y)),
                         (int(zone.max_x), int(zone.max_y)),
                         (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(vis_image, center, 10, (0, 255, 0), -1)
            cv2.circle(vis_image, center, 15, (255, 255, 255), 2)
            
            # Draw distance and bearing info
            if result.world_position:
                info_text = f"Dist: {result.distance_to_target:.1f}m"
                cv2.putText(vis_image, info_text, (center[0] + 20, center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                           
        # Draw status info
        status_color = (0, 255, 0) if result.status == 'TARGET_ACQUIRED' else (0, 0, 255)
        status_text = f"{result.status} ({result.confidence:.2f})"
        cv2.putText(vis_image, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                   
        # Draw phase info
        phase_text = f"Phase: {self.current_phase}"
        cv2.putText(vis_image, phase_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                   
        # Draw velocity commands
        vel_text = f"Cmd: [{result.forward_velocity:.1f}, {result.right_velocity:.1f}, {result.descent_rate:.1f}]"
        cv2.putText(vis_image, vel_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                   
        return vis_image
        
    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert segmentation mask to colored visualization."""
        
        # Color map for different classes
        colors = {
            0: [0, 0, 0],        # Background - black
            1: [0, 255, 0],      # Suitable - green
            2: [255, 255, 0],    # Marginal - yellow
            3: [255, 0, 0],      # Obstacles - red
            4: [128, 0, 128],    # Unsafe - purple
            5: [128, 128, 128]   # Unknown - gray
        }
        
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        for class_id, color in colors.items():
            colored_mask[mask == class_id] = color
            
        return colored_mask
        
    def _create_error_result(self, error_msg: str, start_time: float) -> ROSLandingResult:
        """Create error result for exception handling."""
        
        print(f"❌ {error_msg}")
        
        return ROSLandingResult(
            status="ERROR",
            confidence=0.0,
            processing_time=(time.time() - start_time) * 1000
        )
        
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for monitoring."""
        
        if self.frame_count == 0:
            return {"avg_processing_time_ms": 0.0, "frame_rate": 0.0}
            
        avg_time = self.total_processing_time / self.frame_count
        frame_rate = 1000.0 / avg_time if avg_time > 0 else 0.0
        
        return {
            "avg_processing_time_ms": avg_time,
            "frame_rate": frame_rate,
            "total_frames": self.frame_count,
            "current_phase": self.current_phase,
            "target_lock_frames": self.target_lock_frames
        }
        
    def reset_state(self):
        """Reset internal state for new flight."""
        
        self.current_phase = "SEARCH"
        self.target_lock_frames = 0
        self.last_good_target = None
        self.frame_count = 0
        self.total_processing_time = 0.0
        
        print("🔄 Landing detector state reset")

# ROS Integration Example
if __name__ == "__main__":
    """
    Example usage for ROS integration testing.
    In actual ROS node, this would be called from image callback.
    """
    
    # Initialize detector
    detector = ROSLandingDetector(
        model_path="models/bisenetv2_udd6_final.onnx",  # Will use placeholder if not found
        enable_visualization=True,
        safety_mode=True
    )
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        exit(1)
        
    print("🎥 Starting ROS detector test (press 'q' to quit)")
    
    altitude = 5.0  # meters
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame (this is what ROS callback would do)
        result = detector.process_frame(frame, altitude=altitude)
        
        # Print result (in ROS, this would be published to topics)
        if result.status == 'TARGET_ACQUIRED':
            print(f"🎯 Target: {result.distance_to_target:.1f}m @ {np.degrees(result.bearing_to_target):.1f}°")
            print(f"   Commands: [{result.forward_velocity:.1f}, {result.right_velocity:.1f}, {result.descent_rate:.1f}]")
            
        # Show visualization
        if result.annotated_image is not None:
            cv2.imshow("ROS Landing Detector", result.annotated_image)
            
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset_state()
            print("🔄 State reset")
            
        # Simulate altitude change for testing
        if result.status == 'TARGET_ACQUIRED' and result.distance_to_target < 1.0:
            altitude = max(1.0, altitude - 0.1)  # Simulate descent
            
    cap.release()
    cv2.destroyAllWindows()
    
    # Print performance stats
    stats = detector.get_performance_stats()
    print(f"\n📊 Performance Stats:")
    print(f"   Average processing: {stats['avg_processing_time_ms']:.1f}ms")
    print(f"   Frame rate: {stats['frame_rate']:.1f} FPS")
    print(f"   Total frames: {stats['total_frames']}")
