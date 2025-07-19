# visual_odometry.py
"""
Visual Odometry and Relative Positioning for GPS-Free UAV Navigation
Implements markerless visual SLAM techniques for relative position estimation.
"""

import cv2
import numpy as np
import math
from typing import Tuple, List, Dict, Optional
import config
from collections import deque


class VisualOdometry:
    """
    Visual odometry system for estimating camera motion and scale
    without GPS or external markers.
    """
    
    def __init__(self, camera_matrix: Optional[np.ndarray] = None, 
                 dist_coeffs: Optional[np.ndarray] = None):
        """
        Initialize visual odometry system.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix [fx 0 cx; 0 fy cy; 0 0 1]
            dist_coeffs: Distortion coefficients [k1, k2, p1, p2, k3]
        """
        # Camera calibration parameters (default values for typical camera)
        if camera_matrix is None:
            # Default camera matrix - should be calibrated for your specific camera
            self.camera_matrix = np.array([
                [800.0, 0.0, 320.0],    # fx, 0, cx
                [0.0, 800.0, 240.0],    # 0, fy, cy  
                [0.0, 0.0, 1.0]         # 0, 0, 1
            ])
        else:
            self.camera_matrix = camera_matrix
        
        if dist_coeffs is None:
            self.dist_coeffs = np.zeros(5)  # Assume no distortion
        else:
            self.dist_coeffs = dist_coeffs
        
        # Feature detector and matcher
        self.feature_detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Motion estimation
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        
        # Pose tracking
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z in meters
        self.rotation = np.eye(3)  # 3x3 rotation matrix
        self.scale = 1.0  # Current scale estimate
        
        # Motion history for smoothing
        self.motion_history = deque(maxlen=10)
        
        # Ground plane estimation
        self.ground_plane = None  # Will be estimated from features
        self.altitude_estimate = 10.0  # Initial altitude estimate in meters
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a new frame and update pose estimation.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Dictionary with motion estimation results
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
        
        motion_info = {
            'position': self.position.copy(),
            'altitude': self.altitude_estimate,
            'rotation': self.rotation.copy(),
            'scale': self.scale,
            'num_features': len(keypoints) if keypoints else 0,
            'motion_confidence': 0.0
        }
        
        if self.prev_frame is None:
            # First frame - just store it
            self.prev_frame = gray.copy()
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return motion_info
        
        if descriptors is None or self.prev_descriptors is None:
            return motion_info
        
        # Match features between frames
        matches = self.matcher.match(self.prev_descriptors, descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 10:  # Need minimum number of matches
            motion_info['motion_confidence'] = 0.0
            return motion_info
        
        # Extract matched points
        prev_pts = np.array([self.prev_keypoints[m.queryIdx].pt for m in matches])
        curr_pts = np.array([keypoints[m.trainIdx].pt for m in matches])
        
        # Estimate motion
        motion_estimate = self._estimate_motion(prev_pts, curr_pts)
        
        if motion_estimate is not None:
            # Update pose
            self._update_pose(motion_estimate)
            motion_info.update({
                'position': self.position.copy(),
                'altitude': self.altitude_estimate,
                'rotation': self.rotation.copy(),
                'scale': self.scale,
                'motion_confidence': motion_estimate['confidence']
            })
        
        # Store current frame for next iteration
        self.prev_frame = gray.copy()
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        
        return motion_info
    
    def _estimate_motion(self, prev_pts: np.ndarray, curr_pts: np.ndarray) -> Optional[Dict]:
        """Estimate camera motion from matched points."""
        if len(prev_pts) < 8:
            return None
        
        # Find essential matrix
        E, mask = cv2.findEssentialMat(
            prev_pts, curr_pts, self.camera_matrix, 
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        
        if E is None:
            return None
        
        # Recover pose from essential matrix
        _, R, t, mask = cv2.recoverPose(E, prev_pts, curr_pts, self.camera_matrix)
        
        # Calculate confidence based on inlier ratio
        inlier_ratio = np.sum(mask) / len(mask) if len(mask) > 0 else 0.0
        confidence = min(inlier_ratio * 2.0, 1.0)  # Scale to 0-1
        
        return {
            'rotation': R,
            'translation': t.flatten(),
            'confidence': confidence,
            'inliers': np.sum(mask)
        }
    
    def _update_pose(self, motion_estimate: Dict):
        """Update the current pose estimate."""
        R = motion_estimate['rotation']
        t = motion_estimate['translation']
        confidence = motion_estimate['confidence']
        
        # Scale translation by current altitude estimate (monocular scale ambiguity)
        scaled_translation = t * self.altitude_estimate * 0.1  # Scale factor
        
        # Update rotation (accumulate rotations)
        self.rotation = R @ self.rotation
        
        # Update position (in camera coordinates)
        self.position += self.rotation @ scaled_translation
        
        # Store motion for smoothing
        self.motion_history.append({
            'translation': scaled_translation,
            'rotation': R,
            'confidence': confidence
        })
    
    def estimate_altitude_from_features(self, frame: np.ndarray, seg_map: np.ndarray) -> float:
        """
        Estimate altitude using ground plane features and known object sizes.
        This helps resolve the scale ambiguity in monocular vision.
        """
        # Find ground features (safe landing surfaces)
        ground_mask = (seg_map == config.SAFE_LANDING_CLASS_ID).astype(np.uint8)
        
        # Detect features on the ground
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints = self.feature_detector.detect(gray, mask=ground_mask)
        
        if len(keypoints) < 10:
            return self.altitude_estimate  # Return current estimate
        
        # Assume typical ground features have certain sizes
        # This is a simplified approach - in practice, you'd use more sophisticated methods
        feature_sizes = [kp.size for kp in keypoints]
        avg_feature_size = np.mean(feature_sizes)
        
        # Empirical relationship between feature size and altitude
        # This would need calibration for your specific camera and environment
        estimated_altitude = max(5.0, min(50.0, 1000.0 / avg_feature_size))
        
        # Smooth the altitude estimate
        alpha = 0.1  # Smoothing factor
        self.altitude_estimate = alpha * estimated_altitude + (1 - alpha) * self.altitude_estimate
        
        return self.altitude_estimate
    
    def get_smoothed_motion(self) -> Dict:
        """Get smoothed motion estimate from recent history."""
        if not self.motion_history:
            return {
                'velocity': np.array([0.0, 0.0, 0.0]),
                'angular_velocity': np.array([0.0, 0.0, 0.0]),
                'confidence': 0.0
            }
        
        # Average recent motions weighted by confidence
        total_weight = 0.0
        avg_translation = np.zeros(3)
        avg_rotation = np.eye(3)
        
        for motion in self.motion_history:
            weight = motion['confidence']
            total_weight += weight
            avg_translation += motion['translation'] * weight
        
        if total_weight > 0:
            avg_translation /= total_weight
        
        avg_confidence = np.mean([m['confidence'] for m in self.motion_history])
        
        return {
            'velocity': avg_translation,
            'angular_velocity': np.array([0.0, 0.0, 0.0]),  # Simplified
            'confidence': avg_confidence
        }


class RelativePositioning:
    """
    Converts pixel coordinates to relative positions without GPS.
    Uses visual odometry and geometric projections.
    """
    
    def __init__(self, visual_odometry: VisualOdometry):
        self.vo = visual_odometry
        self.reference_position = None  # Will be set when landing zone is selected
        
    def pixel_to_relative_position(self, pixel_coords: Tuple[int, int], 
                                 altitude: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to relative position in meters.
        
        Args:
            pixel_coords: (x, y) pixel coordinates
            altitude: Current altitude above ground in meters
            
        Returns:
            (x, y) relative position in meters from camera center
        """
        x_pixel, y_pixel = pixel_coords
        
        # Get camera parameters from visual odometry
        camera_matrix = self.vo.camera_matrix
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1] 
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        
        # Normalize pixel coordinates
        x_norm = (x_pixel - cx) / fx
        y_norm = (y_pixel - cy) / fy
        
        # Project to ground plane using altitude
        x_meters = x_norm * altitude
        y_meters = y_norm * altitude
        
        return x_meters, y_meters
    
    def get_landing_vector(self, landing_zone_pixel: Tuple[int, int], 
                          altitude: float) -> Dict[str, float]:
        x_pixel, y_pixel = pixel_coords
        
        # Get camera parameters
        fx = self.vo.camera_matrix[0, 0]
        fy = self.vo.camera_matrix[1, 1]
        cx = self.vo.camera_matrix[0, 2]
        cy = self.vo.camera_matrix[1, 2]
        
        # Convert to normalized camera coordinates
        x_norm = (x_pixel - cx) / fx
        y_norm = (y_pixel - cy) / fy
        
        # Project to ground plane using altitude
        x_meters = x_norm * altitude
        y_meters = y_norm * altitude
        
        return x_meters, y_meters
    
    def get_landing_vector(self, landing_zone_pixel: Tuple[int, int], 
                          altitude: float) -> Dict:
        """
        Calculate the relative movement needed to reach the landing zone.
        
        Args:
            landing_zone_pixel: (x, y) pixel coordinates of landing zone center
            altitude: Current altitude estimate
            
        Returns:
            Dictionary with movement commands
        """
        # Convert to relative position
        dx, dy = self.pixel_to_relative_position(landing_zone_pixel, altitude)
        
        # Calculate distance and bearing
        distance = math.sqrt(dx**2 + dy**2)
        bearing = math.atan2(dy, dx)  # Radians from positive x-axis
        
        # Convert to UAV navigation commands
        # Positive x = forward, positive y = right in camera frame
        forward_command = dx  # meters
        right_command = dy    # meters
        
        return {
            'forward_meters': forward_command,
            'right_meters': right_command,
            'distance_meters': distance,
            'bearing_radians': bearing,
            'bearing_degrees': math.degrees(bearing),
            'altitude_meters': altitude,
            'confidence': self.vo.get_smoothed_motion()['confidence']
        }
    
    def set_reference_position(self, current_pixel: Tuple[int, int]):
        """Set the current position as reference for relative navigation."""
        self.reference_position = current_pixel
    
    def get_relative_displacement(self) -> Dict:
        """Get the displacement from the reference position."""
        if self.reference_position is None:
            return {
                'displacement_x': 0.0,
                'displacement_y': 0.0,
                'total_displacement': 0.0
            }
        
        current_pos = self.vo.position
        
        return {
            'displacement_x': current_pos[0],
            'displacement_y': current_pos[1], 
            'displacement_z': current_pos[2],
            'total_displacement': np.linalg.norm(current_pos)
        }
