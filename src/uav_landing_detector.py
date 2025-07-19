#!/usr/bin/env python3
"""
UAV Landing Zone Detector - Single Class Implementation
Real-time semantic segmentation + neuro-symbolic reasoning for UAV landing

Usage:
    detector = UAVLandingDetector()
    result = detector.process_frame(image, altitude=5.0)
"""

import cv2
import numpy as np
import time
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX Runtime not available, running in placeholder mode")

@dataclass
class LandingResult:
    """Complete landing detection result"""
    # Core detection
    status: str  # 'TARGET_ACQUIRED', 'NO_TARGET', 'UNSAFE'
    confidence: float  # 0.0-1.0
    
    # Target information
    target_pixel: Optional[Tuple[int, int]] = None  # (x, y) in image
    target_world: Optional[Tuple[float, float]] = None  # (x, y) in meters
    distance: Optional[float] = None  # meters from camera
    bearing: Optional[float] = None  # radians
    
    # Movement commands
    forward_velocity: float = 0.0  # m/s (positive = forward)
    right_velocity: float = 0.0   # m/s (positive = right)
    descent_rate: float = 0.0     # m/s (positive = down)
    yaw_rate: float = 0.0         # rad/s (positive = clockwise)
    
    # Performance metrics
    processing_time: float = 0.0  # milliseconds
    fps: float = 0.0
    
    # Visualization (optional)
    annotated_image: Optional[np.ndarray] = None

class UAVLandingDetector:
    """
    High-performance UAV landing zone detector.
    
    Combines BiSeNetV2 semantic segmentation with rule-based reasoning
    for real-time landing zone detection and navigation commands.
    """
    
    def __init__(self, 
                 model_path: str = None,
                 camera_matrix: Optional[np.ndarray] = None,
                 enable_visualization: bool = False,
                 device: str = "auto"):
        """
        Initialize the landing detector.
        
        Args:
            model_path: Path to ONNX model (uses placeholder if None)
            camera_matrix: Camera intrinsic matrix [fx,0,cx; 0,fy,cy; 0,0,1]
            enable_visualization: Generate annotated output images
            device: 'cpu', 'cuda', or 'auto'
        """
        self.enable_visualization = enable_visualization
        self.model_path = model_path
        
        # Camera parameters (default values, should be calibrated)
        if camera_matrix is not None:
            self.camera_matrix = camera_matrix
        else:
            self.camera_matrix = np.array([
                [800.0, 0.0, 320.0],    # fx, 0, cx
                [0.0, 800.0, 240.0],    # 0, fy, cy
                [0.0, 0.0, 1.0]         # 0, 0, 1
            ], dtype=np.float32)
        
        # Extract camera parameters
        self.fx = self.camera_matrix[0, 0]
        self.fy = self.camera_matrix[1, 1]
        self.cx = self.camera_matrix[0, 2]
        self.cy = self.camera_matrix[1, 2]
        
        # Model configuration
        self.input_size = (512, 512)  # Standard input size
        self.num_classes = 6
        
        # Class definitions for landing zones
        self.classes = {
            0: "background",    # Non-relevant areas
            1: "suitable",      # Perfect for landing (flat, clear)
            2: "marginal",      # Potentially suitable (low grass, rough ground)
            3: "obstacles",     # Buildings, structures, trees
            4: "unsafe",        # Water, vehicles, steep slopes
            5: "unknown"        # Uncertain areas
        }
        
        # Control parameters
        self.max_velocity = 2.0      # m/s maximum movement speed
        self.position_gain = 0.8     # Position control gain
        self.min_zone_area = 1000    # Minimum pixels for valid landing zone
        self.safety_margin = 2.0     # Safety distance from obstacles (meters)
        
        # Performance tracking
        self.frame_times = []
        self.max_history = 30
        
        # Landing phase tracking
        self.landing_phase = "SEARCH"  # SEARCH, APPROACH, PRECISION, LANDING
        self.target_lock_count = 0
        self.last_target = None
        
        # Initialize neural network
        self._initialize_model(device)
        
        print(f"🚁 UAV Landing Detector initialized")
        print(f"   Model: {'ONNX' if self.session else 'Placeholder'}")
        print(f"   Input size: {self.input_size}")
        print(f"   Camera: fx={self.fx:.0f}, fy={self.fy:.0f}")
        print(f"   Visualization: {enable_visualization}")
        
    def _initialize_model(self, device: str):
        """Initialize the ONNX model for inference."""
        self.session = None
        
        if not ONNX_AVAILABLE or not self.model_path or not Path(self.model_path).exists():
            print("⚠️  Using placeholder mode (no ONNX model)")
            return
            
        try:
            # Set up execution providers
            providers = []
            if device == "auto":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            elif device == "cuda":
                providers = ['CUDAExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
                
            # Create session
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Get input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            print(f"✅ Model loaded: {Path(self.model_path).name}")
            print(f"   Provider: {self.session.get_providers()[0]}")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            self.session = None
    
    def process_frame(self, 
                     image: np.ndarray, 
                     altitude: float,
                     current_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> LandingResult:
        """
        Process single frame for landing zone detection.
        
        Args:
            image: Input BGR image from camera
            altitude: Current altitude above ground (meters)
            current_velocity: Current velocity [vx, vy, vz] (m/s)
            
        Returns:
            LandingResult with detection and navigation commands
        """
        start_time = time.time()
        
        # Validate inputs
        if image is None or image.size == 0:
            return LandingResult(status="ERROR", confidence=0.0)
        if altitude <= 0:
            return LandingResult(status="ERROR", confidence=0.0)
            
        try:
            # Step 1: Semantic Segmentation
            segmentation_map = self._run_segmentation(image)
            
            # Step 2: Find Landing Zones
            zones = self._find_landing_zones(segmentation_map)
            
            # Step 3: Evaluate Best Zone
            result = self._evaluate_zones(zones, segmentation_map, image, altitude)
            
            # Step 4: Generate Navigation Commands
            result = self._generate_commands(result, altitude, current_velocity)
            
            # Step 5: Update State Tracking
            self._update_tracking(result)
            
            # Step 6: Performance Tracking
            processing_time = (time.time() - start_time) * 1000
            result.processing_time = processing_time
            result.fps = self._update_fps(processing_time)
            
            # Step 7: Visualization (if enabled)
            if self.enable_visualization:
                result.annotated_image = self._create_visualization(image, segmentation_map, result)
                
            return result
            
        except Exception as e:
            print(f"❌ Processing error: {e}")
            return LandingResult(status="ERROR", confidence=0.0, 
                               processing_time=(time.time() - start_time) * 1000)
    
    def _run_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Run semantic segmentation on input image."""
        
        if self.session is None:
            # Placeholder segmentation for testing
            h, w = image.shape[:2]
            seg_map = np.zeros((h, w), dtype=np.uint8)
            
            # Create some synthetic landing zones
            center_x, center_y = w // 2, h // 2
            
            # Main landing zone (suitable)
            cv2.rectangle(seg_map, 
                         (center_x - 60, center_y - 40), 
                         (center_x + 60, center_y + 40), 1, -1)
            
            # Some obstacles
            cv2.rectangle(seg_map, (50, 50), (100, 150), 3, -1)
            cv2.circle(seg_map, (w - 80, 80), 30, 3, -1)
            
            # Marginal areas
            cv2.rectangle(seg_map, (center_x - 100, center_y + 80), 
                         (center_x + 100, center_y + 120), 2, -1)
            
            return seg_map
        
        # Real model inference
        # Preprocess image
        input_tensor = self._preprocess_image(image)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        
        # Postprocess output
        segmentation_map = self._postprocess_segmentation(outputs[0], image.shape[:2])
        
        return segmentation_map
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for neural network input."""
        
        # Resize to input size
        resized = cv2.resize(image, self.input_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0,1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        
        # Convert HWC to CHW and add batch dimension
        tensor = normalized.transpose(2, 0, 1)  # HWC -> CHW
        tensor = np.expand_dims(tensor, axis=0).astype(np.float32)  # Add batch dimension and ensure float32
        
        return tensor
    
    def _postprocess_segmentation(self, output: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Convert model output to segmentation map."""
        
        # Remove batch dimension if present
        if output.ndim == 4:
            output = output[0]
            
        # Convert logits to class predictions
        if output.ndim == 3:  # Multi-class output
            seg_map = np.argmax(output, axis=0)
        else:
            seg_map = output
            
        # Resize to original image size
        seg_map = cv2.resize(seg_map.astype(np.uint8), 
                           (target_shape[1], target_shape[0]), 
                           interpolation=cv2.INTER_NEAREST)
        
        return seg_map
    
    def _find_landing_zones(self, segmentation_map: np.ndarray) -> List[Dict]:
        """Find potential landing zones in segmentation map."""
        
        zones = []
        
        # Find suitable areas (class 1)
        suitable_mask = (segmentation_map == 1).astype(np.uint8)
        
        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        suitable_mask = cv2.morphologyEx(suitable_mask, cv2.MORPH_OPEN, kernel)
        suitable_mask = cv2.morphologyEx(suitable_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(suitable_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area >= self.min_zone_area:
                # Calculate zone properties
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Zone properties
                    zones.append({
                        'id': i,
                        'center': (cx, cy),
                        'area': area,
                        'bbox': (x, y, w, h),
                        'contour': contour,
                        'aspect_ratio': w / h if h > 0 else 1.0,
                        'solidity': area / cv2.contourArea(cv2.convexHull(contour)) if cv2.contourArea(cv2.convexHull(contour)) > 0 else 0
                    })
        
        return zones
    
    def _evaluate_zones(self, zones: List[Dict], seg_map: np.ndarray, image: np.ndarray, altitude: float) -> LandingResult:
        """Evaluate zones and select the best landing target."""
        
        if not zones:
            return LandingResult(status="NO_TARGET", confidence=0.0)
        
        # Score each zone
        scored_zones = []
        
        for zone in zones:
            score = self._calculate_zone_score(zone, seg_map, image.shape[:2], altitude)
            
            if score > 0.3:  # Minimum threshold
                scored_zones.append((score, zone))
        
        if not scored_zones:
            return LandingResult(status="NO_TARGET", confidence=0.0)
        
        # Select best zone
        scored_zones.sort(reverse=True, key=lambda x: x[0])
        best_score, best_zone = scored_zones[0]
        
        # Check safety
        if self._is_zone_safe(best_zone, seg_map, altitude):
            return LandingResult(
                status="TARGET_ACQUIRED",
                confidence=best_score,
                target_pixel=best_zone['center'],
                target_world=self._pixel_to_world(best_zone['center'], altitude),
                distance=self._calculate_distance(best_zone['center'], altitude),
                bearing=self._calculate_bearing(best_zone['center'])
            )
        else:
            return LandingResult(status="UNSAFE", confidence=best_score)
    
    def _calculate_zone_score(self, zone: Dict, seg_map: np.ndarray, image_shape: Tuple[int, int], altitude: float) -> float:
        """Calculate landing suitability score for a zone."""
        
        score = 0.0
        
        # Size score (larger is better, up to a point)
        normalized_area = zone['area'] / (image_shape[0] * image_shape[1])
        size_score = min(normalized_area * 10, 1.0)  # Cap at 1.0
        score += size_score * 0.3
        
        # Shape score (prefer squares/rectangles)
        aspect_ratio = zone['aspect_ratio']
        shape_score = 1.0 - abs(1.0 - aspect_ratio) if aspect_ratio <= 2.0 else 0.5
        score += shape_score * 0.2
        
        # Solidity score (prefer solid shapes)
        score += zone['solidity'] * 0.1
        
        # Center preference (prefer zones near image center)
        img_center = (image_shape[1] // 2, image_shape[0] // 2)
        distance_from_center = math.sqrt(
            (zone['center'][0] - img_center[0])**2 + 
            (zone['center'][1] - img_center[1])**2
        )
        max_distance = math.sqrt(img_center[0]**2 + img_center[1]**2)
        center_score = 1.0 - (distance_from_center / max_distance)
        score += center_score * 0.2
        
        # Altitude consideration (closer = higher confidence)
        altitude_score = max(0.5, 1.0 - altitude / 20.0)  # Decrease confidence with altitude
        score *= altitude_score
        
        # Temporal consistency bonus (if we've been tracking this area)
        if self.last_target and self._zones_overlap(zone, self.last_target):
            score += 0.2
        
        return min(score, 1.0)
    
    def _is_zone_safe(self, zone: Dict, seg_map: np.ndarray, altitude: float) -> bool:
        """Check if zone is safe for landing."""
        
        # Check for obstacles around the zone
        x, y, w, h = zone['bbox']
        
        # Expand search area for obstacles
        margin = max(20, int(self.safety_margin * 10))  # pixels
        
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(seg_map.shape[1], x + w + margin)
        y2 = min(seg_map.shape[0], y + h + margin)
        
        roi = seg_map[y1:y2, x1:x2]
        
        # Check for obstacles (classes 3, 4)
        obstacle_pixels = np.sum((roi == 3) | (roi == 4))
        total_pixels = roi.size
        
        obstacle_ratio = obstacle_pixels / total_pixels if total_pixels > 0 else 0
        
        # Safe if less than 10% obstacles in vicinity
        return obstacle_ratio < 0.1
    
    def _pixel_to_world(self, pixel: Tuple[int, int], altitude: float) -> Tuple[float, float]:
        """Convert pixel coordinates to world coordinates."""
        
        x_pixel, y_pixel = pixel
        
        # Normalize pixel coordinates
        x_norm = (x_pixel - self.cx) / self.fx
        y_norm = (y_pixel - self.cy) / self.fy
        
        # Project to ground plane
        x_world = x_norm * altitude
        y_world = y_norm * altitude
        
        return x_world, y_world
    
    def _calculate_distance(self, pixel: Tuple[int, int], altitude: float) -> float:
        """Calculate distance to target in world coordinates."""
        
        x_world, y_world = self._pixel_to_world(pixel, altitude)
        return math.sqrt(x_world**2 + y_world**2)
    
    def _calculate_bearing(self, pixel: Tuple[int, int]) -> float:
        """Calculate bearing to target from camera center."""
        
        x_pixel, y_pixel = pixel
        
        # Relative to image center
        dx = x_pixel - self.cx
        dy = y_pixel - self.cy
        
        return math.atan2(dx, dy)  # Note: dx, dy for camera coordinate system
    
    def _generate_commands(self, result: LandingResult, altitude: float, current_velocity: Tuple[float, float, float]) -> LandingResult:
        """Generate navigation commands based on target."""
        
        if result.status != "TARGET_ACQUIRED" or not result.target_world:
            # Emergency hover
            result.forward_velocity = 0.0
            result.right_velocity = 0.0
            result.descent_rate = 0.0
            result.yaw_rate = 0.0
            return result
        
        dx, dy = result.target_world
        
        # Determine landing phase based on altitude and distance
        if altitude > 5.0:
            phase = "SEARCH"
            max_vel = self.max_velocity
            descent_rate = 0.3
        elif altitude > 2.0:
            phase = "APPROACH"
            max_vel = self.max_velocity * 0.7
            descent_rate = 0.2
        elif altitude > 0.8:
            phase = "PRECISION"
            max_vel = self.max_velocity * 0.3
            descent_rate = 0.1
        else:
            phase = "LANDING"
            max_vel = self.max_velocity * 0.1
            descent_rate = 0.05
        
        self.landing_phase = phase
        
        # Position control
        result.forward_velocity = np.clip(dx * self.position_gain, -max_vel, max_vel)
        result.right_velocity = np.clip(dy * self.position_gain, -max_vel, max_vel)
        
        # Descent control
        if result.distance < 1.0:  # Close to target
            result.descent_rate = descent_rate
        else:
            result.descent_rate = 0.0  # Hover until positioned
        
        # Yaw control (minimal for stability)
        if abs(result.bearing) > math.radians(15):
            result.yaw_rate = np.clip(result.bearing * 0.3, -0.5, 0.5)
        else:
            result.yaw_rate = 0.0
        
        return result
    
    def _update_tracking(self, result: LandingResult):
        """Update target tracking state."""
        
        if result.status == "TARGET_ACQUIRED":
            self.target_lock_count += 1
            self.last_target = {
                'center': result.target_pixel,
                'confidence': result.confidence
            }
        else:
            self.target_lock_count = max(0, self.target_lock_count - 1)
            if self.target_lock_count == 0:
                self.last_target = None
    
    def _zones_overlap(self, zone1: Dict, zone2: Dict) -> bool:
        """Check if two zones overlap significantly."""
        
        if not isinstance(zone2, dict) or 'center' not in zone2:
            return False
        
        c1 = zone1['center']
        c2 = zone2['center']
        
        distance = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
        
        return distance < 50  # pixels
    
    def _update_fps(self, processing_time: float) -> float:
        """Update and return current FPS."""
        
        self.frame_times.append(processing_time)
        
        if len(self.frame_times) > self.max_history:
            self.frame_times.pop(0)
        
        avg_time = sum(self.frame_times) / len(self.frame_times)
        
        return 1000.0 / avg_time if avg_time > 0 else 0.0
    
    def _create_visualization(self, image: np.ndarray, seg_map: np.ndarray, result: LandingResult) -> np.ndarray:
        """Create annotated visualization image."""
        
        vis = image.copy()
        
        # Overlay segmentation (semi-transparent)
        colored_seg = self._colorize_segmentation(seg_map)
        vis = cv2.addWeighted(vis, 0.7, colored_seg, 0.3, 0)
        
        # Draw target
        if result.target_pixel:
            center = result.target_pixel
            
            # Target circle
            cv2.circle(vis, center, 20, (0, 255, 0), 3)
            cv2.circle(vis, center, 5, (0, 255, 0), -1)
            
            # Distance info
            if result.distance:
                text = f"{result.distance:.1f}m"
                cv2.putText(vis, text, (center[0] + 25, center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Status overlay
        status_color = (0, 255, 0) if result.status == "TARGET_ACQUIRED" else (0, 0, 255)
        cv2.putText(vis, f"{result.status} ({result.confidence:.2f})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Phase and FPS
        cv2.putText(vis, f"Phase: {self.landing_phase}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f"FPS: {result.fps:.1f}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Commands
        cmd_text = f"[{result.forward_velocity:.1f}, {result.right_velocity:.1f}, {result.descent_rate:.1f}]"
        cv2.putText(vis, cmd_text, (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return vis
    
    def _colorize_segmentation(self, seg_map: np.ndarray) -> np.ndarray:
        """Convert segmentation map to colored visualization."""
        
        colors = {
            0: [0, 0, 0],        # background - black
            1: [0, 255, 0],      # suitable - green
            2: [255, 255, 0],    # marginal - yellow
            3: [255, 0, 0],      # obstacles - red
            4: [128, 0, 128],    # unsafe - purple
            5: [128, 128, 128]   # unknown - gray
        }
        
        colored = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
        
        for class_id, color in colors.items():
            colored[seg_map == class_id] = color
        
        return colored
    
    def reset_state(self):
        """Reset detector state for new flight."""
        
        self.landing_phase = "SEARCH"
        self.target_lock_count = 0
        self.last_target = None
        self.frame_times.clear()
        
        print("🔄 Detector state reset")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        
        avg_fps = sum(1000.0 / t for t in self.frame_times) / len(self.frame_times) if self.frame_times else 0.0
        
        return {
            'fps': avg_fps,
            'frame_count': len(self.frame_times),
            'landing_phase': self.landing_phase,
            'target_lock_count': self.target_lock_count,
            'model_loaded': self.session is not None
        }

# Example usage
if __name__ == "__main__":
    """
    Example usage of the UAV Landing Detector
    """
    
    # Initialize detector
    detector = UAVLandingDetector(
        model_path="models/bisenetv2_landing.onnx",  # Your ONNX model
        enable_visualization=True
    )
    
    # Test with webcam or synthetic data
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No webcam, using synthetic test")
            cap = None
    except:
        cap = None
    
    print("🎮 Controls: 'q' to quit, 'r' to reset, 's' for stats")
    
    altitude = 8.0  # Simulated altitude
    
    while True:
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # Generate synthetic test frame
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            time.sleep(0.033)  # ~30 FPS
        
        # Process frame
        result = detector.process_frame(frame, altitude=altitude)
        
        # Print results
        if result.status == "TARGET_ACQUIRED":
            print(f"🎯 Target at {result.distance:.1f}m, commands: [{result.forward_velocity:.1f}, {result.right_velocity:.1f}, {result.descent_rate:.1f}]")
            
            # Simulate descent
            if result.descent_rate > 0:
                altitude = max(0.5, altitude - 0.05)
        
        # Show visualization
        if result.annotated_image is not None:
            cv2.imshow("UAV Landing Detector", result.annotated_image)
        else:
            cv2.imshow("UAV Landing Detector", frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset_state()
            altitude = 8.0
        elif key == ord('s'):
            stats = detector.get_performance_stats()
            print(f"📊 Stats: {stats['fps']:.1f} FPS, Phase: {stats['landing_phase']}")
    
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    
    print("✅ Demo completed")
