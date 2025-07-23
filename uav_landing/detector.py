#!/usr/bin/env python3
"""
Production-Ready UAV Landing Detector with Neurosymbolic Memory

A complete, clean implementation combining neural perception with symbolic reasoning
and persistent memory for robust autonomous UAV landing.
"""

import cv2
import numpy as np
import time
import math
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX Runtime not available, running in placeholder mode")

from .types import LandingResult, MemoryZone
from .memory import NeuroSymbolicMemory


class UAVLandingDetector:
    """
    Production-ready UAV Landing Detector with Neurosymbolic Memory.
    
    Features:
    - Neural semantic segmentation for landing zone detection
    - Symbolic reasoning for safety and decision making
    - Persistent memory system for robust operation in challenging conditions
    - Real-time performance optimized for flight-critical applications
    """
    
    def __init__(self,
                 model_path: str = "models/bisenetv2_uav_landing.onnx",
                 input_resolution: Tuple[int, int] = (512, 512),
                 camera_fx: float = 800,
                 camera_fy: float = 800,
                 enable_memory: bool = True,
                 enable_visualization: bool = False,
                 memory_config: Optional[Dict] = None,
                 memory_persistence_file: str = "uav_memory.json",
                 device: str = "auto"):
        """
        Initialize UAV Landing Detector.
        
        Args:
            model_path: Path to ONNX segmentation model
            input_resolution: Model input size (width, height)
            camera_fx: Camera focal length in x direction
            camera_fy: Camera focal length in y direction  
            enable_memory: Enable neurosymbolic memory system
            enable_visualization: Generate debug visualizations
            memory_config: Configuration for memory system
            memory_persistence_file: File to save/load memory state
            device: Inference device ("auto", "cuda", "cpu")
        """
        
        self.model_path = Path(model_path)
        self.input_size = input_resolution
        self.enable_visualization = enable_visualization
        self.memory_persistence_file = Path(memory_persistence_file)
        
        # Camera parameters
        self.camera_matrix = np.array([
            [camera_fx, 0, input_resolution[0]/2],
            [0, camera_fy, input_resolution[1]/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.fx = camera_fx
        self.fy = camera_fy
        self.cx = input_resolution[0] / 2
        self.cy = input_resolution[1] / 2
        
        # Initialize ONNX model
        self.session = None
        self.input_name = None
        self.output_name = None
        self._initialize_model(device)
        
        # Detection parameters
        self.min_zone_area = max(1000, (input_resolution[0] * input_resolution[1]) // 100)
        self.max_velocity = 2.0
        self.position_gain = 0.5
        self.safety_margin = 0.3
        
        # State tracking
        self.landing_phase = "SEARCH"
        self.target_lock_count = 0
        self.last_target = None
        self.frame_times = []
        self.max_history = 30
        
        # Visual confidence tracking
        self.visual_confidence_history = []
        self.no_target_count = 0
        self.recovery_mode = False
        
        # Memory fusion parameters
        self.min_visual_confidence = 0.4
        self.memory_fusion_threshold = 0.6
        
        # Memory system
        if enable_memory:
            default_config = {
                'memory_horizon': 300.0,
                'spatial_resolution': 0.5,
                'confidence_decay_rate': 0.98,
                'min_observations': 2
            }
            if memory_config:
                default_config.update(memory_config)
            
            self.memory = NeuroSymbolicMemory(**default_config)
            
            # Load persistent memory
            if self.memory_persistence_file.exists():
                self.memory.load_memory(str(self.memory_persistence_file))
        else:
            self.memory = None
        
        # Segmentation output tracking
        self.last_segmentation_output = None
        self.last_raw_output = None
        self.last_confidence_map = None
        
        print(f" UAV Landing Detector initialized")
        print(f"   Resolution: {input_resolution}")
        print(f"   Memory: {'enabled' if enable_memory else 'disabled'}")
        print(f"   Model: {self.model_path.name if self.model_path.exists() else 'placeholder'}")
    
    def _initialize_model(self, device: str):
        """Initialize the ONNX model for inference"""
        
        if not ONNX_AVAILABLE or not self.model_path.exists():
            print("⚠️  Using placeholder mode (no ONNX model)")
            return
        
        try:
            providers = []
            if device == "auto":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            elif device == "cuda":
                providers = ['CUDAExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            print(f" Model loaded: {self.model_path.name}")
            print(f"   Provider: {self.session.get_providers()[0]}")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            self.session = None
    
    def process_frame(self,
                     image: np.ndarray,
                     altitude: float,
                     current_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                     drone_position: Optional[Tuple[float, float]] = None,
                     drone_heading: float = 0.0) -> LandingResult:
        """
        Process single frame for landing zone detection.
        
        Args:
            image: Input BGR image from camera
            altitude: Current altitude above ground (meters)
            current_velocity: Current velocity [vx, vy, vz] (m/s)
            drone_position: Current drone position in world coordinates
            drone_heading: Current drone heading in radians
            
        Returns:
            LandingResult with detection, navigation, and memory information
        """
        
        start_time = time.time()
        
        # Validate inputs
        if image is None or image.size == 0 or altitude <= 0:
            return LandingResult(status="ERROR", confidence=0.0)
        
        # Update memory system with current drone state
        if self.memory and drone_position:
            self.memory.update_drone_state(drone_position, altitude, drone_heading)
        
        try:
            # Step 1: Semantic Segmentation
            segmentation_map = self._run_segmentation(image)
            
            # Step 2: Find Landing Zones
            zones = self._find_landing_zones(segmentation_map)
            
            # Step 3: Assess Visual Confidence
            visual_confidence = self._assess_visual_confidence(image, zones)
            self.visual_confidence_history.append(visual_confidence)
            if len(self.visual_confidence_history) > 10:
                self.visual_confidence_history.pop(0)
            
            # Step 4: Memory Integration
            if self.memory:
                zones, fusion_mode, memory_confidence = self._integrate_memory(
                    zones, image, altitude, visual_confidence, drone_position
                )
            else:
                fusion_mode = "perception_only"
                memory_confidence = 0.0
            
            # Step 5: Evaluate Best Zone
            result = self._evaluate_zones(zones, segmentation_map, image, altitude)
            
            # Step 6: Add Memory Information
            result.memory_confidence = memory_confidence
            result.perception_memory_fusion = fusion_mode
            if self.memory:
                result.memory_zones = self.memory.predict_zones_from_memory(min_confidence=0.2)
                result.memory_status = self.memory.get_memory_status()
            
            # Step 7: Handle Recovery Behavior
            result = self._handle_recovery(result, visual_confidence)
            
            # Step 8: Generate Navigation Commands
            result = self._generate_commands(result, altitude, current_velocity)
            
            # Step 9: Update Tracking
            self._update_tracking(result)
            
            # Step 10: Update Memory with Observations
            if self.memory and result.status == "TARGET_ACQUIRED":
                self._update_memory_with_observations(result, altitude, image)
            
            # Step 11: Performance Tracking
            processing_time = (time.time() - start_time) * 1000
            result.processing_time = processing_time
            result.fps = self._update_fps(processing_time)
            
            # Step 12: Visualization
            if self.enable_visualization:
                result.annotated_image = self._create_visualization(image, segmentation_map, result)
            
            return result
            
        except Exception as e:
            print(f"❌ Processing error: {e}")
            return LandingResult(
                status="ERROR", 
                confidence=0.0,
                processing_time=(time.time() - start_time) * 1000
            )
    
    def _run_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Run semantic segmentation on input image"""
        
        if self.session is None:
            return self._placeholder_segmentation(image)
        
        # Real model inference
        input_tensor = self._preprocess_image(image)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        
        # Store outputs
        self.last_raw_output = outputs[0].copy()
        
        # Postprocess
        segmentation_map = self._postprocess_segmentation(outputs[0], image.shape[:2])
        self.last_segmentation_output = segmentation_map.copy()
        
        # Create confidence map
        self._create_confidence_map(outputs[0], image.shape[:2])
        
        return segmentation_map
    
    def _placeholder_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Generate placeholder segmentation for testing"""
        
        h, w = image.shape[:2]
        seg_map = np.zeros((h, w), dtype=np.uint8)
        
        # Create synthetic landing zone
        center_x, center_y = w // 2, h // 2
        cv2.rectangle(seg_map, (center_x - 60, center_y - 40), 
                     (center_x + 60, center_y + 40), 1, -1)
        
        # Add some obstacles
        cv2.rectangle(seg_map, (50, 50), (100, 150), 3, -1)
        cv2.circle(seg_map, (w - 80, 80), 30, 3, -1)
        
        return seg_map
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for neural network"""
        
        resized = cv2.resize(image, self.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        
        # CHW format with batch dimension
        tensor = normalized.transpose(2, 0, 1)
        tensor = np.expand_dims(tensor, axis=0).astype(np.float32)
        
        return tensor
    
    def _postprocess_segmentation(self, output: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Convert model output to segmentation map"""
        
        if output.ndim == 4:
            output = output[0]
        
        if output.ndim == 3:
            seg_map = np.argmax(output, axis=0)
        else:
            seg_map = output
        
        seg_map = cv2.resize(seg_map.astype(np.uint8), 
                           (target_shape[1], target_shape[0]), 
                           interpolation=cv2.INTER_NEAREST)
        
        return seg_map
    
    def _create_confidence_map(self, raw_output: np.ndarray, target_shape: Tuple[int, int]):
        """Create confidence map from raw model output"""
        
        if raw_output.ndim == 4:
            raw_output = raw_output[0]
        
        if raw_output.ndim == 3:
            exp_output = np.exp(raw_output - np.max(raw_output, axis=0, keepdims=True))
            probabilities = exp_output / np.sum(exp_output, axis=0, keepdims=True)
            confidence_map = np.max(probabilities, axis=0)
        else:
            confidence_map = np.ones_like(raw_output) * 0.8
        
        self.last_confidence_map = cv2.resize(
            confidence_map.astype(np.float32),
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
    
    def _find_landing_zones(self, segmentation_map: np.ndarray) -> List[Dict]:
        """Find potential landing zones in segmentation map"""
        
        zones = []
        suitable_mask = (segmentation_map == 1).astype(np.uint8)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        suitable_mask = cv2.morphologyEx(suitable_mask, cv2.MORPH_OPEN, kernel)
        suitable_mask = cv2.morphologyEx(suitable_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(suitable_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area >= self.min_zone_area:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    x, y, w, h = cv2.boundingRect(contour)
                    
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
    
    def _assess_visual_confidence(self, image: np.ndarray, zones: List[Dict]) -> float:
        """Assess overall confidence in visual perception"""
        
        # Detection confidence
        detection_confidence = 0.8 if zones else 0.0
        
        # Visual analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast_score = np.std(gray) / 128.0
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        color_diversity = 1.0 - np.max(hist) / np.sum(hist)
        
        visual_confidence = (
            detection_confidence * 0.4 +
            contrast_score * 0.3 +
            edge_density * 10.0 * 0.2 +
            color_diversity * 0.1
        )
        
        return min(1.0, max(0.0, visual_confidence))
    
    def _integrate_memory(self, zones: List[Dict], image: np.ndarray, altitude: float, 
                         visual_confidence: float, drone_position: Optional[Tuple[float, float]]) -> Tuple[List[Dict], str, float]:
        """Integrate memory predictions with current perception"""
        
        memory_zones = self.memory.predict_zones_from_memory(min_confidence=0.2, max_zones=3)
        memory_confidence = self.memory.get_memory_confidence(drone_position) if drone_position else 0.0
        
        if visual_confidence >= self.memory_fusion_threshold:
            return zones, "perception_only", memory_confidence
        elif visual_confidence <= self.min_visual_confidence:
            # Use memory zones
            enhanced_zones = zones + memory_zones
            return enhanced_zones, "memory_only", memory_confidence
        else:
            # Fuse perception and memory
            enhanced_zones = zones + memory_zones
            return enhanced_zones, "fused", memory_confidence
    
    def _evaluate_zones(self, zones: List[Dict], seg_map: np.ndarray, image: np.ndarray, altitude: float) -> LandingResult:
        """Evaluate zones and select the best landing target"""
        
        if not zones:
            return LandingResult(status="NO_TARGET", confidence=0.0)
        
        # Score zones
        scored_zones = []
        for zone in zones:
            score = self._calculate_zone_score(zone, seg_map, image.shape[:2], altitude)
            if score > 0.3:
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
        """Calculate landing suitability score using neurosymbolic reasoning"""
        
        try:
            # Validate zone has required fields
            required_fields = ['area', 'bbox', 'aspect_ratio', 'center']
            for field in required_fields:
                if field not in zone:
                    print(f"⚠️  Zone missing required field: {field}")
                    return 0.0
            
            # Neural confidence
            neural_confidence = 0.8
            if hasattr(self, 'last_confidence_map') and self.last_confidence_map is not None:
                x, y, w, h = zone['bbox']
                zone_mask = np.zeros_like(seg_map, dtype=bool)
                zone_mask[y:y+h, x:x+w] = True
                neural_confidence = np.mean(self.last_confidence_map[zone_mask])
            
            # Symbolic reasoning
            symbolic_score = 0.0
            
            # Size reasoning
            normalized_area = zone['area'] / (image_shape[0] * image_shape[1])
            if normalized_area > 0.05:
                size_score = min(normalized_area * 8, 1.0)
            elif normalized_area > 0.02:
                size_score = normalized_area * 15
            else:
                size_score = normalized_area * 20
            symbolic_score += size_score * 0.3
            
            # Shape reasoning
            aspect_ratio = zone['aspect_ratio']
            if 0.7 <= aspect_ratio <= 1.4:
                shape_score = 1.0
            elif 0.5 <= aspect_ratio <= 2.0:
                shape_score = 0.8 - abs(1.0 - aspect_ratio) * 0.3
            else:
                shape_score = 0.4
            
            shape_score *= zone['solidity']
            symbolic_score += shape_score * 0.25
            
            # Spatial reasoning
            img_center = (image_shape[1] // 2, image_shape[0] // 2)
            distance_from_center = math.sqrt(
                (zone['center'][0] - img_center[0])**2 + 
                (zone['center'][1] - img_center[1])**2
            )
            max_distance = math.sqrt(img_center[0]**2 + img_center[1]**2)
            center_preference = 1.0 - (distance_from_center / max_distance)
            symbolic_score += center_preference * 0.2
            
            # Altitude reasoning
            if altitude > 10.0:
                altitude_factor = 1.2 if normalized_area > 0.08 else 0.8
            elif altitude > 5.0:
                altitude_factor = 1.1 if normalized_area > 0.04 else 0.9
            else:
                altitude_factor = 1.0
            
            symbolic_score *= altitude_factor
            
            # Temporal reasoning
            temporal_bonus = 0.0
            if self.last_target and self._zones_overlap(zone, self.last_target):
                consistency = self._calculate_zone_consistency(zone, self.last_target)
                temporal_bonus = consistency * 0.15
            
            # Safety penalty
            safety_penalty = self._calculate_safety_penalty(zone, seg_map, altitude)
            
            # Final score
            neuro_symbolic_score = (
                neural_confidence * 0.4 +
                symbolic_score * 0.6 +
                temporal_bonus -
                safety_penalty
            )
            
            return max(0.0, min(neuro_symbolic_score, 1.0))
        
        except Exception as e:
            print(f"⚠️  Zone scoring error: {e}")
            return 0.0
    
    def _calculate_zone_consistency(self, current_zone: Dict, last_zone: Dict) -> float:
        """Calculate consistency between current and previous zone"""
        pos_diff = math.sqrt(
            (current_zone['center'][0] - last_zone['center'][0])**2 + 
            (current_zone['center'][1] - last_zone['center'][1])**2
        )
        pos_consistency = max(0, 1.0 - pos_diff / 100.0)
        
        size_ratio = min(current_zone['area'], last_zone['area']) / max(current_zone['area'], last_zone['area'])
        size_consistency = size_ratio
        
        return (pos_consistency * 0.7 + size_consistency * 0.3)
    
    def _calculate_safety_penalty(self, zone: Dict, seg_map: np.ndarray, altitude: float) -> float:
        """Calculate safety penalty based on surrounding context"""
        x, y, w, h = zone['bbox']
        safety_margin = max(10, int(altitude * 2))
        
        x1 = max(0, x - safety_margin)
        y1 = max(0, y - safety_margin)
        x2 = min(seg_map.shape[1], x + w + safety_margin)
        y2 = min(seg_map.shape[0], y + h + safety_margin)
        
        safety_area = seg_map[y1:y2, x1:x2]
        
        if safety_area.size > 0:
            unsuitable_pixels = np.sum(safety_area >= 3)
            total_pixels = safety_area.size
            unsuitable_ratio = unsuitable_pixels / total_pixels
            
            if unsuitable_ratio > 0.5:
                return 0.8
            elif unsuitable_ratio > 0.2:
                return 0.4
            elif unsuitable_ratio > 0.1:
                return 0.2
            else:
                return 0.0
        
        return 0.0
    
    def _is_zone_safe(self, zone: Dict, seg_map: np.ndarray, altitude: float) -> bool:
        """Check if zone is safe for landing"""
        x, y, w, h = zone['bbox']
        margin = max(20, int(self.safety_margin * 10))
        
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(seg_map.shape[1], x + w + margin)
        y2 = min(seg_map.shape[0], y + h + margin)
        
        roi = seg_map[y1:y2, x1:x2]
        obstacle_pixels = np.sum((roi == 3) | (roi == 4))
        total_pixels = roi.size
        
        obstacle_ratio = obstacle_pixels / total_pixels if total_pixels > 0 else 0
        return obstacle_ratio < 0.1
    
    def _pixel_to_world(self, pixel: Tuple[int, int], altitude: float) -> Tuple[float, float]:
        """Convert pixel coordinates to world coordinates"""
        x_pixel, y_pixel = pixel
        
        x_norm = (x_pixel - self.cx) / self.fx
        y_norm = (y_pixel - self.cy) / self.fy
        
        x_world = x_norm * altitude
        y_world = y_norm * altitude
        
        return x_world, y_world
    
    def _calculate_distance(self, pixel: Tuple[int, int], altitude: float) -> float:
        """Calculate distance to target in world coordinates"""
        x_world, y_world = self._pixel_to_world(pixel, altitude)
        return math.sqrt(x_world**2 + y_world**2)
    
    def _calculate_bearing(self, pixel: Tuple[int, int]) -> float:
        """Calculate bearing to target from camera center"""
        x_pixel, y_pixel = pixel
        dx = x_pixel - self.cx
        dy = y_pixel - self.cy
        return math.atan2(dx, dy)
    
    def _handle_recovery(self, result: LandingResult, visual_confidence: float) -> LandingResult:
        """Handle recovery behaviors when target is lost"""
        
        if result.status == "NO_TARGET":
            self.no_target_count += 1
        else:
            self.no_target_count = 0
            self.recovery_mode = False
        
        if self.no_target_count >= 5:
            self.recovery_mode = True
            result.recovery_mode = True
            
            avg_visual_confidence = np.mean(self.visual_confidence_history) if self.visual_confidence_history else 0.5
            
            if avg_visual_confidence < 0.3:
                result.search_pattern = "memory_guided"
                result.status = "SEARCHING_MEMORY"
            elif avg_visual_confidence < 0.6:
                result.search_pattern = "spiral_search"
                result.status = "SEARCHING_PATTERN"
            else:
                result.search_pattern = "hover_observe"
                result.status = "SEARCHING_VISUAL"
        
        return result
    
    def _generate_commands(self, result: LandingResult, altitude: float, current_velocity: Tuple[float, float, float]) -> LandingResult:
        """Generate navigation commands based on target"""
        
        if result.status != "TARGET_ACQUIRED" or not result.target_world:
            result.forward_velocity = 0.0
            result.right_velocity = 0.0
            result.descent_rate = 0.0
            result.yaw_rate = 0.0
            return result
        
        dx, dy = result.target_world
        
        # Determine landing phase
        if altitude > 5.0:
            self.landing_phase = "SEARCH"
            max_vel = self.max_velocity
            descent_rate = 0.3
        elif altitude > 2.0:
            self.landing_phase = "APPROACH"
            max_vel = self.max_velocity * 0.7
            descent_rate = 0.2
        elif altitude > 0.8:
            self.landing_phase = "PRECISION"
            max_vel = self.max_velocity * 0.3
            descent_rate = 0.1
        else:
            self.landing_phase = "LANDING"
            max_vel = self.max_velocity * 0.1
            descent_rate = 0.05
        
        # Position control
        result.forward_velocity = np.clip(dx * self.position_gain, -max_vel, max_vel)
        result.right_velocity = np.clip(dy * self.position_gain, -max_vel, max_vel)
        
        # Descent control
        if result.distance < 1.0:
            result.descent_rate = descent_rate
        else:
            result.descent_rate = 0.0
        
        # Yaw control
        if abs(result.bearing) > math.radians(15):
            result.yaw_rate = np.clip(result.bearing * 0.3, -0.5, 0.5)
        else:
            result.yaw_rate = 0.0
        
        return result
    
    def _update_tracking(self, result: LandingResult):
        """Update target tracking state"""
        
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
    
    def _update_memory_with_observations(self, result: LandingResult, altitude: float, image: np.ndarray):
        """Update memory system with current observations"""
        
        if not self.memory or result.status != "TARGET_ACQUIRED":
            return
        
        zone = {
            'center': result.target_pixel,
            'area': 1000,
            'confidence': result.confidence
        }
        
        environment_context = self._analyze_environment_context(image)
        
        if result.target_world:
            self.memory.observe_zones(
                zones=[zone],
                world_positions=[result.target_world],
                confidences=[result.confidence],
                environment_context=environment_context
            )
    
    def _analyze_environment_context(self, image: np.ndarray) -> Dict:
        """Analyze current environment for context"""
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Analyze dominant colors
        grass_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        grass_ratio = np.sum(grass_mask > 0) / grass_mask.size
        
        concrete_mask = cv2.inRange(hsv, np.array([0, 0, 50]), np.array([180, 50, 200]))
        concrete_ratio = np.sum(concrete_mask > 0) / concrete_mask.size
        
        if grass_ratio > 0.6:
            env_type = "grass_field"
        elif concrete_ratio > 0.5:
            env_type = "concrete_surface"
        elif grass_ratio > 0.3 and concrete_ratio > 0.3:
            env_type = "mixed_terrain"
        else:
            env_type = "unknown"
        
        return {
            'environment_type': env_type,
            'grass_ratio': grass_ratio,
            'concrete_ratio': concrete_ratio,
            'lighting_condition': 'normal',
            'image_quality': np.std(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) / 128.0
        }
    
    def _zones_overlap(self, zone1: Dict, zone2: Dict) -> bool:
        """Check if two zones overlap significantly"""
        
        if not isinstance(zone2, dict) or 'center' not in zone2:
            return False
        
        c1 = zone1['center']
        c2 = zone2['center']
        
        distance = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
        return distance < 50
    
    def _update_fps(self, processing_time: float) -> float:
        """Update and return current FPS"""
        
        self.frame_times.append(processing_time)
        
        if len(self.frame_times) > self.max_history:
            self.frame_times.pop(0)
        
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1000.0 / avg_time if avg_time > 0 else 0.0
    
    def _create_visualization(self, image: np.ndarray, seg_map: np.ndarray, result: LandingResult) -> np.ndarray:
        """Create annotated visualization image"""
        
        vis = image.copy()
        
        # Overlay segmentation
        colored_seg = self._colorize_segmentation(seg_map)
        vis = cv2.addWeighted(vis, 0.7, colored_seg, 0.3, 0)
        
        # Draw target
        if result.target_pixel:
            center = result.target_pixel
            cv2.circle(vis, center, 20, (0, 255, 0), 3)
            cv2.circle(vis, center, 5, (0, 255, 0), -1)
            
            if result.distance:
                text = f"{result.distance:.1f}m"
                cv2.putText(vis, text, (center[0] + 25, center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Status overlay
        status_color = (0, 255, 0) if result.status == "TARGET_ACQUIRED" else (0, 0, 255)
        cv2.putText(vis, f"{result.status} ({result.confidence:.2f})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Memory info
        if result.perception_memory_fusion != "perception_only":
            cv2.putText(vis, f"Memory: {result.perception_memory_fusion}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Phase and FPS
        cv2.putText(vis, f"Phase: {self.landing_phase}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f"FPS: {result.fps:.1f}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return vis
    
    def _colorize_segmentation(self, seg_map: np.ndarray) -> np.ndarray:
        """Convert segmentation map to colored visualization"""
        
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
    
    def save_memory(self):
        """Save current memory state for persistence"""
        if self.memory:
            self.memory.save_memory(str(self.memory_persistence_file))
    
    def reset_state(self):
        """Reset detector state for new flight"""
        self.landing_phase = "SEARCH"
        self.target_lock_count = 0
        self.last_target = None
        self.frame_times.clear()
        self.visual_confidence_history.clear()
        self.no_target_count = 0
        self.recovery_mode = False
        
        if self.memory:
            self.memory.reset_memory()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        
        avg_fps = sum(1000.0 / t for t in self.frame_times) / len(self.frame_times) if self.frame_times else 0.0
        
        return {
            'fps': avg_fps,
            'frame_count': len(self.frame_times),
            'landing_phase': self.landing_phase,
            'target_lock_count': self.target_lock_count,
            'model_loaded': self.session is not None,
            'memory_enabled': self.memory is not None
        }
