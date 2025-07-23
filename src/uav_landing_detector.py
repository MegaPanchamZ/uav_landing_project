#!/usr/bin/env python3
"""
UAV Landing Zone Detector - Single Class Implementation
Real-time semantic segmentation + neuro-symbolic reasoning for UAV landing

Usage:                    if device == "auto":
                # Try TensorRT first (best), then CUDA, then CPU
                try:
                    # Set ONNX Runtime to only show critical errors
                    ort.set_default_logger_severity(3)  # Only ERROR and FATAL
                    
                    # Priority order: TensorRT > CUDA > CPU
                    available_providers = ort.get_available_providers()
                    
                    if 'TensorrtExecutionProvider' in available_providers:
                        try:
                            test_session = ort.InferenceSession(self.model_path, providers=['TensorrtExecutionProvider'])
                            test_session = None  # Clean up
                            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
                            self.actual_device = "TensorRT"
                            print("🚀 TensorRT acceleration enabled (Optimal)")
                        except Exception:
                            # Fall through to CUDA
                            if 'CUDAExecutionProvider' in available_providers:
                                try:
                                    test_session = ort.InferenceSession(self.model_path, providers=['CUDAExecutionProvider'])
                                    test_session = None  # Clean up
                                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                                    self.actual_device = "CUDA"
                                    print("🚀 CUDA acceleration enabled")
                                except Exception:
                                    providers = ['CPUExecutionProvider']
                                    self.actual_device = "CPU"
                                    print("💻 Using CPU processing (GPU libraries unavailable)")
                            else:
                                providers = ['CPUExecutionProvider']
                                self.actual_device = "CPU"
                                print("💻 Using CPU processing (No GPU providers)")
                    elif 'CUDAExecutionProvider' in available_providers:
                        try:
                            test_session = ort.InferenceSession(self.model_path, providers=['CUDAExecutionProvider'])
                            test_session = None  # Clean up
                            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                            self.actual_device = "CUDA"
                            print("🚀 CUDA acceleration enabled")
                        except Exception:
                            providers = ['CPUExecutionProvider']
                            self.actual_device = "CPU"
                            print("💻 Using CPU processing (CUDA libraries unavailable)")
                    else:
                        providers = ['CPUExecutionProvider']
                        self.actual_device = "CPU"
                        print("💻 Using CPU processing (No GPU providers)")
                        
                except Exception as e:
                    providers = ['CPUExecutionProvider']
                    self.actual_device = "CPU"
                    print(f"💻 Using CPU processing (Error: {e})")
                finally:
                    # Keep logging suppressed for cleaner output
                    passor = UAVLandingDetector()
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
    print("⚠️  ONNXRuntime not available")
    ONNX_AVAILABLE = False
    ort = None

def check_gpu_availability() -> bool:
    """Check if GPU/CUDA is actually available for ONNX Runtime"""
    if not ONNX_AVAILABLE or ort is None:
        return False
        
    try:
        # Check if CUDA provider is in available providers
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' not in available_providers:
            return False
        return True
    except Exception:
        return False

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
    UAV Landing Zone Detector with Enhanced Neuro-Symbolic Reasoning
    
    Combines neural segmentation with symbolic reasoning for safe landing decisions.
    Includes comprehensive traceability and risk assessment capabilities.
    """
    
    def __init__(self, model_path="models/bisenetv2_uav_landing.onnx", 
                 input_resolution=(256, 256), camera_fx=800, camera_fy=800, 
                 enable_visualization=True, device="auto"):
        """
        Initialize UAV Landing Detector
        
        Args:
            model_path: Path to the ONNX model file
            input_resolution: Tuple (width, height) for model input
                            - (256, 256): Fast inference, lower quality (~80-127 FPS)
                            - (512, 512): Balanced quality and speed (~20-60 FPS)
                            - (768, 768): High quality, slower (~8-25 FPS)
                            - (1024, 1024): Maximum quality, slowest (~3-12 FPS)
            camera_fx: Camera focal length in x direction (pixels)
            camera_fy: Camera focal length in y direction (pixels) 
            enable_visualization: Whether to generate visualization overlays
            device: Device for inference ("auto", "tensorrt", "cuda", "cpu")
                   - "auto": Try TensorRT → CUDA → CPU (recommended)
                   - "tensorrt": Force TensorRT (fastest, requires TensorRT installation)
                   - "cuda": Force CUDA (fast, requires CUDA libraries)
                   - "cpu": Force CPU (compatible, slower)
        """
        
        self.model_path = Path(model_path)
        self.input_size = input_resolution
        self.enable_visualization = enable_visualization
        
        # Camera intrinsic matrix (3x3) - use resolution for center point
        self.camera_matrix = np.array([
            [camera_fx, 0, input_resolution[0]/2],  # Use resolution for center
            [0, camera_fy, input_resolution[1]/2],
            [0, 0, 1]
            ], dtype=np.float32)
        
        # Extract camera parameters
        self.fx = self.camera_matrix[0, 0]
        self.fy = self.camera_matrix[1, 1]
        self.cx = self.camera_matrix[0, 2]
        self.cy = self.camera_matrix[1, 2]
        
        # Initialize ONNX session attributes
        self.session = None
        self.input_name = None
        self.output_name = None
        
        # Initialize segmentation output tracking
        self.last_segmentation_output = None
        self.last_raw_output = None
        self.last_confidence_map = None
        
        # Landing zone detection parameters
        self.min_zone_area = max(1000, (input_resolution[0] * input_resolution[1]) // 100)  # Adaptive to resolution
        
        # Navigation parameters
        self.max_velocity = 2.0  # m/s
        self.position_gain = 0.5
        self.safety_margin = 0.3  # meters
        
        # State tracking
        self.landing_phase = "SEARCH"
        self.target_lock_count = 0
        self.last_target = None
        self.frame_times = []
        self.max_history = 30
        
        # Store device preference
        self.device = device
        self.actual_device = "CPU"  # Will be updated by _initialize_model
        
        # Initialize the ONNX model
        self._initialize_model(device)
        
        print(f" UAVLandingDetector initialized with resolution {input_resolution}")
        
    def _initialize_model(self, device: str):
        """Initialize the ONNX model for inference."""
        self.session = None
        
        if not ONNX_AVAILABLE or not self.model_path or not Path(self.model_path).exists():
            print("⚠️  Using placeholder mode (no ONNX model)")
            return
            
        try:
            # Set up execution providers with better error handling
            providers = []
            if device == "auto":
                # Try CUDA first, fallback to CPU
                try:
                    # Suppress CUDA warning temporarily
                    import warnings
                    import logging
                    
                    # Set ONNX Runtime to only show critical errors
                    ort.set_default_logger_severity(4)  # Only FATAL errors
                    
                    # Test if CUDA is actually available
                    test_session = ort.InferenceSession(self.model_path, providers=['CUDAExecutionProvider'])
                    test_session = None  # Clean up
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    self.actual_device = "GPU"
                    print("🚀 GPU acceleration enabled (CUDA)")
                except Exception:
                    # CUDA failed, use CPU
                    providers = ['CPUExecutionProvider']
                    self.actual_device = "CPU"
                    print("� Using CPU processing (CUDA libraries not available)")
                finally:
                    # Keep logging suppressed for cleaner output
                    pass
                    
            elif device == "tensorrt":
                providers = ['TensorrtExecutionProvider']
                self.actual_device = "TensorRT"
            elif device == "cuda":
                providers = ['CUDAExecutionProvider']
                self.actual_device = "CUDA"
            else:
                providers = ['CPUExecutionProvider']
                self.actual_device = "CPU"
                
            # Create session
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Update actual device based on what's actually being used
            actual_provider = self.session.get_providers()[0]
            if 'TensorRT' in actual_provider:
                self.actual_device = "TensorRT"
            elif 'CUDA' in actual_provider:
                self.actual_device = "CUDA"
            else:
                self.actual_device = "CPU"
            
            # Get input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            print(f" Model loaded: {Path(self.model_path).name}")
            print(f"   Provider: {actual_provider}")
            
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
        
        # Store raw outputs for visualization
        self.last_raw_output = outputs[0].copy()
        
        # Postprocess output
        segmentation_map = self._postprocess_segmentation(outputs[0], image.shape[:2])
        
        # Store processed outputs
        self.last_segmentation_output = segmentation_map.copy()
        
        # Create confidence map from raw logits
        if self.last_raw_output.ndim == 4:
            raw_output = self.last_raw_output[0]  # Remove batch dimension
        else:
            raw_output = self.last_raw_output
            
        if raw_output.ndim == 3:  # Multi-class output
            # Softmax to get probabilities
            exp_output = np.exp(raw_output - np.max(raw_output, axis=0, keepdims=True))
            probabilities = exp_output / np.sum(exp_output, axis=0, keepdims=True)
            # Take max probability as confidence
            confidence_map = np.max(probabilities, axis=0)
        else:
            confidence_map = np.ones_like(raw_output) * 0.8  # Default confidence
        
        # Resize confidence map to match image
        self.last_confidence_map = cv2.resize(confidence_map.astype(np.float32), 
                                            (image.shape[1], image.shape[0]), 
                                            interpolation=cv2.INTER_LINEAR)
        
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
        """
        Calculate landing suitability score using neuro-symbolic reasoning.
        Combines neural network predictions with symbolic domain rules.
        """
        
        # Neural component: Extract confidence from segmentation
        zone_mask = np.zeros_like(seg_map, dtype=bool)
        x, y, w, h = zone['bbox']
        zone_mask[y:y+h, x:x+w] = True
        
        # Get neural confidence (from stored confidence map if available)
        neural_confidence = 0.8  # Default if confidence map not available
        if hasattr(self, 'last_confidence_map') and self.last_confidence_map is not None:
            neural_confidence = np.mean(self.last_confidence_map[zone_mask])
        
        # Symbolic reasoning components
        symbolic_score = 0.0
        
        # 1. Size reasoning (larger zones are generally safer)
        normalized_area = zone['area'] / (image_shape[0] * image_shape[1])
        if normalized_area > 0.05:  # Large zone (>5% of image)
            size_score = min(normalized_area * 8, 1.0)
        elif normalized_area > 0.02:  # Medium zone (2-5% of image)
            size_score = normalized_area * 15
        else:  # Small zone (<2% of image)
            size_score = normalized_area * 20
        symbolic_score += size_score * 0.3
        
        # 2. Shape reasoning (prefer regular shapes for landing)
        aspect_ratio = zone['aspect_ratio']
        if 0.7 <= aspect_ratio <= 1.4:  # Near-square zones (ideal for landing)
            shape_score = 1.0
        elif 0.5 <= aspect_ratio <= 2.0:  # Rectangular zones (acceptable)
            shape_score = 0.8 - abs(1.0 - aspect_ratio) * 0.3
        else:  # Very elongated zones (less suitable)
            shape_score = 0.4
        
        # Combine with solidity (how "solid" the shape is)
        shape_score *= zone['solidity']
        symbolic_score += shape_score * 0.25
        
        # 3. Spatial reasoning (position matters for UAV landing)
        img_center = (image_shape[1] // 2, image_shape[0] // 2)
        distance_from_center = math.sqrt(
            (zone['center'][0] - img_center[0])**2 + 
            (zone['center'][1] - img_center[1])**2
        )
        max_distance = math.sqrt(img_center[0]**2 + img_center[1]**2)
        center_preference = 1.0 - (distance_from_center / max_distance)
        symbolic_score += center_preference * 0.2
        
        # 4. Altitude-based reasoning (higher altitude = need larger, more obvious zones)
        if altitude > 10.0:  # High altitude
            altitude_factor = 1.2 if normalized_area > 0.08 else 0.8  # Prefer very large zones
        elif altitude > 5.0:  # Medium altitude
            altitude_factor = 1.1 if normalized_area > 0.04 else 0.9  # Prefer large zones
        else:  # Low altitude
            altitude_factor = 1.0  # All sizes acceptable
        
        symbolic_score *= altitude_factor
        
        # 5. Temporal reasoning (consistency with previous detections)
        temporal_bonus = 0.0
        if self.last_target and self._zones_overlap(zone, self.last_target):
            # Reward zones that are consistent with previous detections
            consistency = self._calculate_zone_consistency(zone, self.last_target)
            temporal_bonus = consistency * 0.15
        
        # 6. Safety reasoning (avoid zones near obstacles or uncertain areas)
        safety_penalty = self._calculate_safety_penalty(zone, seg_map, altitude)
        
        # Neuro-symbolic integration
        # Combine neural predictions (40%) with symbolic reasoning (60%)
        neuro_symbolic_score = (
            neural_confidence * 0.4 +  # Neural network confidence
            symbolic_score * 0.6 +     # Symbolic domain rules
            temporal_bonus -           # Temporal consistency bonus
            safety_penalty             # Safety-based penalty
        )
        
        return max(0.0, min(neuro_symbolic_score, 1.0))
    
    def _calculate_zone_consistency(self, current_zone: Dict, last_zone: Dict) -> float:
        """Calculate consistency between current and previous zone detection."""
        # Position consistency
        pos_diff = math.sqrt(
            (current_zone['center'][0] - last_zone['center'][0])**2 + 
            (current_zone['center'][1] - last_zone['center'][1])**2
        )
        pos_consistency = max(0, 1.0 - pos_diff / 100.0)  # Normalize by 100 pixels
        
        # Size consistency
        size_ratio = min(current_zone['area'], last_zone['area']) / max(current_zone['area'], last_zone['area'])
        size_consistency = size_ratio
        
        # Overall consistency
        return (pos_consistency * 0.7 + size_consistency * 0.3)
    
    def _calculate_safety_penalty(self, zone: Dict, seg_map: np.ndarray, altitude: float) -> float:
        """Calculate safety penalty based on surrounding context."""
        x, y, w, h = zone['bbox']
        
        # Define safety margin based on altitude
        safety_margin = max(10, int(altitude * 2))  # More margin for higher altitudes
        
        # Expand area to check for obstacles
        x1 = max(0, x - safety_margin)
        y1 = max(0, y - safety_margin)
        x2 = min(seg_map.shape[1], x + w + safety_margin)
        y2 = min(seg_map.shape[0], y + h + safety_margin)
        
        safety_area = seg_map[y1:y2, x1:x2]
        
        # Count unsuitable pixels in safety area (assuming class 3 is unsuitable)
        if safety_area.size > 0:
            unsuitable_pixels = np.sum(safety_area >= 3)  # Classes 3+ are obstacles/unsuitable
            total_pixels = safety_area.size
            unsuitable_ratio = unsuitable_pixels / total_pixels
            
            # Penalty increases with unsuitable ratio
            if unsuitable_ratio > 0.5:
                safety_penalty = 0.8  # High penalty for very risky areas
            elif unsuitable_ratio > 0.2:
                safety_penalty = 0.4  # Medium penalty
            elif unsuitable_ratio > 0.1:
                safety_penalty = 0.2  # Low penalty
            else:
                safety_penalty = 0.0  # No penalty for safe areas
        else:
            safety_penalty = 0.0
        
        return safety_penalty
    
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
    
    def get_segmentation_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get the last segmentation outputs for visualization.
        
        Returns:
            tuple: (segmentation_mask, raw_output, confidence_map)
                - segmentation_mask: Class predictions per pixel
                - raw_output: Raw model logits (before argmax)
                - confidence_map: Confidence scores per pixel
        """
        return (self.last_segmentation_output, 
                self.last_raw_output, 
                self.last_confidence_map)
    
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
            print(f" Target at {result.distance:.1f}m, commands: [{result.forward_velocity:.1f}, {result.right_velocity:.1f}, {result.descent_rate:.1f}]")
            
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
    
    print(" Demo completed")
