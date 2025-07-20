#!/usr/bin/env python3
"""
Enhanced UAV Landing Detector with Scallop Integration

This module extends the existing UAVLandingDetector with Scallop-based
neuro-symbolic reasoning capabilities for improved decision making.
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Import with fallback for different execution contexts
try:
    from .uav_landing_detector import UAVLandingDetector, LandingResult
    from .scallop_reasoning_engine import ScallopReasoningEngine, ScallopLandingResult
except ImportError:
    # Fallback for direct execution or testing
    try:
        from uav_landing_detector import UAVLandingDetector, LandingResult
        from scallop_reasoning_engine import ScallopReasoningEngine, ScallopLandingResult
    except ImportError:
        # Create mock classes for testing
        class LandingResult:
            def __init__(self, status="UNKNOWN", confidence=0.0, **kwargs):
                self.status = status
                self.confidence = confidence
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class UAVLandingDetector:
            def __init__(self, **kwargs):
                self.enable_visualization = False
                self.safety_margin = 0.5
            
            def _run_segmentation(self, image):
                return np.zeros((512, 512), dtype=np.uint8)
            
            def _find_landing_zones(self, seg_map):
                # Create mock zones with proper structure
                return [
                    {
                        'id': 0,
                        'center': (256, 256),
                        'area': 10000,
                        'bbox': (200, 200, 112, 112),
                        'contour': None,
                        'aspect_ratio': 1.0,
                        'solidity': 0.8
                    }
                ]
            
            def _is_zone_safe(self, zone, seg_map, altitude=None):
                return True
            
            def _calculate_zone_score(self, zone, seg_map, image_shape, altitude):
                return 0.5
            
            def _pixel_to_world(self, pixel, altitude):
                return (0.0, 0.0, altitude)
            
            def _calculate_distance(self, pixel, altitude):
                return altitude
            
            def _calculate_bearing(self, pixel):
                return 0.0
            
            def _generate_commands(self, result, altitude, velocity):
                return result
            
            def _update_tracking(self, result):
                pass
            
            def _update_fps(self, processing_time):
                return 30.0
            
            def _create_visualization(self, image, seg_map, result):
                return image.copy()
            
            def get_performance_stats(self):
                return {"fps": 30.0}
            
            def get_segmentation_data(self):
                return (None, None, None)
            
            def reset_state(self):
                pass
        
        from scallop_reasoning_engine import ScallopReasoningEngine, ScallopLandingResult

logger = logging.getLogger(__name__)

class EnhancedUAVDetector(UAVLandingDetector):
    """UAV Landing Detector with Scallop-based neuro-symbolic reasoning"""
    
    def __init__(self, context: str = "commercial", 
                 use_scallop: bool = True,
                 scallop_provenance: str = "difftopkproofs",
                 scallop_k: int = 5,
                 **kwargs):
        """
        Initialize Enhanced UAV Detector
        
        Args:
            context: Mission context ('commercial', 'emergency', 'precision', 'delivery')
            use_scallop: Whether to use Scallop reasoning (fallback to original if False)
            scallop_provenance: Scallop provenance algorithm
            scallop_k: Number of top proofs/solutions to track
            **kwargs: Arguments for base UAVLandingDetector
        """
        super().__init__(**kwargs)
        
        self.context = context
        self.use_scallop = use_scallop
        
        # Initialize Scallop reasoning engine
        if self.use_scallop:
            try:
                self.scallop_engine = ScallopReasoningEngine(
                    context=context,
                    provenance=scallop_provenance, 
                    k=scallop_k
                )
                self.scallop_available = True
                logger.info(f"Enhanced UAV Detector initialized with Scallop reasoning")
            except Exception as e:
                logger.warning(f"Failed to initialize Scallop engine: {e}, falling back to original")
                self.scallop_available = False
                self.use_scallop = False
        else:
            self.scallop_available = False
            logger.info("Enhanced UAV Detector initialized without Scallop")
        
        # Performance tracking
        self.scallop_reasoning_times = []
        self.fallback_count = 0
        
    def process_frame(self, image: np.ndarray, altitude: float,
                     current_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> LandingResult:
        """
        Process single frame with enhanced neuro-symbolic reasoning
        
        Args:
            image: Input BGR image from camera
            altitude: Current altitude above ground (meters)
            current_velocity: Current velocity [vx, vy, vz] (m/s)
            
        Returns:
            LandingResult with enhanced reasoning
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
            
            # Step 2: Enhanced Zone Evaluation with Scallop
            if self.use_scallop and self.scallop_available:
                result = self._evaluate_zones_enhanced_scallop(
                    segmentation_map, image, altitude
                )
                
                if result.status == "ERROR":
                    # Fallback to original method
                    logger.warning("Scallop reasoning failed, falling back to original method")
                    self.fallback_count += 1
                    result = self._evaluate_zones_enhanced_fallback(
                        segmentation_map, image, altitude
                    )
            else:
                # Use enhanced fallback method
                result = self._evaluate_zones_enhanced_fallback(
                    segmentation_map, image, altitude
                )
            
            # Step 3: Generate Navigation Commands
            result = self._generate_commands(result, altitude, current_velocity)
            
            # Step 4: Update State Tracking
            self._update_tracking(result)
            
            # Step 5: Performance Tracking
            processing_time = (time.time() - start_time) * 1000
            result.processing_time = processing_time
            result.fps = self._update_fps(processing_time)
            
            # Step 6: Visualization (if enabled)
            if self.enable_visualization:
                result.annotated_image = self._create_enhanced_visualization(
                    image, segmentation_map, result
                )
                
            return result
            
        except Exception as e:
            logger.error(f"Enhanced processing error: {e}")
            return LandingResult(status="ERROR", confidence=0.0, 
                               processing_time=(time.time() - start_time) * 1000)
    
    def _evaluate_zones_enhanced_scallop(self, seg_map: np.ndarray, 
                                        image: np.ndarray, 
                                        altitude: float) -> LandingResult:
        """Enhanced zone evaluation using Scallop reasoning"""
        
        # Get segmentation confidence map
        _, _, confidence_map = self.get_segmentation_data()
        
        if confidence_map is None:
            # Create a simple confidence map if not available
            confidence_map = np.ones_like(seg_map, dtype=np.float32) * 0.8
            
        # Calculate flatness map for enhanced reasoning
        flatness_map = self._calculate_flatness_map(image)
        
        # Use Scallop for reasoning
        scallop_start = time.time()
        try:
            scallop_result = self.scallop_engine.reason(
                segmentation_output=seg_map,
                confidence_map=confidence_map,
                image_shape=image.shape[:2],
                altitude=altitude,
                flatness_map=flatness_map
            )
            
            scallop_time = time.time() - scallop_start
            self.scallop_reasoning_times.append(scallop_time)
            
            # Convert Scallop result to LandingResult
            return self._convert_scallop_result(scallop_result, altitude, image.shape[:2])
            
        except Exception as e:
            logger.error(f"Scallop reasoning failed: {e}")
            return LandingResult(status="ERROR", confidence=0.0)
    
    def _evaluate_zones_enhanced_fallback(self, seg_map: np.ndarray, 
                                         image: np.ndarray, 
                                         altitude: float) -> LandingResult:
        """Enhanced fallback evaluation with improved heuristics"""
        
        # Find landing zones using original method
        zones = self._find_landing_zones(seg_map)
        
        if not zones:
            return LandingResult(status="NO_TARGET", confidence=0.0)
        
        # Enhanced scoring with context awareness
        scored_zones = []
        flatness_map = self._calculate_flatness_map(image)
        
        for zone in zones:
            # Get enhanced score with context
            score = self._calculate_enhanced_zone_score(
                zone, seg_map, image.shape[:2], altitude, flatness_map
            )
            
            if score > self._get_context_threshold():
                scored_zones.append((score, zone))
        
        if not scored_zones:
            return LandingResult(status="NO_TARGET", confidence=0.0)
        
        # Select best zone
        scored_zones.sort(reverse=True, key=lambda x: x[0])
        best_score, best_zone = scored_zones[0]
        
        # Check safety with context-aware rules
        if self._is_zone_safe_enhanced(best_zone, seg_map, altitude):
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
    
    def _calculate_enhanced_zone_score(self, zone: Dict, seg_map: np.ndarray, 
                                     image_shape: Tuple[int, int], altitude: float,
                                     flatness_map: np.ndarray) -> float:
        """Enhanced zone scoring with context awareness"""
        
        try:
            # Validate zone has required fields - add 'area' if missing
            if 'area' not in zone and 'bbox' in zone:
                # Calculate area from bounding box
                x, y, w, h = zone['bbox']
                zone['area'] = w * h
            
            required_fields = ['bbox', 'center']
            for field in required_fields:
                if field not in zone:
                    logger.warning(f"Zone missing required field: {field}")
                    return 0.0
            
            # Use parent method for base calculation 
            if not self._is_zone_safe(zone, seg_map, altitude):
                if self.enable_debug_output:
                    logger.debug(f"Zone {zone.get('id', 'unknown')} marked as unsafe")
                    return 0.0
            
            base_score = self._calculate_zone_score(zone, seg_map, image_shape, altitude)
            
            # Add flatness scoring
            x, y, w, h = zone['bbox']
            roi_flatness = flatness_map[y:y+h, x:x+w]
            avg_flatness = np.mean(roi_flatness)
            flatness_bonus = avg_flatness * 0.2
            
            # Context-specific adjustments
            context_multiplier = self._get_context_multiplier()
            
            # Altitude-based adjustments
            altitude_factor = self._get_altitude_factor(altitude)
            
            enhanced_score = (base_score + flatness_bonus) * context_multiplier * altitude_factor
            
            return max(0.0, min(enhanced_score, 1.0))
            
        except Exception as e:
            logger.error(f"Enhanced zone scoring error: {e}")
            # Fallback to basic zone score calculation
            try:
                return self._calculate_zone_score(zone, seg_map, image_shape, altitude)
            except:
                return 0.0
    
    def _get_context_threshold(self) -> float:
        """Get minimum score threshold based on context"""
        thresholds = {
            "emergency": 0.2,    # More lenient
            "commercial": 0.3,   # Standard
            "precision": 0.5,    # Stricter  
            "delivery": 0.35     # Moderate
        }
        return thresholds.get(self.context, 0.3)
    
    def _get_context_multiplier(self) -> float:
        """Get score multiplier based on context"""
        multipliers = {
            "emergency": 1.3,    # Boost scores for urgency
            "commercial": 1.0,   # Standard
            "precision": 0.8,    # Be more conservative
            "delivery": 1.1      # Slightly optimistic
        }
        return multipliers.get(self.context, 1.0)
    
    def _get_altitude_factor(self, altitude: float) -> float:
        """Get altitude-based scoring factor"""
        if self.context == "emergency":
            return 1.0  # Altitude doesn't matter in emergency
        elif altitude > 15.0:
            return 0.9  # More cautious at high altitude
        elif altitude < 2.0:
            return 1.2  # Boost for low altitude precision
        else:
            return 1.0
    
    def _is_zone_safe_enhanced(self, zone: Dict, seg_map: np.ndarray, altitude: float) -> bool:
        """Enhanced safety check with context awareness"""
        
        # Base safety check
        base_safe = self._is_zone_safe(zone, seg_map, altitude)
        
        # Context-specific safety margins
        safety_margins = {
            "emergency": 0.5,    # Very lenient
            "commercial": 1.0,   # Standard
            "precision": 1.5,    # Stricter
            "delivery": 0.8      # Moderate
        }
        
        margin_factor = safety_margins.get(self.context, 1.0)
        
        # Apply context-specific safety logic
        if self.context == "emergency":
            # In emergency, only avoid critical obstacles
            return self._check_critical_obstacles_only(zone, seg_map)
        else:
            # Standard safety with context margin
            return base_safe and self._additional_safety_checks(zone, seg_map, margin_factor)
    
    def _check_critical_obstacles_only(self, zone: Dict, seg_map: np.ndarray) -> bool:
        """Check only for critical obstacles in emergency mode"""
        # This would check for people, vehicles, etc. 
        # For now, just use reduced safety margin
        return self._is_zone_safe(zone, seg_map, 0.1)  # Very small margin
    
    def _additional_safety_checks(self, zone: Dict, seg_map: np.ndarray, margin_factor: float) -> bool:
        """Additional context-specific safety checks"""
        
        # Adjust safety margin based on context
        adjusted_margin = self.safety_margin * margin_factor
        
        x, y, w, h = zone['bbox']
        margin = max(10, int(adjusted_margin * 10))  # pixels
        
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(seg_map.shape[1], x + w + margin)
        y2 = min(seg_map.shape[0], y + h + margin)
        
        roi = seg_map[y1:y2, x1:x2]
        
        # Check for obstacles with context-specific tolerances
        obstacle_pixels = np.sum((roi == 3) | (roi == 4))
        total_pixels = roi.size
        
        obstacle_ratio = obstacle_pixels / total_pixels if total_pixels > 0 else 0
        
        # Context-specific tolerance
        tolerance = {
            "emergency": 0.2,    # 20% obstacles OK
            "commercial": 0.1,   # 10% obstacles OK  
            "precision": 0.05,   # 5% obstacles OK
            "delivery": 0.15     # 15% obstacles OK
        }.get(self.context, 0.1)
        
        return obstacle_ratio < tolerance
    
    def _calculate_flatness_map(self, image: np.ndarray) -> np.ndarray:
        """Calculate flatness map for the entire image"""
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate gradients using Sobel
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize and invert (so higher values mean flatter)
        max_val = 100.0  # Empirical maximum value for normalization
        flatness_map = 1.0 - np.minimum(magnitude / max_val, 1.0)
        
        return flatness_map.astype(np.float32)
    
    def _convert_scallop_result(self, scallop_result: ScallopLandingResult, 
                               altitude: float, image_shape: Tuple[int, int]) -> LandingResult:
        """Convert Scallop result to standard LandingResult"""
        
        if scallop_result.status == "TARGET_ACQUIRED" and scallop_result.target_pixel:
            return LandingResult(
                status="TARGET_ACQUIRED",
                confidence=scallop_result.confidence,
                target_pixel=scallop_result.target_pixel,
                target_world=self._pixel_to_world(scallop_result.target_pixel, altitude),
                distance=self._calculate_distance(scallop_result.target_pixel, altitude),
                bearing=self._calculate_bearing(scallop_result.target_pixel)
            )
        elif scallop_result.status == "NO_TARGET":
            return LandingResult(status="NO_TARGET", confidence=scallop_result.confidence)
        elif scallop_result.status == "UNSAFE":
            return LandingResult(status="UNSAFE", confidence=scallop_result.confidence)
        else:
            return LandingResult(status="ERROR", confidence=0.0)
    
    def _create_enhanced_visualization(self, image: np.ndarray, seg_map: np.ndarray, 
                                     result: LandingResult) -> np.ndarray:
        """Create enhanced visualization with Scallop information"""
        
        # Start with base visualization
        vis = self._create_visualization(image, seg_map, result)
        
        # Add enhanced information
        self._add_context_info(vis)
        self._add_reasoning_info(vis)
        
        return vis
    
    def _add_context_info(self, vis: np.ndarray):
        """Add context information to visualization"""
        
        # Context indicator
        context_color = {
            "commercial": (0, 255, 0),   # Green
            "emergency": (0, 0, 255),    # Red  
            "precision": (255, 255, 0),  # Yellow
            "delivery": (255, 0, 255)    # Magenta
        }.get(self.context, (255, 255, 255))
        
        cv2.putText(vis, f"Context: {self.context.upper()}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, context_color, 2)
    
    def _add_reasoning_info(self, vis: np.ndarray):
        """Add reasoning engine information to visualization"""
        
        reasoning_text = "Scallop" if (self.use_scallop and self.scallop_available) else "Heuristic"
        reasoning_color = (0, 255, 255) if self.scallop_available else (255, 255, 255)
        
        cv2.putText(vis, f"Reasoning: {reasoning_text}", (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, reasoning_color, 2)
        
        # Add fallback count if applicable
        if self.fallback_count > 0:
            cv2.putText(vis, f"Fallbacks: {self.fallback_count}", (10, 210),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    
    def set_context(self, context: str):
        """Dynamically change reasoning context"""
        if context != self.context:
            old_context = self.context
            self.context = context
            
            # Update Scallop engine context if available
            if self.scallop_available:
                self.scallop_engine.set_context(context)
            
            logger.info(f"Context changed from {old_context} to {context}")
    
    def get_reasoning_explanation(self) -> Dict:
        """Get explanation of the reasoning process"""
        
        explanation = {
            "reasoning_engine": "Enhanced UAV Detector",
            "context": self.context,
            "scallop_available": getattr(self, 'scallop_available', False),
            "use_scallop": getattr(self, 'use_scallop', False),
            "fallback_count": getattr(self, 'fallback_count', 0)
        }
        
        if hasattr(self, 'scallop_available') and self.scallop_available and self.scallop_engine:
            try:
                scallop_stats = self.scallop_engine.get_performance_stats()
                explanation.update(scallop_stats)
                # Override with correct scallop availability status
                explanation["scallop_available"] = self.scallop_available
            except Exception as e:
                logger.warning(f"Failed to get Scallop performance stats: {e}")
            
        if hasattr(self, 'scallop_reasoning_times') and self.scallop_reasoning_times:
            explanation["avg_scallop_time"] = np.mean(self.scallop_reasoning_times)
            explanation["total_scallop_reasonings"] = len(self.scallop_reasoning_times)
        
        return explanation
    
    def get_enhanced_performance_stats(self) -> Dict[str, Any]:
        """Get enhanced performance statistics"""
        
        base_stats = self.get_performance_stats()
        enhanced_stats = {
            "context": self.context,
            "scallop_available": self.scallop_available,
            "use_scallop": self.use_scallop,
            "fallback_count": self.fallback_count,
            "scallop_reasoning_count": len(self.scallop_reasoning_times)
        }
        
        if self.scallop_reasoning_times:
            enhanced_stats.update({
                "avg_scallop_reasoning_time": np.mean(self.scallop_reasoning_times),
                "total_scallop_reasoning_time": np.sum(self.scallop_reasoning_times),
                "min_scallop_reasoning_time": np.min(self.scallop_reasoning_times),
                "max_scallop_reasoning_time": np.max(self.scallop_reasoning_times)
            })
        
        if self.scallop_available:
            enhanced_stats["scallop_engine_stats"] = self.scallop_engine.get_performance_stats()
        
        return {**base_stats, **enhanced_stats}
    
    def reset_state(self):
        """Reset detector state for new flight"""
        super().reset_state()
        self.scallop_reasoning_times.clear()
        self.fallback_count = 0
        logger.info("Enhanced detector state reset")

# Convenience function for creating enhanced detector
def create_enhanced_detector(context: str = "commercial", 
                           model_path: str = "models/bisenetv2_uav_landing.onnx",
                           **kwargs) -> EnhancedUAVDetector:
    """
    Create an enhanced UAV detector with recommended settings
    
    Args:
        context: Mission context
        model_path: Path to ONNX model
        **kwargs: Additional arguments
        
    Returns:
        EnhancedUAVDetector instance
    """
    
    return EnhancedUAVDetector(
        context=context,
        model_path=model_path,
        use_scallop=True,
        scallop_provenance="difftopkproofs",
        scallop_k=5,
        **kwargs
    )
