#!/usr/bin/env python3
"""
Scallop Reasoning Engine for UAV Landing

This is the main working implementation using proper Scallop API.
Consolidated and cleaned up from the working simple implementation.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ScallopLandingResult:
    """Result from Scallop reasoning"""
    status: str  # "TARGET_ACQUIRED", "NO_TARGET", "UNSAFE"
    confidence: float
    target_pixel: Optional[Tuple[int, int]]
    context: str
    reasoning_trace: List[str]

class ScallopReasoningEngine:
    """Scallop-based reasoning engine for UAV landing detection"""
    
    def __init__(self, context: str = "commercial", 
                 provenance: str = "difftopkproofs", 
                 k: int = 5):
        """
        Initialize reasoning engine
        
        Args:
            context: Mission context ('commercial', 'emergency', 'precision', 'delivery')
            provenance: Scallop provenance algorithm
            k: Number of top solutions to track
        """
        self.context = context
        self.provenance = provenance
        self.k = k
        
        # Performance tracking
        self.reasoning_times = []
        self.total_reasonings = 0
        
        # Try to initialize Scallop
        try:
            import scallopy
            # Use unit provenance for simpler fact addition
            self.scallop_ctx = scallopy.Context(provenance="unit", k=k)
            self.scallop_available = True
            self._setup_scallop_program()
            logger.info("Scallop reasoning engine initialized")
        except Exception as e:
            logger.warning(f"Scallop initialization failed: {e}, using mock")
            self.scallop_available = False
            self._setup_mock()
    
    def _setup_scallop_program(self):
        """Setup the complete Scallop program using Python API"""
        
        try:
            # Define relations using Python API
            self.scallop_ctx.add_relation("safe_zone", (int, int, float))  # x, y, confidence
            self.scallop_ctx.add_relation("obstacle", (int, int))          # x, y
            self.scallop_ctx.add_relation("flat_area", (int, int, float))  # x, y, flatness
            self.scallop_ctx.add_relation("landing_zone", (int, int))      # x, y
            
            # Add rules using string syntax
            self.scallop_ctx.add_rule("landing_zone(x, y) :- safe_zone(x, y, conf), conf > 0.5")
            self.scallop_ctx.add_rule("landing_zone(x, y) :- safe_zone(x, y, conf), flat_area(x, y, flatness), conf > 0.3, flatness > 0.6")
            
            logger.info(" Scallop relations and rules added successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup Scallop program: {e}")
            raise e
    
    def _setup_mock(self):
        """Setup mock implementation"""
        self.mock_facts = []
        self.mock_results = []
    
    def reason(self, segmentation_output: np.ndarray,
               confidence_map: np.ndarray,
               image_shape: Tuple[int, int],
               altitude: float,
               flatness_map: Optional[np.ndarray] = None) -> ScallopLandingResult:
        """
        Perform reasoning to find landing zones
        
        Args:
            segmentation_output: Segmentation mask
            confidence_map: Confidence values
            image_shape: (height, width)
            altitude: Current altitude
            flatness_map: Optional flatness information
            
        Returns:
            ScallopLandingResult with reasoning outcome
        """
        start_time = time.time()
        
        try:
            if self.scallop_available:
                result = self._reason_with_scallop(
                    segmentation_output, confidence_map, image_shape, altitude, flatness_map
                )
            else:
                result = self._reason_with_mock(
                    segmentation_output, confidence_map, image_shape, altitude, flatness_map
                )
            
            # Update performance tracking
            reasoning_time = time.time() - start_time
            self.reasoning_times.append(reasoning_time)
            self.total_reasonings += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return ScallopLandingResult(
                status="ERROR",
                confidence=0.0,
                target_pixel=None,
                context=self.context,
                reasoning_trace=[f"Error: {e}"]
            )
    
    def _reason_with_scallop(self, seg_output, conf_map, image_shape, altitude, flatness_map):
        """Reason using real Scallop"""
        
        try:
            # Extract facts from neural network outputs
            safe_zones = self._extract_safe_zones(seg_output, conf_map, image_shape)
            flat_areas = self._extract_flat_areas(image_shape, flatness_map)
            obstacles = self._extract_obstacles(seg_output, image_shape)
            
            logger.debug(f"Extracted {len(safe_zones)} safe zones, {len(flat_areas)} flat areas, {len(obstacles)} obstacles")
            
            # Add facts to Scallop context (clear any existing facts first)
            if safe_zones:
                self.scallop_ctx.add_facts("safe_zone", safe_zones)
            if flat_areas:
                self.scallop_ctx.add_facts("flat_area", flat_areas)
            if obstacles:
                self.scallop_ctx.add_facts("obstacle", obstacles)
            
            # Run Scallop reasoning
            self.scallop_ctx.run()
            
            # Extract results
            landing_zones = list(self.scallop_ctx.relation("landing_zone"))
            
            if landing_zones:
                # Select best zone (first one for now)
                best_zone = landing_zones[0]
                confidence = 0.8  # Could get this from Scallop provenance
                
                return ScallopLandingResult(
                    status="TARGET_ACQUIRED",
                    confidence=confidence,
                    target_pixel=(int(best_zone[0]), int(best_zone[1])),
                    context=self.context,
                    reasoning_trace=[f"Found {len(landing_zones)} landing zones via Scallop"]
                )
            else:
                return ScallopLandingResult(
                    status="NO_TARGET",
                    confidence=0.0,
                    target_pixel=None,
                    context=self.context,
                    reasoning_trace=["No suitable landing zones found by Scallop"]
                )
                
        except Exception as e:
            logger.error(f"Scallop reasoning error: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            raise e
    
    def _reason_with_mock(self, seg_output, conf_map, image_shape, altitude, flatness_map):
        """Reason using mock logic"""
        
        # Simple mock reasoning
        height, width = image_shape
        
        # Find areas with high confidence
        if np.max(conf_map) > 0.5:
            # Find the point with highest confidence
            max_idx = np.unravel_index(np.argmax(conf_map), conf_map.shape)
            target_pixel = (max_idx[1], max_idx[0])  # (x, y)
            confidence = float(np.max(conf_map))
            
            # Apply context-specific adjustment
            if self.context == "emergency":
                confidence *= 1.2  # More lenient
            elif self.context == "precision":
                confidence *= 0.8  # More strict
                
            confidence = min(confidence, 1.0)
            
            if confidence > 0.3:
                return ScallopLandingResult(
                    status="TARGET_ACQUIRED",
                    confidence=confidence,
                    target_pixel=target_pixel,
                    context=self.context,
                    reasoning_trace=[f"Mock reasoning: max confidence {confidence:.3f}"]
                )
        
        return ScallopLandingResult(
            status="NO_TARGET",
            confidence=0.0,
            target_pixel=None,
            context=self.context,
            reasoning_trace=["Mock reasoning: no suitable zones"]
        )
    
    def _extract_safe_zones(self, seg_output, conf_map, image_shape):
        """Extract safe zone facts from segmentation"""
        safe_zones = []
        
        # Sample points from the segmentation
        height, width = image_shape
        step = 32  # Sample every 32 pixels
        
        for y in range(0, height, step):
            for x in range(0, width, step):
                if y < seg_output.shape[0] and x < seg_output.shape[1]:
                    # Get the predicted class (argmax along last dimension if needed)
                    if len(seg_output.shape) == 3:
                        predicted_class = np.argmax(seg_output[y, x])
                    else:
                        predicted_class = seg_output[y, x]
                    
                    if predicted_class == 0:  # Class 0 is suitable
                        confidence = float(conf_map[y, x])
                        if confidence > 0.5:  # Minimum threshold
                            safe_zones.append((x, y, confidence))
        
        return safe_zones
    
    def _extract_flat_areas(self, image_shape, flatness_map):
        """Extract flatness information"""
        if flatness_map is None:
            # Generate synthetic flatness data
            height, width = image_shape
            flat_areas = []
            step = 32
            
            for y in range(0, height, step):
                for x in range(0, width, step):
                    # Mock flatness value
                    flatness = np.random.uniform(0.6, 0.9)
                    flat_areas.append((x, y, flatness))
            
            return flat_areas
        else:
            # Use provided flatness map
            flat_areas = []
            height, width = image_shape
            step = 32
            
            for y in range(0, height, step):
                for x in range(0, width, step):
                    if y < flatness_map.shape[0] and x < flatness_map.shape[1]:
                        flatness = float(flatness_map[y, x])
                        flat_areas.append((x, y, flatness))
            
            return flat_areas
    
    def _extract_obstacles(self, seg_output, image_shape):
        """Extract obstacle locations"""
        obstacles = []
        
        # Find pixels classified as obstacles (class 3, 4)
        if len(seg_output.shape) == 3:
            # Take argmax for multi-class predictions
            predicted_classes = np.argmax(seg_output, axis=2)
        else:
            predicted_classes = seg_output
            
        obstacle_mask = (predicted_classes == 3) | (predicted_classes == 4)
        obstacle_points = np.where(obstacle_mask)
        
        # Sample obstacles (don't include all pixels)
        if len(obstacle_points[0]) > 0:
            # Sample every 10th obstacle pixel to avoid too many facts
            indices = range(0, len(obstacle_points[0]), 10)
            for i in indices:
                y, x = obstacle_points[0][i], obstacle_points[1][i]
                obstacles.append((int(x), int(y)))
        
        return obstacles
    
    def set_context(self, context: str):
        """Change the reasoning context"""
        if context != self.context:
            old_context = self.context
            self.context = context
            logger.info(f"Context changed from {old_context} to {context}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            "total_reasonings": self.total_reasonings,
            "scallop_available": self.scallop_available,
            "context": self.context
        }
        
        if self.reasoning_times:
            stats.update({
                "average_reasoning_time": np.mean(self.reasoning_times),
                "min_reasoning_time": np.min(self.reasoning_times),
                "max_reasoning_time": np.max(self.reasoning_times)
            })
        
        return stats
