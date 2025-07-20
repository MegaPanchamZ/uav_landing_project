#!/usr/bin/env python3
"""
Scallop Neuro-Symbolic Reasoning Engine for UAV Landing Detection

This module implements the core neuro-symbolic reasoning engine using Scallop
for UAV landing site detection and selection.
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Try to import real Scallop, fall back to mock
try:
    import scallopy
    SCALLOP_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Using real Scallop implementation")
except ImportError:
    # Use our mock implementation
    from . import scallop_mock as scallopy
    SCALLOP_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Using mock Scallop implementation - install scallopy for full functionality")

@dataclass
class ScallopLandingResult:
    """Enhanced landing result with probabilistic reasoning"""
    status: str  # 'TARGET_ACQUIRED', 'NO_TARGET', 'UNSAFE'
    confidence: float  # 0.0-1.0
    target_pixel: Optional[Tuple[int, int]] = None  # (x, y) in image
    reasoning_trace: Dict = None  # Scallop proof trace
    alternative_sites: List[Tuple[Tuple[int, int], float]] = None
    context: str = "commercial"
    
    def __post_init__(self):
        if self.reasoning_trace is None:
            self.reasoning_trace = {}
        if self.alternative_sites is None:
            self.alternative_sites = []

class ScallopReasoningEngine:
    """Neuro-symbolic reasoning engine using Scallop for UAV landing detection"""
    
    def __init__(self, context: str = "commercial", 
                 provenance: str = "difftopkproofs", k: int = 5):
        """
        Initialize Scallop reasoning engine
        
        Args:
            context: Mission context ('commercial', 'emergency', 'precision', 'delivery')
            provenance: Scallop provenance algorithm
            k: Number of top proofs/solutions to track
        """
        self.context = context
        self.provenance = provenance
        self.k = k
        
        # Initialize Scallop context
        self.scallop_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
        self._setup_relations()
        self._setup_rules()
        
        # Performance tracking
        self.reasoning_count = 0
        self.total_reasoning_time = 0.0
        
        logger.info(f"ScallopReasoningEngine initialized: context={context}, "
                   f"provenance={provenance}, k={k}, real_scallop={SCALLOP_AVAILABLE}")
    
    def _setup_relations(self):
        """Define Scallop relations for landing detection"""
        
        # Input relations from neural network
        self.scallop_ctx.add_relation("seg_result", (int, int, str, float))  # x, y, class, confidence
        self.scallop_ctx.add_relation("image_info", (int, int, float))       # width, height, altitude
        self.scallop_ctx.add_relation("context_info", (str,))                # mission context
        
        # Spatial relations  
        self.scallop_ctx.add_relation("obstacle", (int, int, str))           # x, y, obstacle_type
        self.scallop_ctx.add_relation("safe_area", (int, int, int, int))     # x1, y1, x2, y2 (bbox)
        self.scallop_ctx.add_relation("flatness", (int, int, float))         # x, y, flatness_score
        
        # Derived relations
        self.scallop_ctx.add_relation("suitable_zone", (int, int))           # x, y
        self.scallop_ctx.add_relation("marginal_zone", (int, int))           # x, y  
        self.scallop_ctx.add_relation("safe_from_obstacles", (int, int))     # x, y
        self.scallop_ctx.add_relation("obstacle_nearby", (int, int))         # x, y
        
        # Output relations
        self.scallop_ctx.add_relation("landing_candidate", (int, int, float)) # x, y, score
        self.scallop_ctx.add_relation("best_landing_site", (int, int))         # x, y
        self.scallop_ctx.add_relation("top_landing_sites", (int, int, int))    # x, y, rank
        
        logger.debug("Scallop relations defined")
    
    def _setup_rules(self):
        """Setup rules based on context"""
        
        if self.scallop_available:
            # Add type declarations first
            self.scallop_ctx.add_program("""
                // Define types for our predicates
                type safe_zone(i32, i32, f32)
                type obstacle(i32, i32) 
                type flat_area(i32, i32, f32)
                type landing_candidate(i32, i32, f32)
                type landing_zone(i32, i32)
                type unsafe_zone(i32, i32)
            """)
        
        # Setup context-specific rules
        if self.context == "commercial":
            self._setup_commercial_rules()
        elif self.context == "emergency":
            self._setup_emergency_rules()
        elif self.context == "precision":
            self._setup_precision_rules()
        elif self.context == "delivery":
            self._setup_delivery_rules()
    
    def _setup_commercial_rules(self):
        """Commercial flight rules: balance safety, efficiency, and comfort"""
        
        if self.scallop_available:
            # Basic suitability rules
            self.scallop_ctx.add_rule("landing_candidate(x, y, score) :- safe_zone(x, y, conf), flat_area(x, y, flatness), score = conf * 0.7 + flatness * 0.3")
            self.scallop_ctx.add_rule("landing_zone(x, y) :- landing_candidate(x, y, score), score > 0.3")
            self.scallop_ctx.add_rule("unsafe_zone(x, y) :- obstacle(ox, oy), |x - ox| < 30, |y - oy| < 30")
        else:
            # Mock rules
            self.rules.extend([
                "landing_candidate(x, y, score) :- safe_zone(x, y, conf), flat_area(x, y, flatness), score = conf * 0.7 + flatness * 0.3",
                "landing_zone(x, y) :- landing_candidate(x, y, score), score > 0.3",
                "unsafe_zone(x, y) :- obstacle(ox, oy), |x - ox| < 30, |y - oy| < 30"
            ])
    
    def _setup_emergency_rules(self):
        """Emergency landing rules: prioritize immediate safety"""
        
        # More permissive suitability
        self.scallop_ctx.add_rule(
            'suitable_zone(x, y) = seg_result(x, y, "suitable", conf) and conf > 0.5',
            tag=0.8
        )
        
        self.scallop_ctx.add_rule(
            'marginal_zone(x, y) = seg_result(x, y, "marginal", conf) and conf > 0.4',
            tag=0.6
        )
        
        # Critical safety - avoid people/vehicles only (smaller safety margin)
        self.scallop_ctx.add_rule(
            'obstacle_nearby(x, y) = obstacle(ox, oy, type) and x >= ox - 15 and x <= ox + 15 and y >= oy - 15 and y <= oy + 15'
        )
        
        # More aggressive scoring for emergency
        self.scallop_ctx.add_rule(
            'landing_candidate(x, y, 0.9) = suitable_zone(x, y) and not obstacle_nearby(x, y)',
            tag=0.9
        )
        
        self.scallop_ctx.add_rule(
            'landing_candidate(x, y, 0.7) = marginal_zone(x, y) and not obstacle_nearby(x, y)',
            tag=0.7
        )
    
    def _setup_precision_rules(self):
        """Precision landing rules: high accuracy requirements"""
        
        # Stricter requirements
        self.scallop_ctx.add_rule(
            'suitable_zone(x, y) = seg_result(x, y, "suitable", conf) and conf > 0.85',
            tag=0.95
        )
        
        # Larger safety margins
        self.scallop_ctx.add_rule(
            'obstacle_nearby(x, y) = obstacle(ox, oy, _) and x >= ox - 35 and x <= ox + 35 and y >= oy - 35 and y <= oy + 35'
        )
        
        self.scallop_ctx.add_rule(
            'landing_candidate(x, y, 0.95) = suitable_zone(x, y) and not obstacle_nearby(x, y)',
            tag=0.95
        )
    
    def _setup_delivery_rules(self):
        """Delivery mission rules: focus on accessibility and ground clearance"""
        
        self.scallop_ctx.add_rule(
            'suitable_zone(x, y) = seg_result(x, y, "suitable", conf) and conf > 0.75',
            tag=0.85
        )
        
        # Medium safety margin for delivery
        self.scallop_ctx.add_rule(
            'obstacle_nearby(x, y) = obstacle(ox, oy, _) and x >= ox - 20 and x <= ox + 20 and y >= oy - 20 and y <= oy + 20'
        )
        
        self.scallop_ctx.add_rule(
            'landing_candidate(x, y, 0.85) = suitable_zone(x, y) and not obstacle_nearby(x, y)',
            tag=0.85
        )
    
    def _setup_default_rules(self):
        """Default rules for unknown contexts"""
        self._setup_commercial_rules()  # Use commercial as default
    
    def reason(self, segmentation_output: np.ndarray, 
               confidence_map: np.ndarray,
               image_shape: Tuple[int, int],
               altitude: float,
               flatness_map: Optional[np.ndarray] = None) -> ScallopLandingResult:
        """
        Execute neuro-symbolic reasoning for landing site selection
        
        Args:
            segmentation_output: Semantic segmentation class predictions
            confidence_map: Per-pixel confidence scores
            image_shape: (height, width) of the image
            altitude: UAV altitude in meters
            flatness_map: Optional flatness analysis
            
        Returns:
            ScallopLandingResult with reasoning results
        """
        import time
        start_time = time.time()
        
        try:
            # Clear previous facts
            self.scallop_ctx.clear_facts()
            
            # Add neural network outputs as probabilistic facts
            self._add_segmentation_facts(segmentation_output, confidence_map, image_shape)
            self._add_context_facts(image_shape, altitude)
            self._add_spatial_facts(segmentation_output, image_shape, flatness_map)
            
            # Execute reasoning
            self.scallop_ctx.run()
            
            # Extract results
            landing_sites = list(self.scallop_ctx.relation("landing_candidate"))
            best_site = list(self.scallop_ctx.relation("best_landing_site"))
            top_sites = list(self.scallop_ctx.relation("top_landing_sites"))
            
            # Create result
            result = self._create_result(landing_sites, best_site, top_sites)
            
            # Update performance tracking
            reasoning_time = time.time() - start_time
            self.reasoning_count += 1
            self.total_reasoning_time += reasoning_time
            
            logger.debug(f"Reasoning completed in {reasoning_time:.3f}s, "
                        f"found {len(landing_sites)} candidates")
            
            return result
            
        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return ScallopLandingResult(
                status="ERROR",
                confidence=0.0,
                context=self.context
            )
    
    def _add_segmentation_facts(self, seg_map: np.ndarray, 
                              confidence_map: np.ndarray, 
                              image_shape: Tuple[int, int]):
        """Convert segmentation output to Scallop facts"""
        
        seg_facts = []
        height, width = image_shape
        
        # Sample points for efficiency (every N pixels)
        step = max(1, min(height, width) // 50)  # ~50 sample points per dimension
        
        class_names = {1: "suitable", 2: "marginal", 3: "obstacle", 0: "background"}
        
        for y in range(0, height, step):
            for x in range(0, width, step):
                class_id = seg_map[y, x]
                if class_id in class_names and class_id > 0:  # Skip background
                    confidence = float(confidence_map[y, x])
                    if confidence > 0.3:  # Only add confident predictions
                        seg_facts.append((x, y, class_names[class_id], confidence))
        
        self.scallop_ctx.add_facts("seg_result", seg_facts)
        logger.debug(f"Added {len(seg_facts)} segmentation facts")
        
    def _add_context_facts(self, image_shape: Tuple[int, int], altitude: float):
        """Add contextual information"""
        
        height, width = image_shape
        self.scallop_ctx.add_facts("image_info", [(width, height, altitude)])
        self.scallop_ctx.add_facts("context_info", [(self.context,)])
        
    def _add_spatial_facts(self, seg_map: np.ndarray, image_shape: Tuple[int, int],
                          flatness_map: Optional[np.ndarray] = None):
        """Add spatial reasoning facts (obstacles, safe areas, flatness)"""
        
        # Find obstacles
        obstacles = []
        height, width = image_shape
        
        # Sample obstacle locations
        step = max(1, min(height, width) // 30)
        for y in range(0, height, step):
            for x in range(0, width, step):
                if seg_map[y, x] == 3:  # Obstacle class
                    obstacles.append((x, y, "generic"))
        
        self.scallop_ctx.add_facts("obstacle", obstacles)
        
        # Add flatness information if available
        if flatness_map is not None:
            flatness_facts = []
            for y in range(0, height, step):
                for x in range(0, width, step):
                    flatness_score = float(flatness_map[y, x])
                    if flatness_score > 0.5:  # Only add reasonably flat areas
                        flatness_facts.append((x, y, flatness_score))
            
            self.scallop_ctx.add_facts("flatness", flatness_facts)
            logger.debug(f"Added {len(flatness_facts)} flatness facts")
        
        logger.debug(f"Added {len(obstacles)} obstacle facts")
    
    def _create_result(self, landing_sites: List, best_site: List, 
                      top_sites: List) -> ScallopLandingResult:
        """Create ScallopLandingResult from Scallop output"""
        
        if best_site:
            target_pixel = (int(best_site[0][0]), int(best_site[0][1]))
            
            # Get confidence from landing candidates
            target_confidence = 0.0
            for site in landing_sites:
                if len(site) >= 3 and site[0] == target_pixel[0] and site[1] == target_pixel[1]:
                    target_confidence = float(site[2])
                    break
                    
            # Get alternative sites
            alternatives = []
            for site in landing_sites:
                if len(site) >= 3:
                    x, y, score = site[0], site[1], site[2]
                    if (x, y) != target_pixel:
                        alternatives.append(((int(x), int(y)), float(score)))
            
            alternatives.sort(key=lambda x: x[1], reverse=True)
            
            return ScallopLandingResult(
                status="TARGET_ACQUIRED",
                confidence=target_confidence,
                target_pixel=target_pixel,
                reasoning_trace=self._get_reasoning_trace(),
                alternative_sites=alternatives[:3],  # Top 3 alternatives
                context=self.context
            )
        else:
            return ScallopLandingResult(
                status="NO_TARGET",
                confidence=0.0,
                reasoning_trace=self._get_reasoning_trace(),
                alternative_sites=[],
                context=self.context
            )
    
    def _get_reasoning_trace(self) -> Dict:
        """Get explanation trace from Scallop reasoning"""
        
        # In a real implementation, this would extract proof traces
        # For now, return basic statistics
        return {
            "reasoning_engine": "Scallop",
            "context": self.context,
            "provenance": self.provenance,
            "k": self.k,
            "scallop_available": SCALLOP_AVAILABLE,
            "reasoning_count": self.reasoning_count,
            "avg_reasoning_time": (self.total_reasoning_time / self.reasoning_count 
                                 if self.reasoning_count > 0 else 0.0)
        }
    
    def set_context(self, context: str):
        """Dynamically change reasoning context"""
        if context != self.context:
            self.context = context
            # Recreate context with new rules
            self.scallop_ctx = scallopy.ScallopContext(
                provenance=self.provenance, k=self.k
            )
            self._setup_relations()
            self._setup_rules()
            logger.info(f"Context changed to: {context}")
    
    def update_rule_weights(self, weights: Dict[str, float]):
        """Update rule weights for different contexts (for training)"""
        # This would update rule probabilities for differentiable training
        # Implementation depends on the specific training setup
        logger.info(f"Rule weights update requested: {weights}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "reasoning_count": self.reasoning_count,
            "total_reasoning_time": self.total_reasoning_time,
            "avg_reasoning_time": (self.total_reasoning_time / self.reasoning_count 
                                 if self.reasoning_count > 0 else 0.0),
            "context": self.context,
            "scallop_available": SCALLOP_AVAILABLE
        }

class ScallopTrainingModule(torch.nn.Module):
    """Differentiable Scallop module for end-to-end training"""
    
    def __init__(self, context: str = "commercial", k: int = 3):
        super().__init__()
        self.context = context
        self.k = k
        
        if SCALLOP_AVAILABLE:
            # Create differentiable Scallop module
            self.scallop_module = scallopy.Module(
                program=self._get_program_string(),
                input_mappings={
                    "seg_confidence": list(range(100)),  # 0-99 confidence levels
                    "spatial_feature": list(range(50))   # Spatial feature indices
                },
                output_mappings={
                    "landing_score": list(range(100))    # 0-99 score levels
                },
                provenance="difftopkproofs",
                k=k
            )
        else:
            # Use mock module
            self.scallop_module = scallopy.Module(
                program="mock_program",
                input_mappings={"seg_confidence": list(range(100))},
                output_mappings={"landing_score": list(range(100))}
            )
        
        # Learnable rule weights
        self.rule_weights = torch.nn.Parameter(torch.tensor([0.9, 0.8, 0.7, 0.6]))
        
    def forward(self, seg_confidences: torch.Tensor, 
                spatial_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through Scallop reasoning"""
        
        # Execute Scallop reasoning with current rule weights
        if SCALLOP_AVAILABLE:
            result = self.scallop_module(
                seg_confidence=seg_confidences,
                spatial_feature=spatial_features
            )
        else:
            result = self.scallop_module(
                seg_confidence=seg_confidences,
                spatial_feature=spatial_features
            )
        
        return result
        
    def _get_program_string(self) -> str:
        """Get Scallop program as string for the module"""
        return """
        type seg_confidence(i32)
        type spatial_feature(i32) 
        type landing_score(i32)
        
        rel landing_score(s) = seg_confidence(s1) and spatial_feature(s2) and s = (s1 * s2) / 10
        """
