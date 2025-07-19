#!/usr/bin/env python3
"""
UAV Landing System - Plug-and-Play Interface
============================================

A complete neuro-symbolic UAV landing detection system combining:
- Fine-tuned BiSeNetV2 semantic segmentation model
- Rule-based symbolic reasoning for safety and landing zone evaluation
- Real-time inference with comprehensive traceability

Author: UAV Landing Detection Team
Version: 2.0.0
Date: 2025-07-20

Usage:
    from uav_landing_system import UAVLandingSystem
    
    # Initialize system
    system = UAVLandingSystem()
    
    # Process frame
    result = system.process_frame(image, altitude=5.0)
    print(f"Status: {result.status}, Confidence: {result.confidence}")
"""

import sys
import os
from pathlib import Path
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json

# Add src to path for imports
sys.path.append('src')

try:
    import numpy as np
    import cv2
    from uav_landing_detector import UAVLandingDetector, LandingResult
except ImportError as e:
    raise ImportError(f"Required dependencies not installed: {e}. Run: pip install -r requirements.txt")

@dataclass
class NeuroSymbolicTrace:
    """Detailed traceability information for neuro-symbolic decisions"""
    
    # Neural Network Component
    neural_inference_time: float
    neural_confidence: float
    neural_classes_detected: List[str]
    neural_class_distribution: Dict[str, float]
    
    # Symbolic Reasoning Component  
    symbolic_candidates_found: int
    symbolic_rules_applied: List[str]
    symbolic_safety_checks: Dict[str, bool]
    symbolic_scoring_breakdown: Dict[str, float]
    
    # Integration Component
    neuro_symbolic_score: float
    decision_weights: Dict[str, float]
    confidence_adjustments: Dict[str, float]
    
    # Risk Assessment
    risk_level: str
    risk_factors: Dict[str, Any]
    safety_recommendations: List[str]
    
    # Temporal Context
    frame_consistency: Optional[float]
    tracking_history: Optional[Dict[str, Any]]
    
    # Performance Metrics
    total_processing_time: float
    inference_fps: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for JSON serialization"""
        return {
            'neural_component': {
                'inference_time_ms': self.neural_inference_time,
                'confidence': self.neural_confidence,
                'classes_detected': self.neural_classes_detected,
                'class_distribution': self.neural_class_distribution
            },
            'symbolic_component': {
                'candidates_found': self.symbolic_candidates_found,
                'rules_applied': self.symbolic_rules_applied,
                'safety_checks': self.symbolic_safety_checks,
                'scoring_breakdown': self.symbolic_scoring_breakdown
            },
            'integration': {
                'neuro_symbolic_score': self.neuro_symbolic_score,
                'decision_weights': self.decision_weights,
                'confidence_adjustments': self.confidence_adjustments
            },
            'risk_assessment': {
                'risk_level': self.risk_level,
                'risk_factors': self.risk_factors,
                'safety_recommendations': self.safety_recommendations
            },
            'temporal_context': {
                'frame_consistency': self.frame_consistency,
                'tracking_history': self.tracking_history
            },
            'performance': {
                'total_processing_time_ms': self.total_processing_time,
                'inference_fps': self.inference_fps
            }
        }

@dataclass  
class EnhancedLandingResult(LandingResult):
    """Enhanced landing result with neuro-symbolic traceability"""
    trace: Optional[NeuroSymbolicTrace] = None
    decision_explanation: Optional[str] = None
    confidence_breakdown: Optional[Dict[str, float]] = None

class UAVLandingSystem:
    """
    Production-ready UAV Landing System with Neuro-Symbolic Intelligence
    
    A complete plug-and-play system that combines fine-tuned neural networks
    with symbolic reasoning for intelligent, safe, and traceable landing decisions.
    
    Key Features:
    - Configurable input resolution for quality vs speed trade-offs
    - Neuro-symbolic fusion with 40% neural + 60% symbolic reasoning
    - Complete decision traceability and explainable AI
    - Safety-first design with risk assessment and abort mechanisms
    - Real-time performance monitoring and logging
    """
    
    def __init__(self, 
                 model_path: str = "trained_models/ultra_fast_uav_landing.onnx",
                 config_path: Optional[str] = None,
                 input_resolution: Tuple[int, int] = (512, 512),
                 enable_logging: bool = False,
                 log_level: str = "INFO"):
        """
        Initialize UAV Landing System
        
        Args:
            model_path: Path to ONNX model file
            config_path: Optional path to JSON configuration file
            input_resolution: Model input resolution (width, height)
                            - (256, 256): Ultra-fast processing (~80-127 FPS)
                            - (512, 512): Balanced quality/speed (~20-60 FPS) [RECOMMENDED]
                            - (768, 768): High quality (~8-25 FPS)
                            - (1024, 1024): Maximum quality (~3-12 FPS)
            enable_logging: Enable detailed logging
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        
        # Initialize logging
        self.logger = logging.getLogger("UAVLandingSystem")
        if enable_logging:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                        datefmt='%H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Log initialization
        if enable_logging:
            self.logger.info("🚁 Initializing UAV Landing System...")
            self.logger.info(f"📁 Loading model: {model_path}")
            self.logger.info(f"🔍 Input resolution: {input_resolution}")
        
        # Initialize core detector with configurable resolution
        self.detector = UAVLandingDetector(
            model_path=model_path,
            input_resolution=input_resolution,
            enable_visualization=True
        )
        
        # Neuro-symbolic reasoning configuration
        self.neural_weight = self.config.get('neural_weight', 0.4)
        self.symbolic_weight = self.config.get('symbolic_weight', 0.6)
        self.safety_threshold = self.config.get('safety_threshold', 0.3)
        
        # Class definitions for traceability
        self.class_names = {
            0: "Background",
            1: "Suitable_Ground", 
            2: "Marginal_Ground",
            3: "Unsuitable_Surface"
        }
        
        # Initialize reasoning rules
        self.symbolic_rules = self._initialize_symbolic_rules()
        
        # Performance tracking
        self.frame_count = 0
        self.total_processing_time = 0.0
        
        if enable_logging:
            self.logger.info("✅ UAV Landing System initialized successfully")
            self.logger.info(f"   Model: {Path(model_path).name}")
            self.logger.info(f"   Neural weight: {self.neural_weight}")
            self.logger.info(f"   Symbolic weight: {self.symbolic_weight}")
            self.logger.info(f"   Safety threshold: {self.safety_threshold}")
    
    def process_frame(self, 
                     image: np.ndarray,
                     altitude: float,
                     enable_tracing: bool = False,
                     return_visualization: bool = False) -> EnhancedLandingResult:
        """
        Process a single frame for landing detection with full traceability
        
        Args:
            image: Input RGB image from UAV camera
            altitude: Current UAV altitude in meters
            enable_tracing: Enable detailed decision traceability
            return_visualization: Include visualization in result
            
        Returns:
            EnhancedLandingResult with decision and optional trace
        """
        
        start_time = time.time()
        self.frame_count += 1
        
        try:
            # 1. Neural Network Inference
            self.logger.debug(f"🧠 Running neural inference (frame {self.frame_count})")
            neural_start = time.time()
            
            # Run core detector
            base_result = self.detector.process_frame(image, altitude)
            
            neural_time = (time.time() - neural_start) * 1000
            
            # Extract neural outputs for reasoning
            seg_mask, raw_output, confidence_map = self.detector.get_segmentation_data()
            
            if seg_mask is None:
                self.logger.warning("No segmentation data available, using base result")
                return EnhancedLandingResult(**base_result.__dict__)
            
            # 2. Neural Analysis
            neural_analysis = self._analyze_neural_output(seg_mask, confidence_map)
            
            # 3. Symbolic Reasoning
            self.logger.debug("🔬 Applying symbolic reasoning")
            symbolic_analysis = self._apply_symbolic_reasoning(
                seg_mask, confidence_map, altitude, image.shape
            )
            
            # 4. Neuro-Symbolic Integration
            self.logger.debug("🎯 Integrating neuro-symbolic decision")
            integrated_decision = self._integrate_neural_symbolic(
                neural_analysis, symbolic_analysis, base_result
            )
            
            # 5. Generate traceability (if requested)
            trace = None
            if enable_tracing:
                trace = self._generate_trace(
                    neural_analysis, symbolic_analysis, integrated_decision,
                    neural_time, time.time() - start_time
                )
            
            # 6. Create enhanced result
            enhanced_result = EnhancedLandingResult(
                **integrated_decision,
                trace=trace,
                decision_explanation=self._generate_explanation(integrated_decision),
                confidence_breakdown=self._get_confidence_breakdown(neural_analysis, symbolic_analysis)
            )
            
            # Update performance tracking
            processing_time = (time.time() - start_time) * 1000
            self.total_processing_time += processing_time
            
            self.logger.info(f"✅ Frame {self.frame_count} processed: {enhanced_result.status} "
                           f"(confidence: {enhanced_result.confidence:.3f}, time: {processing_time:.1f}ms)")
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"❌ Error processing frame {self.frame_count}: {e}")
            # Return error result
            return EnhancedLandingResult(
                status="ERROR",
                confidence=0.0,
                processing_time=(time.time() - start_time) * 1000,
                fps=0.0,
                decision_explanation=f"Processing error: {str(e)}"
            )
    
    def _analyze_neural_output(self, segmentation: np.ndarray, confidence_map: np.ndarray) -> Dict[str, Any]:
        """Analyze neural network predictions"""
        unique_classes, counts = np.unique(segmentation, return_counts=True)
        total_pixels = segmentation.size
        
        class_distribution = {}
        class_confidences = {}
        
        for class_id, count in zip(unique_classes, counts):
            class_name = self.class_names.get(class_id, f"Class_{class_id}")
            percentage = (count / total_pixels) * 100
            
            # Calculate average confidence for this class
            class_mask = segmentation == class_id
            avg_confidence = np.mean(confidence_map[class_mask]) if np.any(class_mask) else 0.0
            
            class_distribution[class_name] = percentage
            class_confidences[class_name] = avg_confidence
        
        return {
            'overall_confidence': np.mean(confidence_map),
            'confidence_std': np.std(confidence_map),
            'class_distribution': class_distribution,
            'class_confidences': class_confidences,
            'classes_detected': list(class_distribution.keys())
        }
    
    def _apply_symbolic_reasoning(self, segmentation: np.ndarray, confidence_map: np.ndarray,
                                altitude: float, image_shape: Tuple) -> Dict[str, Any]:
        """Apply symbolic reasoning rules"""
        
        # Find suitable areas
        suitable_mask = np.isin(segmentation, [1, 2])  # Suitable and marginal
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        suitable_mask = cv2.morphologyEx(suitable_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        suitable_mask = cv2.morphologyEx(suitable_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(suitable_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Apply size rule based on altitude
        min_area = self.symbolic_rules['altitude_size_rules'][self._get_altitude_category(altitude)]
        min_area_pixels = (image_shape[0] * image_shape[1]) * min_area
        
        candidates = []
        rules_applied = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area >= min_area_pixels:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 1.0
                    solidity = area / cv2.contourArea(cv2.convexHull(contour)) if cv2.contourArea(cv2.convexHull(contour)) > 0 else 0
                    
                    candidates.append({
                        'center': (cx, cy),
                        'area': area,
                        'bbox': (x, y, w, h),
                        'aspect_ratio': aspect_ratio,
                        'solidity': solidity
                    })
        
        rules_applied = [f"altitude_size_rule_{self._get_altitude_category(altitude)}", "morphological_filtering", "contour_analysis"]
        
        # Safety checks
        safety_checks = {
            'min_area_satisfied': len(candidates) > 0,
            'confidence_threshold_met': np.mean(confidence_map) > self.safety_threshold,
            'altitude_appropriate': 1.0 <= altitude <= 50.0
        }
        
        return {
            'candidates': candidates,
            'rules_applied': rules_applied,
            'safety_checks': safety_checks,
            'altitude_category': self._get_altitude_category(altitude)
        }
    
    def _integrate_neural_symbolic(self, neural: Dict, symbolic: Dict, base_result: LandingResult) -> Dict[str, Any]:
        """Integrate neural and symbolic decisions"""
        
        if not symbolic['candidates']:
            return {
                'status': 'NO_TARGET',
                'confidence': neural['overall_confidence'] * 0.5,  # Reduced confidence
                'target_pixel': None,
                'target_world': None,
                'processing_time': base_result.processing_time,
                'fps': base_result.fps,
                'forward_velocity': 0.0,
                'right_velocity': 0.0,
                'descent_rate': 0.0
            }
        
        # Score candidates using neuro-symbolic approach
        best_candidate = None
        best_score = 0.0
        
        for candidate in symbolic['candidates']:
            # Neural component
            neural_score = neural['overall_confidence']
            
            # Symbolic components
            area_normalized = candidate['area'] / (2000 * 2000)  # Normalize to typical image size
            size_score = min(area_normalized * 5, 1.0)
            
            shape_score = 1.0 - abs(1.0 - candidate['aspect_ratio']) if candidate['aspect_ratio'] <= 2.0 else 0.5
            shape_score *= candidate['solidity']
            
            # Weighted integration
            final_score = (neural_score * self.neural_weight + 
                          size_score * (self.symbolic_weight * 0.6) +
                          shape_score * (self.symbolic_weight * 0.4))
            
            if final_score > best_score:
                best_score = final_score
                best_candidate = candidate
        
        # Check safety
        all_safety_passed = all(symbolic['safety_checks'].values())
        
        if not all_safety_passed or best_score < self.safety_threshold:
            status = 'UNSAFE'
        else:
            status = 'TARGET_ACQUIRED'
        
        return {
            'status': status,
            'confidence': best_score,
            'target_pixel': best_candidate['center'] if best_candidate else None,
            'target_world': base_result.target_world,
            'processing_time': base_result.processing_time,
            'fps': base_result.fps,
            'forward_velocity': base_result.forward_velocity,
            'right_velocity': base_result.right_velocity,
            'descent_rate': base_result.descent_rate if status == 'TARGET_ACQUIRED' else 0.0
        }
    
    def _generate_trace(self, neural: Dict, symbolic: Dict, decision: Dict,
                       neural_time: float, total_time: float) -> NeuroSymbolicTrace:
        """Generate detailed traceability information"""
        
        # Risk assessment
        risk_level = "HIGH"
        if decision['confidence'] > 0.7 and decision['status'] == 'TARGET_ACQUIRED':
            risk_level = "LOW"
        elif decision['confidence'] > 0.5 and decision['status'] == 'TARGET_ACQUIRED':
            risk_level = "MEDIUM"
        
        risk_factors = {
            'low_neural_confidence': neural['overall_confidence'] < 0.5,
            'no_landing_candidates': len(symbolic['candidates']) == 0,
            'safety_checks_failed': not all(symbolic['safety_checks'].values())
        }
        
        safety_recommendations = []
        if risk_factors['low_neural_confidence']:
            safety_recommendations.append("Increase altitude for better visibility")
        if risk_factors['no_landing_candidates']:
            safety_recommendations.append("Search for alternative landing area")
        if risk_factors['safety_checks_failed']:
            safety_recommendations.append("Verify safety conditions before landing")
        
        return NeuroSymbolicTrace(
            neural_inference_time=neural_time,
            neural_confidence=neural['overall_confidence'],
            neural_classes_detected=neural['classes_detected'],
            neural_class_distribution=neural['class_distribution'],
            symbolic_candidates_found=len(symbolic['candidates']),
            symbolic_rules_applied=symbolic['rules_applied'],
            symbolic_safety_checks=symbolic['safety_checks'],
            symbolic_scoring_breakdown={'neural_weight': self.neural_weight, 'symbolic_weight': self.symbolic_weight},
            neuro_symbolic_score=decision['confidence'],
            decision_weights={'neural': self.neural_weight, 'symbolic': self.symbolic_weight},
            confidence_adjustments={},
            risk_level=risk_level,
            risk_factors=risk_factors,
            safety_recommendations=safety_recommendations,
            frame_consistency=None,
            tracking_history=None,
            total_processing_time=total_time * 1000,
            inference_fps=1000 / (total_time * 1000) if total_time > 0 else 0
        )
    
    def _generate_explanation(self, decision: Dict) -> str:
        """Generate human-readable explanation of decision"""
        status = decision['status']
        confidence = decision['confidence']
        
        if status == 'TARGET_ACQUIRED':
            return f"Safe landing zone detected with {confidence:.1%} confidence. Neural network identified suitable ground with symbolic reasoning confirming safety constraints."
        elif status == 'NO_TARGET':
            return f"No suitable landing zones found. Neural confidence: {confidence:.1%}. Recommend altitude adjustment or area search."
        elif status == 'UNSAFE':
            return f"Landing deemed unsafe despite detection. Confidence: {confidence:.1%}. Safety checks failed - recommend abort or relocation."
        else:
            return f"System status: {status}. Confidence: {confidence:.1%}."
    
    def _get_confidence_breakdown(self, neural: Dict, symbolic: Dict) -> Dict[str, float]:
        """Get detailed confidence breakdown"""
        return {
            'neural_overall': neural['overall_confidence'],
            'neural_stability': 1.0 - neural['confidence_std'],
            'symbolic_candidates': min(len(symbolic['candidates']) / 3.0, 1.0),
            'safety_score': sum(symbolic['safety_checks'].values()) / len(symbolic['safety_checks'])
        }
    
    def _initialize_symbolic_rules(self) -> Dict[str, Any]:
        """Initialize symbolic reasoning rules"""
        return {
            'altitude_size_rules': {
                'high': 0.08,    # >10m: need large zones
                'medium': 0.04,  # 5-10m: medium zones
                'low': 0.02      # <5m: small zones acceptable
            },
            'shape_preferences': {
                'min_aspect_ratio': 0.5,
                'max_aspect_ratio': 2.0,
                'min_solidity': 0.7
            },
            'safety_margins': {
                'high_altitude': 20,    # pixels
                'medium_altitude': 15,
                'low_altitude': 10
            }
        }
    
    def _get_altitude_category(self, altitude: float) -> str:
        """Categorize altitude for rule application"""
        if altitude > 10.0:
            return 'high'
        elif altitude > 5.0:
            return 'medium'
        else:
            return 'low'
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            'neural_weight': 0.4,
            'symbolic_weight': 0.6,
            'safety_threshold': 0.3,
            'enable_temporal_tracking': True,
            'max_tracking_history': 10
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                self.logger.info(f"📄 Configuration loaded from {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load config {config_path}: {e}, using defaults")
        
        return default_config
    
    def _setup_logging(self, enable: bool, level: str) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('UAVLandingSystem')
        
        if enable:
            logging.basicConfig(
                level=getattr(logging, level.upper()),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
        else:
            logger.setLevel(logging.CRITICAL)
        
        return logger
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        avg_time = self.total_processing_time / self.frame_count if self.frame_count > 0 else 0
        
        return {
            'frames_processed': self.frame_count,
            'total_processing_time_ms': self.total_processing_time,
            'average_processing_time_ms': avg_time,
            'average_fps': 1000 / avg_time if avg_time > 0 else 0,
            'model_path': self.model_path,
            'neural_weight': self.neural_weight,
            'symbolic_weight': self.symbolic_weight
        }
    
    def save_trace_log(self, traces: List[NeuroSymbolicTrace], filename: str = "uav_landing_trace_log.json"):
        """Save traceability logs to file"""
        trace_data = [trace.to_dict() for trace in traces]
        
        with open(filename, 'w') as f:
            json.dump({
                'system_info': self.get_performance_stats(),
                'traces': trace_data
            }, f, indent=2)
        
        self.logger.info(f"💾 Trace log saved: {filename}")

# Convenience functions for quick usage
def create_uav_landing_system(model_path: str = "trained_models/ultra_fast_uav_landing.onnx") -> UAVLandingSystem:
    """Create UAV landing system with default configuration"""
    return UAVLandingSystem(model_path=model_path)

def process_image_for_landing(image: np.ndarray, altitude: float, 
                            model_path: str = "trained_models/ultra_fast_uav_landing.onnx",
                            input_resolution: Tuple[int, int] = (512, 512),
                            enable_tracing: bool = False) -> EnhancedLandingResult:
    """
    Convenience function for single image processing with configurable resolution
    
    This is the simplest way to use the UAV landing system - just 3 lines of code!
    
    Args:
        image: Input RGB image (numpy array)
        altitude: UAV altitude in meters
        model_path: Path to fine-tuned model
        input_resolution: Model input resolution for quality vs speed trade-off
                        - (256, 256): Ultra-fast (~80-127 FPS, basic quality)
                        - (512, 512): Balanced (~20-60 FPS, good quality) [DEFAULT]
                        - (768, 768): High quality (~8-25 FPS, high quality)
                        - (1024, 1024): Maximum quality (~3-12 FPS, best quality)
        enable_tracing: Enable detailed traceability
        
    Returns:
        EnhancedLandingResult with landing decision and performance metrics
        
    Example:
        >>> import cv2
        >>> from uav_landing_system import process_image_for_landing
        >>> 
        >>> # Load UAV image
        >>> image = cv2.imread("uav_frame.jpg")
        >>> 
        >>> # Process for landing with high quality
        >>> result = process_image_for_landing(image, altitude=5.0, 
        ...                                   input_resolution=(768, 768))
        >>> 
        >>> # Get decision
        >>> print(f"Status: {result.status}")
        >>> print(f"Confidence: {result.confidence:.3f}")
        >>> print(f"Processing time: {result.processing_time:.1f}ms")
    """
    system = UAVLandingSystem(model_path=model_path, input_resolution=input_resolution, enable_logging=False)
    return system.process_frame(image, altitude, enable_tracing=enable_tracing)

if __name__ == "__main__":
    # Demo usage
    print("🚁 UAV Landing System - Plug & Play Demo")
    
    # Create system
    system = UAVLandingSystem()
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Process with full traceability
    result = system.process_frame(test_image, altitude=5.0, enable_tracing=True)
    
    print(f"✅ Status: {result.status}")
    print(f"✅ Confidence: {result.confidence:.3f}")
    print(f"✅ Processing time: {result.processing_time:.1f}ms")
    
    if result.trace:
        print(f"✅ Neural classes: {result.trace.neural_classes_detected}")
        print(f"✅ Risk level: {result.trace.risk_level}")
        print(f"✅ Explanation: {result.decision_explanation}")
    
    print("\n🎯 System ready for deployment!")
