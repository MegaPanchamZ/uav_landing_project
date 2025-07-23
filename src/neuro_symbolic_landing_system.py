#!/usr/bin/env python3
"""
Neuro-Symbolic UAV Landing System with Scallop Integration
========================================================

Combines:
1. Neural: 24-class semantic segmentation (working WITH dataset structure)
2. Symbolic: Scallop-based logical reasoning for landing decisions
3. Integration: Probabilistic facts → Logical rules → Landing decisions

Architecture:
Neural Network → Semantic Classes → Probabilistic Facts → Scallop Rules → Landing Decision
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from dataclasses import dataclass
from enum import Enum

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.mmseg_bisenetv2 import MMSegBiSeNetV2
from src.landing_safety_interpreter import LandingSafetyInterpreter, LandingSafety

# Scallop integration
try:
    import scallopy as scallop
except ImportError:
    print("Warning: Scallopy not installed. Install with: pip install scallopy-lang")
    scallop = None

@dataclass
class LandingZone:
    """Represents a potential landing zone with its properties."""
    center_x: int
    center_y: int
    radius: int
    safety_score: float
    area: int
    semantic_composition: Dict[str, float]
    obstacles: List[str]
    confidence: float

class LandingDecision(Enum):
    """Final landing decisions."""
    LAND_IMMEDIATELY = "LAND_IMMEDIATELY"
    LAND_WITH_CAUTION = "LAND_WITH_CAUTION" 
    HOVER_AND_ASSESS = "HOVER_AND_ASSESS"
    FIND_ALTERNATIVE = "FIND_ALTERNATIVE"
    EMERGENCY_PROTOCOL = "EMERGENCY_PROTOCOL"

class NeuroSymbolicLandingSystem:
    """
    Neuro-Symbolic UAV Landing System using Scallop.
    
    Workflow:
    1. Neural network predicts 24 semantic classes
    2. Extract probabilistic facts about the scene
    3. Apply Scallop logical rules for reasoning
    4. Generate landing decision with explanations
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda'):
        """Initialize the neuro-symbolic system."""
        self.device = device
        
        # Initialize neural components
        self.semantic_model = MMSegBiSeNetV2(num_classes=24).to(device)
        if model_path:
            self.load_model(model_path)
            
        self.safety_interpreter = LandingSafetyInterpreter()
        
        # Initialize Scallop context
        if scallop is not None:
            self.scallop_ctx = self._initialize_scallop_rules()
        else:
            self.scallop_ctx = None
            print("Warning: Scallop not available, using fallback logic")
    
    def _initialize_scallop_rules(self) -> scallop.Context:
        """Initialize Scallop context with UAV landing rules."""
        ctx = scallop.Context()
        
        # Define the Scallop program with UAV landing logic
        scallop_program = """
        // ===== TYPE DEFINITIONS =====
        type Semantic = String
        type Safety = String  
        type Position = (i32, i32)
        type Area = i32
        type Confidence = f32
        type Decision = String
        
        // ===== INPUT RELATIONS =====
        // Semantic facts from neural network
        rel semantic_region(Position, Semantic, Area, Confidence)
        rel safety_level(Position, Safety, Confidence)
        rel zone_size(Position, Area)
        rel obstacle_nearby(Position, Semantic)
        rel weather_condition(String, f32)  // weather_type, severity
        rel battery_level(f32)
        rel emergency_status(String)
        
        // ===== SEMANTIC UNDERSTANDING RULES =====
        // Define what constitutes good landing surfaces
        rel good_surface(pos, conf) = semantic_region(pos, "paved-area", area, conf), area > 100
        rel good_surface(pos, conf) = semantic_region(pos, "grass", area, conf), area > 150
        rel good_surface(pos, conf) = semantic_region(pos, "dirt", area, conf), area > 120
        rel good_surface(pos, conf) = semantic_region(pos, "gravel", area, conf), area > 100
        
        // Define dangerous elements
        rel dangerous_element(pos, element, conf) = semantic_region(pos, element, _, conf),
            (element == "water" || element == "rocks" || element == "tree" || 
             element == "car" || element == "person" || element == "obstacle")
        
        // Define caution elements  
        rel caution_element(pos, element, conf) = semantic_region(pos, element, _, conf),
            (element == "vegetation" || element == "pool" || element == "roof")
            
        // ===== SPATIAL REASONING RULES =====
        // Large safe zones are preferred
        rel large_safe_zone(pos, area) = good_surface(pos, _), zone_size(pos, area), area > 200
        rel medium_safe_zone(pos, area) = good_surface(pos, _), zone_size(pos, area), 
            area > 100, area <= 200
        rel small_safe_zone(pos, area) = good_surface(pos, _), zone_size(pos, area), 
            area > 50, area <= 100
            
        // Clear zones (no obstacles nearby)
        rel clear_zone(pos) = good_surface(pos, _), 
            !obstacle_nearby(pos, _)
            
        // Proximity to dangers
        rel near_danger(pos) = good_surface(pos, _), obstacle_nearby(pos, danger),
            dangerous_element(_, danger, _)
            
        // ===== SITUATIONAL AWARENESS RULES =====
        // Emergency conditions
        rel emergency_landing_needed() = battery_level(level), level < 0.15
        rel emergency_landing_needed() = emergency_status("critical")
        rel emergency_landing_needed() = weather_condition("severe", severity), severity > 0.8
        
        // Optimal conditions  
        rel optimal_conditions() = battery_level(level), level > 0.5,
            weather_condition(weather, severity), severity < 0.3
            
        // ===== LANDING DECISION LOGIC =====
        // Immediate landing decisions
        rel decision(pos, "LAND_IMMEDIATELY", conf) = 
            large_safe_zone(pos, area), clear_zone(pos), optimal_conditions(),
            good_surface(pos, conf), conf > 0.8
            
        rel decision(pos, "LAND_IMMEDIATELY", conf) = 
            emergency_landing_needed(), good_surface(pos, conf), 
            !near_danger(pos), conf > 0.5
            
        // Cautious landing decisions
        rel decision(pos, "LAND_WITH_CAUTION", conf) = 
            medium_safe_zone(pos, _), good_surface(pos, conf),
            !near_danger(pos), conf > 0.6
            
        rel decision(pos, "LAND_WITH_CAUTION", conf) = 
            small_safe_zone(pos, _), clear_zone(pos), 
            optimal_conditions(), good_surface(pos, conf), conf > 0.7
            
        // Hover and assess decisions
        rel decision(pos, "HOVER_AND_ASSESS", conf) = 
            good_surface(pos, conf), near_danger(pos), conf > 0.5
            
        rel decision(pos, "HOVER_AND_ASSESS", conf) = 
            caution_element(pos, _, conf), !dangerous_element(pos, _, _), conf > 0.6
            
        // Find alternative decisions
        rel decision(pos, "FIND_ALTERNATIVE", conf) = 
            dangerous_element(pos, _, conf), conf > 0.7
            
        rel decision(pos, "FIND_ALTERNATIVE", conf) = 
            small_safe_zone(pos, area), near_danger(pos), area < 80
            
        // Emergency protocol
        rel decision(pos, "EMERGENCY_PROTOCOL", conf) = 
            emergency_landing_needed(), dangerous_element(pos, _, conf)
            
        // ===== DECISION RANKING =====
        // Rank decisions by safety score
        rel safety_score(pos, decision, score) = 
            decision(pos, decision, conf),
            zone_size(pos, area),
            score := conf * (area as f32) / 300.0
            
        // Best decision for each position
        rel best_decision_score(pos, max_score) = 
            safety_score(pos, _, score),
            max_score := max(score)
            
        rel best_decision(pos, decision) = 
            safety_score(pos, decision, score),
            best_decision_score(pos, score)
            
        // ===== OUTPUT RELATIONS =====
        rel final_landing_decision(Position, Decision, f32)
        rel landing_explanation(Position, String)
        
        // Generate final decisions
        rel final_landing_decision(pos, decision, score) = best_decision(pos, decision),
            safety_score(pos, decision, score)
            
        // Generate explanations
        rel landing_explanation(pos, explanation) = 
            decision(pos, "LAND_IMMEDIATELY", _),
            large_safe_zone(pos, area),
            explanation := $string_concat("Large safe landing zone (", $string(area), " pixels) with optimal conditions")
            
        rel landing_explanation(pos, explanation) = 
            decision(pos, "LAND_WITH_CAUTION", _),
            near_danger(pos),
            explanation := "Safe landing area but obstacles detected nearby - proceed with caution"
            
        rel landing_explanation(pos, explanation) = 
            decision(pos, "HOVER_AND_ASSESS", _),
            explanation := "Uncertain conditions detected - hover and gather more information"
            
        rel landing_explanation(pos, explanation) = 
            decision(pos, "FIND_ALTERNATIVE", _),
            explanation := "Current area unsuitable for landing - search for alternative site"
            
        rel landing_explanation(pos, explanation) = 
            decision(pos, "EMERGENCY_PROTOCOL", _),
            emergency_landing_needed(),
            explanation := "Emergency conditions - execute emergency landing protocol"
        """
        
        # Add the program to context
        ctx.add_program(scallop_program)
        
        return ctx
    
    def load_model(self, model_path: str):
        """Load pretrained semantic segmentation model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.semantic_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.semantic_model.load_state_dict(checkpoint)
        self.semantic_model.eval()
    
    def process_image(self, image: np.ndarray, 
                     battery_level: float = 0.8,
                     weather_condition: Tuple[str, float] = ("clear", 0.1),
                     emergency_status: str = "normal") -> Dict:
        """
        Process an aerial image and generate landing decisions.
        
        Args:
            image: Input aerial image [H, W, 3]
            battery_level: Current battery level (0.0 to 1.0)
            weather_condition: (weather_type, severity_0_to_1)
            emergency_status: "normal", "warning", "critical"
            
        Returns:
            Landing analysis with decisions and explanations
        """
        # Step 1: Neural semantic segmentation
        semantic_predictions = self._predict_semantics(image)
        
        # Step 2: Extract probabilistic facts
        facts = self._extract_probabilistic_facts(semantic_predictions)
        
        # Step 3: Add contextual information
        facts.update({
            'battery_level': battery_level,
            'weather_condition': weather_condition,
            'emergency_status': emergency_status
        })
        
        # Step 4: Apply Scallop reasoning
        if self.scallop_ctx is not None:
            decisions = self._apply_scallop_reasoning(facts)
        else:
            decisions = self._fallback_reasoning(facts)
        
        # Step 5: Generate comprehensive analysis
        analysis = self._generate_analysis(image, semantic_predictions, facts, decisions)
        
        return analysis
    
    def _predict_semantics(self, image: np.ndarray) -> torch.Tensor:
        """Predict semantic classes using neural network."""
        # Preprocess image
        if len(image.shape) == 3:
            image = cv2.resize(image, (512, 512))
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
        else:
            image_tensor = image.to(self.device)
        
        with torch.no_grad():
            outputs = self.semantic_model(image_tensor)
            predictions = torch.argmax(outputs, dim=1).squeeze()
            
        return predictions
    
    def _extract_probabilistic_facts(self, semantic_predictions: torch.Tensor) -> Dict:
        """Extract probabilistic facts from semantic predictions."""
        h, w = semantic_predictions.shape
        facts = {
            'semantic_regions': [],
            'safety_levels': [],
            'zone_sizes': [],
            'obstacles': []
        }
        
        # Analyze connected components for each semantic class
        for class_idx in range(24):
            class_mask = (semantic_predictions == class_idx).cpu().numpy().astype(np.uint8)
            
            if class_mask.sum() > 10:  # Only consider meaningful regions
                # Find connected components
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_mask)
                
                for i in range(1, num_labels):  # Skip background (0)
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area > 20:  # Minimum area threshold
                        center_x = int(centroids[i, 0])
                        center_y = int(centroids[i, 1])
                        
                        # Calculate confidence based on area and compactness
                        confidence = min(0.9, 0.3 + (area / 1000.0))
                        
                        semantic_class = self._get_semantic_class_name(class_idx)
                        
                        facts['semantic_regions'].append({
                            'position': (center_x, center_y),
                            'class': semantic_class,
                            'area': area,
                            'confidence': confidence
                        })
                        
                        facts['zone_sizes'].append({
                            'position': (center_x, center_y),
                            'area': area
                        })
                        
                        # Check if it's an obstacle
                        if semantic_class in ['water', 'rocks', 'tree', 'car', 'person', 'obstacle']:
                            facts['obstacles'].append({
                                'position': (center_x, center_y),
                                'type': semantic_class
                            })
        
        # Convert to safety levels using interpreter
        safety_map = self.safety_interpreter.interpret_semantic_predictions(semantic_predictions)
        
        # Extract safety facts
        for safety_level in [0, 1, 2, 3]:  # SAFE, CAUTION, DANGEROUS, IGNORE
            safety_mask = (safety_map == safety_level).cpu().numpy().astype(np.uint8)
            
            if safety_mask.sum() > 10:
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(safety_mask)
                
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area > 20:
                        center_x = int(centroids[i, 0])
                        center_y = int(centroids[i, 1])
                        confidence = min(0.9, 0.4 + (area / 800.0))
                        
                        safety_name = ['SAFE', 'CAUTION', 'DANGEROUS', 'IGNORE'][safety_level]
                        
                        facts['safety_levels'].append({
                            'position': (center_x, center_y),
                            'safety': safety_name,
                            'confidence': confidence
                        })
        
        return facts
    
    def _apply_scallop_reasoning(self, facts: Dict) -> List[Dict]:
        """Apply Scallop logical reasoning to generate decisions."""
        # Clear previous facts
        self.scallop_ctx.clear_relation("semantic_region")
        self.scallop_ctx.clear_relation("safety_level")
        self.scallop_ctx.clear_relation("zone_size")
        self.scallop_ctx.clear_relation("obstacle_nearby")
        self.scallop_ctx.clear_relation("weather_condition")
        self.scallop_ctx.clear_relation("battery_level")
        self.scallop_ctx.clear_relation("emergency_status")
        
        # Add semantic region facts
        for region in facts['semantic_regions']:
            self.scallop_ctx.add_fact("semantic_region", (
                region['position'],
                region['class'],
                region['area'],
                region['confidence']
            ))
        
        # Add safety level facts
        for safety in facts['safety_levels']:
            self.scallop_ctx.add_fact("safety_level", (
                safety['position'],
                safety['safety'],
                safety['confidence']
            ))
        
        # Add zone size facts
        for zone in facts['zone_sizes']:
            self.scallop_ctx.add_fact("zone_size", (
                zone['position'],
                zone['area']
            ))
        
        # Add obstacle facts
        for obstacle in facts['obstacles']:
            # Find nearby positions (within radius)
            for region in facts['semantic_regions']:
                pos1 = obstacle['position']
                pos2 = region['position']
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                if distance < 50:  # Within 50 pixels
                    self.scallop_ctx.add_fact("obstacle_nearby", (
                        region['position'],
                        obstacle['type']
                    ))
        
        # Add contextual facts
        self.scallop_ctx.add_fact("battery_level", facts['battery_level'])
        self.scallop_ctx.add_fact("weather_condition", facts['weather_condition'])
        self.scallop_ctx.add_fact("emergency_status", facts['emergency_status'])
        
        # Run reasoning
        self.scallop_ctx.run()
        
        # Extract results
        decisions = []
        
        # Get final landing decisions
        for (pos, decision, score) in self.scallop_ctx.relation("final_landing_decision"):
            explanation = "No explanation available"
            
            # Get explanation if available
            for (exp_pos, exp_text) in self.scallop_ctx.relation("landing_explanation"):
                if exp_pos == pos:
                    explanation = exp_text
                    break
            
            decisions.append({
                'position': pos,
                'decision': decision,
                'score': score,
                'explanation': explanation
            })
        
        # Sort by score (highest first)
        decisions.sort(key=lambda x: x['score'], reverse=True)
        
        return decisions
    
    def _fallback_reasoning(self, facts: Dict) -> List[Dict]:
        """Fallback reasoning when Scallop is not available."""
        decisions = []
        
        # Simple rule-based fallback
        for region in facts['semantic_regions']:
            if region['class'] in ['paved-area', 'grass', 'dirt', 'gravel']:
                if region['area'] > 200 and region['confidence'] > 0.7:
                    decision = "LAND_IMMEDIATELY" 
                    score = region['confidence'] * (region['area'] / 300.0)
                    explanation = f"Large safe area ({region['area']} pixels) detected"
                elif region['area'] > 100:
                    decision = "LAND_WITH_CAUTION"
                    score = region['confidence'] * 0.7
                    explanation = f"Medium safe area ({region['area']} pixels) - proceed with caution"
                else:
                    decision = "HOVER_AND_ASSESS"
                    score = region['confidence'] * 0.5
                    explanation = f"Small area ({region['area']} pixels) - need more assessment"
                
                decisions.append({
                    'position': region['position'],
                    'decision': decision,
                    'score': score,
                    'explanation': explanation
                })
        
        decisions.sort(key=lambda x: x['score'], reverse=True)
        return decisions
    
    def _generate_analysis(self, image: np.ndarray, semantic_predictions: torch.Tensor,
                          facts: Dict, decisions: List[Dict]) -> Dict:
        """Generate comprehensive analysis report."""
        # Generate visualizations
        safety_map = self.safety_interpreter.interpret_semantic_predictions(semantic_predictions)
        safety_viz = self.safety_interpreter.visualize_safety_map(safety_map)
        
        # Analyze overall safety
        safety_analysis = self.safety_interpreter.analyze_landing_zone(safety_map)
        
        return {
            'semantic_predictions': semantic_predictions,
            'safety_map': safety_map,
            'safety_visualization': safety_viz,
            'safety_analysis': safety_analysis,
            'extracted_facts': facts,
            'landing_decisions': decisions,
            'best_decision': decisions[0] if decisions else None,
            'context': {
                'battery_level': facts.get('battery_level', 0.8),
                'weather_condition': facts.get('weather_condition', ("clear", 0.1)),
                'emergency_status': facts.get('emergency_status', "normal")
            }
        }
    
    def _get_semantic_class_name(self, class_idx: int) -> str:
        """Get semantic class name from index."""
        class_names = [
            'unlabeled', 'paved-area', 'dirt', 'grass', 'gravel', 'water',
            'rocks', 'pool', 'vegetation', 'roof', 'wall', 'window',
            'door', 'fence', 'fence-pole', 'person', 'dog', 'car',
            'bicycle', 'tree', 'bald-tree', 'ar-marker', 'obstacle', 'conflicting'
        ]
        return class_names[class_idx] if 0 <= class_idx < len(class_names) else 'unknown'

# Example usage and testing
if __name__ == "__main__":
    # Initialize system
    system = NeuroSymbolicLandingSystem()
    
    # Test with sample image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Process different scenarios
    scenarios = [
        {"battery": 0.9, "weather": ("clear", 0.1), "emergency": "normal"},
        {"battery": 0.2, "weather": ("cloudy", 0.4), "emergency": "warning"},
        {"battery": 0.05, "weather": ("storm", 0.8), "emergency": "critical"}
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"\n=== Scenario {i+1}: {scenario['emergency'].upper()} ===")
        
        result = system.process_image(
            test_image,
            battery_level=scenario['battery'],
            weather_condition=scenario['weather'],
            emergency_status=scenario['emergency']
        )
        
        if result['best_decision']:
            decision = result['best_decision']
            print(f"Decision: {decision['decision']}")
            print(f"Score: {decision['score']:.3f}")
            print(f"Position: {decision['position']}")
            print(f"Explanation: {decision['explanation']}")
        else:
            print("No suitable landing zones found")
        
        print(f"Overall Safety: {result['safety_analysis']['overall_safety']}")
        print(f"Recommendation: {result['safety_analysis']['recommendation']}") 