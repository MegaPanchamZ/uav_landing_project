#!/usr/bin/env python3
"""
Neuro-Symbolic UAV Landing System
=================================

Complete integration of:
1. 24-class aerial semantic segmentation neural network
2. Scallop-based logical reasoning for landing decisions
3. Explainable AI for safety-critical UAV applications

This system preserves semantic richness while providing logical,
explainable landing decisions with confidence estimation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional, Any
import json

# Scallop integration (assuming scallopy is available)
try:
    import scallopy
    SCALLOP_AVAILABLE = True
except ImportError:
    print("⚠️  Scallopy not available. Install with: pip install scallopy")
    SCALLOP_AVAILABLE = False

# Import our modules
from datasets.aerial_semantic_24_dataset import AerialSemantic24Dataset
from models.enhanced_architectures import EnhancedBiSeNetV2


class NeuroSymbolicLandingSystem:
    """
    Complete neuro-symbolic system for UAV landing zone detection.
    
    Architecture:
    1. Neural perception: Image → 24-class semantic segmentation
    2. Symbolic reasoning: Semantic classes + logic rules → landing decisions
    3. Explanation generation: Logical trace of decision process
    """
    
    def __init__(
        self,
        neural_model_path: str,
        scallop_rules_path: str,
        class_info: Optional[Dict] = None,
        confidence_threshold: float = 0.1,
        device: str = 'auto'
    ):
        """
        Initialize the neuro-symbolic landing system.
        
        Args:
            neural_model_path: Path to trained 24-class segmentation model
            scallop_rules_path: Path to Scallop rules file
            class_info: Mapping of class IDs to semantic information
            confidence_threshold: Minimum confidence for including predictions
            device: Device for neural network ('auto', 'cuda', 'cpu')
        """
        self.neural_model_path = neural_model_path
        self.scallop_rules_path = scallop_rules_path
        self.confidence_threshold = confidence_threshold
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🛩️  Neuro-Symbolic UAV Landing System")
        print(f"   Neural model: {neural_model_path}")
        print(f"   Scallop rules: {scallop_rules_path}")
        print(f"   Device: {self.device}")
        
        # Load class information
        if class_info is None:
            self.class_info = self._get_default_class_info()
        else:
            self.class_info = class_info
        
        # Initialize components
        self.neural_model = self._load_neural_model()
        self.scallop_engine = self._load_scallop_engine()
        
        print(f" System initialized successfully")
    
    def _get_default_class_info(self) -> Dict[int, Dict[str, Any]]:
        """Get default 24-class semantic information."""
        return {
            0: {"name": "unlabeled", "landing_relevance": "unknown"},
            1: {"name": "paved-area", "landing_relevance": "safe"},
            2: {"name": "dirt", "landing_relevance": "safe"},
            3: {"name": "grass", "landing_relevance": "safe"},
            4: {"name": "gravel", "landing_relevance": "safe"},
            5: {"name": "water", "landing_relevance": "danger"},
            6: {"name": "rocks", "landing_relevance": "caution"},
            7: {"name": "pool", "landing_relevance": "danger"},
            8: {"name": "vegetation", "landing_relevance": "caution"},
            9: {"name": "roof", "landing_relevance": "caution"},
            10: {"name": "wall", "landing_relevance": "obstacle"},
            11: {"name": "window", "landing_relevance": "landmark"},
            12: {"name": "door", "landing_relevance": "landmark"},
            13: {"name": "fence", "landing_relevance": "obstacle"},
            14: {"name": "fence-pole", "landing_relevance": "obstacle"},
            15: {"name": "person", "landing_relevance": "danger"},
            16: {"name": "dog", "landing_relevance": "danger"},
            17: {"name": "car", "landing_relevance": "danger"},
            18: {"name": "bicycle", "landing_relevance": "danger"},
            19: {"name": "tree", "landing_relevance": "danger"},
            20: {"name": "bald-tree", "landing_relevance": "caution"},
            21: {"name": "ar-marker", "landing_relevance": "landmark"},
            22: {"name": "obstacle", "landing_relevance": "danger"},
            23: {"name": "conflicting", "landing_relevance": "unknown"}
        }
    
    def _load_neural_model(self) -> torch.nn.Module:
        """Load the trained 24-class neural segmentation model."""
        
        print(f"📡 Loading neural model...")
        
        # Create model architecture
        model = EnhancedBiSeNetV2(
            num_classes=24,
            input_resolution=(512, 512),
            backbone='resnet50',
            use_attention=True,
            uncertainty_estimation=True,
            dropout_rate=0.1
        )
        
        # Load trained weights
        if self.neural_model_path.endswith('.onnx'):
            # TODO: Implement ONNX loading for production deployment
            raise NotImplementedError("ONNX loading not yet implemented")
        else:
            # Load PyTorch checkpoint
            checkpoint = torch.load(self.neural_model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        print(f" Neural model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        return model
    
    def _load_scallop_engine(self) -> Optional[Any]:
        """Load and initialize the Scallop reasoning engine."""
        
        if not SCALLOP_AVAILABLE:
            print("⚠️  Scallop not available, using rule-based fallback")
            return None
        
        print(f"🧠 Loading Scallop reasoning engine...")
        
        try:
            # Initialize Scallop context
            ctx = scallopy.ScallopContext()
            
            # Load rules from file
            with open(self.scallop_rules_path, 'r') as f:
                rules = f.read()
            
            # Add rules to context
            ctx.add_program(rules)
            
            print(f" Scallop engine loaded with rules from {self.scallop_rules_path}")
            
            return ctx
            
        except Exception as e:
            print(f"❌ Failed to load Scallop engine: {e}")
            print("   Using rule-based fallback")
            return None
    
    def detect_landing_zones(
        self, 
        image: np.ndarray,
        return_visualization: bool = True,
        return_intermediate: bool = False
    ) -> Dict[str, Any]:
        """
        Complete landing zone detection pipeline.
        
        Args:
            image: Input aerial image (RGB, shape: [H, W, 3])
            return_visualization: Whether to return visualization images
            return_intermediate: Whether to return intermediate results
            
        Returns:
            Dictionary containing:
            - landing_recommendations: List of recommended landing zones
            - hazard_analysis: Detected hazards and their locations
            - explanations: Logical explanations for decisions
            - visualizations: Optional visualization images
            - intermediate: Optional intermediate processing results
        """
        
        start_time = time.time()
        
        print(f"🔍 Processing aerial image: {image.shape}")
        
        # Step 1: Neural perception - semantic segmentation
        neural_results = self._neural_perception(image)
        
        # Step 2: Symbolic reasoning - landing logic
        symbolic_results = self._symbolic_reasoning(neural_results)
        
        # Step 3: Generate explanations
        explanations = self._generate_explanations(neural_results, symbolic_results)
        
        # Step 4: Create visualizations
        visualizations = {}
        if return_visualization:
            visualizations = self._create_visualizations(
                image, neural_results, symbolic_results
            )
        
        # Compile final results
        results = {
            'landing_recommendations': symbolic_results.get('landing_recommendations', []),
            'hazard_analysis': symbolic_results.get('hazard_analysis', {}),
            'explanations': explanations,
            'processing_time': time.time() - start_time,
            'neural_classes_detected': len(neural_results['unique_classes']),
            'total_zones_analyzed': len(symbolic_results.get('zone_analysis', []))
        }
        
        if return_visualization:
            results['visualizations'] = visualizations
        
        if return_intermediate:
            results['intermediate'] = {
                'neural_results': neural_results,
                'symbolic_results': symbolic_results
            }
        
        print(f" Processing complete: {results['processing_time']:.2f}s")
        print(f"   Found {len(results['landing_recommendations'])} landing zones")
        
        return results
    
    def _neural_perception(self, image: np.ndarray) -> Dict[str, Any]:
        """Neural network inference for semantic segmentation."""
        
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        with torch.no_grad():
            # Forward pass
            outputs = self.neural_model(processed_image)
            
            # Extract main prediction
            if isinstance(outputs, dict):
                predictions = outputs['main']
                uncertainty = outputs.get('uncertainty', None)
            else:
                predictions = outputs
                uncertainty = None
            
            # Convert to probabilities
            probabilities = F.softmax(predictions, dim=1)
            
            # Get class predictions
            class_predictions = torch.argmax(probabilities, dim=1)
            
            # Convert to numpy
            probabilities = probabilities.cpu().numpy()[0]  # Remove batch dimension
            class_predictions = class_predictions.cpu().numpy()[0]
            
            if uncertainty is not None:
                uncertainty = uncertainty.cpu().numpy()[0]
        
        # Analyze results
        unique_classes = np.unique(class_predictions)
        class_distribution = {}
        for class_id in unique_classes:
            mask = (class_predictions == class_id)
            pixel_count = mask.sum()
            avg_confidence = probabilities[class_id][mask].mean()
            class_distribution[int(class_id)] = {
                'name': self.class_info[class_id]['name'],
                'pixel_count': int(pixel_count),
                'avg_confidence': float(avg_confidence),
                'percentage': float(pixel_count / class_predictions.size * 100)
            }
        
        return {
            'probabilities': probabilities,
            'class_predictions': class_predictions,
            'uncertainty': uncertainty,
            'unique_classes': unique_classes,
            'class_distribution': class_distribution,
            'image_shape': image.shape[:2]
        }
    
    def _symbolic_reasoning(self, neural_results: Dict[str, Any]) -> Dict[str, Any]:
        """Symbolic reasoning using Scallop or rule-based fallback."""
        
        if self.scallop_engine is not None:
            return self._scallop_reasoning(neural_results)
        else:
            return self._rule_based_reasoning(neural_results)
    
    def _scallop_reasoning(self, neural_results: Dict[str, Any]) -> Dict[str, Any]:
        """Scallop-based logical reasoning."""
        
        # Convert neural results to Scallop facts
        facts = self._neural_to_scallop_facts(neural_results)
        
        # Add facts to Scallop context
        for fact in facts:
            self.scallop_engine.add_fact(fact)
        
        # Query for landing recommendations
        landing_recommendations = list(self.scallop_engine.query("query_landing_recommendations"))
        
        # Query for hazard analysis
        hazard_analysis = list(self.scallop_engine.query("query_hazards"))
        
        # Query for zone analysis
        zone_analysis = list(self.scallop_engine.query("query_zone_analysis"))
        
        return {
            'landing_recommendations': landing_recommendations,
            'hazard_analysis': hazard_analysis,
            'zone_analysis': zone_analysis,
            'reasoning_method': 'scallop'
        }
    
    def _rule_based_reasoning(self, neural_results: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based reasoning fallback."""
        
        print("🔄 Using rule-based reasoning fallback")
        
        probabilities = neural_results['probabilities']
        class_predictions = neural_results['class_predictions']
        height, width = neural_results['image_shape']
        
        # Identify safe surfaces
        safe_areas = []
        hazards = []
        
        for y in range(height):
            for x in range(width):
                class_id = class_predictions[y, x]
                confidence = probabilities[class_id, y, x]
                
                if confidence > self.confidence_threshold:
                    class_name = self.class_info[class_id]['name']
                    relevance = self.class_info[class_id]['landing_relevance']
                    
                    if relevance == 'safe' and confidence > 0.7:
                        safe_areas.append({
                            'x': int(x), 'y': int(y),
                            'class': class_name,
                            'confidence': float(confidence)
                        })
                    elif relevance == 'danger' and confidence > 0.5:
                        hazards.append({
                            'x': int(x), 'y': int(y),
                            'class': class_name,
                            'confidence': float(confidence),
                            'type': 'danger'
                        })
        
        # Simple zone detection (group nearby safe areas)
        landing_zones = self._cluster_safe_areas(safe_areas, hazards)
        
        return {
            'landing_recommendations': landing_zones,
            'hazard_analysis': hazards,
            'zone_analysis': [],
            'reasoning_method': 'rule_based'
        }
    
    def _neural_to_scallop_facts(self, neural_results: Dict[str, Any]) -> List[str]:
        """Convert neural network results to Scallop facts."""
        
        facts = []
        probabilities = neural_results['probabilities']
        height, width = neural_results['image_shape']
        
        # Sample pixels for efficiency (every Nth pixel)
        sample_rate = max(1, min(height, width) // 100)  # ~100x100 grid
        
        for y in range(0, height, sample_rate):
            for x in range(0, width, sample_rate):
                for class_id in range(24):
                    confidence = probabilities[class_id, y, x]
                    if confidence > self.confidence_threshold:
                        class_name = self.class_info[class_id]['name']
                        fact = f'PixelClass({x}, {y}, "{class_name}", {confidence:.3f})'
                        facts.append(fact)
        
        return facts
    
    def _cluster_safe_areas(self, safe_areas: List[Dict], hazards: List[Dict]) -> List[Dict]:
        """Simple clustering of safe areas into landing zones."""
        
        if not safe_areas:
            return []
        
        # Convert to numpy array for clustering
        points = np.array([[area['x'], area['y']] for area in safe_areas])
        
        # Simple distance-based clustering
        clusters = []
        visited = set()
        
        for i, point in enumerate(points):
            if i in visited:
                continue
            
            # Start new cluster
            cluster = [i]
            visited.add(i)
            
            # Find nearby points
            for j, other_point in enumerate(points):
                if j in visited:
                    continue
                
                distance = np.linalg.norm(point - other_point)
                if distance < 50:  # 50 pixel radius
                    cluster.append(j)
                    visited.add(j)
            
            if len(cluster) > 10:  # Minimum cluster size
                clusters.append(cluster)
        
        # Convert clusters to landing zone recommendations
        landing_zones = []
        for i, cluster in enumerate(clusters):
            cluster_points = points[cluster]
            center_x = int(cluster_points[:, 0].mean())
            center_y = int(cluster_points[:, 1].mean())
            
            # Check for nearby hazards
            nearby_hazards = []
            for hazard in hazards:
                distance = np.sqrt((hazard['x'] - center_x)**2 + (hazard['y'] - center_y)**2)
                if distance < 30:
                    nearby_hazards.append(hazard)
            
            # Compute safety score
            safety_score = min(0.9, 0.9 - len(nearby_hazards) * 0.1)
            
            if safety_score > 0.5:
                landing_zones.append({
                    'zone_id': i,
                    'center_x': center_x,
                    'center_y': center_y,
                    'area_pixels': len(cluster),
                    'safety_score': safety_score,
                    'confidence': safety_score,
                    'zone_type': 'primary' if safety_score > 0.8 else 'secondary',
                    'nearby_hazards': len(nearby_hazards)
                })
        
        # Sort by safety score
        landing_zones.sort(key=lambda x: x['safety_score'], reverse=True)
        
        return landing_zones
    
    def _generate_explanations(
        self, 
        neural_results: Dict[str, Any], 
        symbolic_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate human-readable explanations for decisions."""
        
        explanations = {
            'summary': '',
            'detailed': [],
            'neural_analysis': '',
            'symbolic_reasoning': '',
            'recommendations': []
        }
        
        # Neural analysis explanation
        class_dist = neural_results['class_distribution']
        dominant_classes = sorted(
            class_dist.items(), 
            key=lambda x: x[1]['percentage'], 
            reverse=True
        )[:5]
        
        neural_summary = f"Neural network detected {len(class_dist)} semantic classes. "
        neural_summary += "Dominant classes: " + ", ".join([
            f"{info['name']} ({info['percentage']:.1f}%)" 
            for _, info in dominant_classes
        ])
        explanations['neural_analysis'] = neural_summary
        
        # Symbolic reasoning explanation
        method = symbolic_results.get('reasoning_method', 'unknown')
        landing_zones = symbolic_results.get('landing_recommendations', [])
        hazards = symbolic_results.get('hazard_analysis', [])
        
        if method == 'scallop':
            symbolic_summary = f"Scallop logical reasoning identified {len(landing_zones)} potential landing zones "
            symbolic_summary += f"and {len(hazards)} hazards through rule-based analysis."
        else:
            symbolic_summary = f"Rule-based reasoning identified {len(landing_zones)} potential landing zones "
            symbolic_summary += f"and {len(hazards)} hazards through heuristic analysis."
        
        explanations['symbolic_reasoning'] = symbolic_summary
        
        # Landing recommendations
        for zone in landing_zones[:3]:  # Top 3 recommendations
            if 'explanation' in zone:
                explanation = zone['explanation']
            else:
                explanation = f"Zone {zone.get('zone_id', 'unknown')}: "
                explanation += f"Area={zone.get('area_pixels', 0)} pixels, "
                explanation += f"Safety={zone.get('safety_score', 0):.2f}, "
                explanation += f"Type={zone.get('zone_type', 'unknown')}"
            
            explanations['recommendations'].append({
                'zone_id': zone.get('zone_id'),
                'explanation': explanation,
                'confidence': zone.get('confidence', 0)
            })
        
        # Overall summary
        if landing_zones:
            best_zone = landing_zones[0]
            explanations['summary'] = f"Recommended landing: Zone {best_zone.get('zone_id')} "
            explanations['summary'] += f"(Safety: {best_zone.get('safety_score', 0):.1f}, "
            explanations['summary'] += f"Type: {best_zone.get('zone_type', 'unknown')})"
        else:
            explanations['summary'] = "No suitable landing zones identified - continue search or emergency procedures"
        
        return explanations
    
    def _create_visualizations(
        self,
        original_image: np.ndarray,
        neural_results: Dict[str, Any],
        symbolic_results: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Create visualization images for analysis."""
        
        visualizations = {}
        
        # 1. Semantic segmentation overlay
        seg_viz = self._visualize_segmentation(original_image, neural_results)
        visualizations['semantic_segmentation'] = seg_viz
        
        # 2. Landing zones visualization
        landing_viz = self._visualize_landing_zones(original_image, symbolic_results)
        visualizations['landing_zones'] = landing_viz
        
        # 3. Hazards visualization
        hazards_viz = self._visualize_hazards(original_image, symbolic_results)
        visualizations['hazards'] = hazards_viz
        
        # 4. Combined analysis
        combined_viz = self._visualize_combined_analysis(
            original_image, neural_results, symbolic_results
        )
        visualizations['combined_analysis'] = combined_viz
        
        return visualizations
    
    def _visualize_segmentation(self, image: np.ndarray, neural_results: Dict[str, Any]) -> np.ndarray:
        """Visualize semantic segmentation results."""
        
        class_predictions = neural_results['class_predictions']
        
        # Create colored segmentation map
        colors = self._get_class_colors()
        seg_colored = np.zeros((*class_predictions.shape, 3), dtype=np.uint8)
        
        for class_id in range(24):
            mask = (class_predictions == class_id)
            seg_colored[mask] = colors[class_id]
        
        # Resize to match original image
        if seg_colored.shape[:2] != image.shape[:2]:
            seg_colored = cv2.resize(seg_colored, (image.shape[1], image.shape[0]))
        
        # Blend with original image
        alpha = 0.6
        blended = cv2.addWeighted(image, 1-alpha, seg_colored, alpha, 0)
        
        return blended
    
    def _visualize_landing_zones(self, image: np.ndarray, symbolic_results: Dict[str, Any]) -> np.ndarray:
        """Visualize detected landing zones."""
        
        viz = image.copy()
        landing_zones = symbolic_results.get('landing_recommendations', [])
        
        for i, zone in enumerate(landing_zones):
            center_x = zone.get('center_x', 0)
            center_y = zone.get('center_y', 0)
            safety_score = zone.get('safety_score', 0)
            zone_type = zone.get('zone_type', 'unknown')
            
            # Color based on zone type
            if zone_type == 'primary':
                color = (0, 255, 0)  # Green
            elif zone_type == 'secondary':
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 165, 255)  # Orange
            
            # Draw circle for landing zone
            radius = max(20, int(zone.get('area_pixels', 100) ** 0.5 / 10))
            cv2.circle(viz, (center_x, center_y), radius, color, 3)
            
            # Add text label
            label = f"Zone {i+1}: {safety_score:.2f}"
            cv2.putText(viz, label, (center_x-30, center_y-radius-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return viz
    
    def _visualize_hazards(self, image: np.ndarray, symbolic_results: Dict[str, Any]) -> np.ndarray:
        """Visualize detected hazards."""
        
        viz = image.copy()
        hazards = symbolic_results.get('hazard_analysis', [])
        
        for hazard in hazards:
            if isinstance(hazard, dict):
                x = hazard.get('x', 0)
                y = hazard.get('y', 0)
                hazard_type = hazard.get('type', 'unknown')
            else:
                # Handle tuple format from Scallop
                x, y, hazard_type, severity = hazard
            
            # Color based on hazard type
            if 'danger' in str(hazard_type).lower():
                color = (0, 0, 255)  # Red
            elif 'obstacle' in str(hazard_type).lower():
                color = (0, 128, 255)  # Orange
            else:
                color = (255, 0, 255)  # Magenta
            
            # Draw hazard marker
            cv2.circle(viz, (int(x), int(y)), 8, color, -1)
            cv2.circle(viz, (int(x), int(y)), 12, color, 2)
        
        return viz
    
    def _visualize_combined_analysis(
        self, 
        image: np.ndarray, 
        neural_results: Dict[str, Any],
        symbolic_results: Dict[str, Any]
    ) -> np.ndarray:
        """Create combined visualization showing all analysis results."""
        
        # Start with segmentation overlay
        viz = self._visualize_segmentation(image, neural_results)
        
        # Add landing zones
        landing_zones = symbolic_results.get('landing_recommendations', [])
        for i, zone in enumerate(landing_zones):
            center_x = zone.get('center_x', 0)
            center_y = zone.get('center_y', 0)
            safety_score = zone.get('safety_score', 0)
            
            # Green circle for recommended zones
            cv2.circle(viz, (center_x, center_y), 25, (0, 255, 0), 3)
            cv2.putText(viz, f"{i+1}", (center_x-8, center_y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add hazards
        hazards = symbolic_results.get('hazard_analysis', [])
        for hazard in hazards:
            if isinstance(hazard, dict):
                x, y = hazard.get('x', 0), hazard.get('y', 0)
            else:
                x, y = hazard[0], hazard[1]
            
            # Red X for hazards
            cv2.line(viz, (int(x)-8, int(y)-8), (int(x)+8, int(y)+8), (0, 0, 255), 3)
            cv2.line(viz, (int(x)-8, int(y)+8), (int(x)+8, int(y)-8), (0, 0, 255), 3)
        
        return viz
    
    def _get_class_colors(self) -> np.ndarray:
        """Get distinct colors for each semantic class."""
        
        colors = np.array([
            [0, 0, 0],        # unlabeled - black
            [128, 64, 128],   # paved-area - purple
            [130, 76, 0],     # dirt - brown
            [0, 102, 0],      # grass - green
            [112, 103, 87],   # gravel - gray
            [28, 42, 168],    # water - blue
            [48, 41, 30],     # rocks - dark brown
            [0, 50, 89],      # pool - dark blue
            [107, 142, 35],   # vegetation - olive
            [70, 70, 70],     # roof - gray
            [102, 102, 156],  # wall - light purple
            [254, 228, 12],   # window - yellow
            [254, 148, 12],   # door - orange
            [190, 153, 153],  # fence - pink
            [153, 153, 153],  # fence-pole - light gray
            [255, 22, 96],    # person - bright pink
            [102, 51, 0],     # dog - dark brown
            [9, 143, 150],    # car - teal
            [119, 11, 32],    # bicycle - dark red
            [51, 51, 0],      # tree - dark green
            [190, 250, 190],  # bald-tree - light green
            [112, 150, 146],  # ar-marker - cyan
            [2, 135, 115],    # obstacle - dark cyan
            [255, 0, 0],      # conflicting - red
        ], dtype=np.uint8)
        
        return colors
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for neural network input."""
        
        # Resize to model input size
        target_size = (512, 512)
        if image.shape[:2] != target_size:
            image = cv2.resize(image, target_size)
        
        # Convert to float and normalize
        image = image.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to file."""
        
        # Prepare data for JSON serialization
        json_data = {
            'landing_recommendations': results['landing_recommendations'],
            'hazard_analysis': results['hazard_analysis'],
            'explanations': results['explanations'],
            'processing_time': results['processing_time'],
            'neural_classes_detected': results['neural_classes_detected'],
            'total_zones_analyzed': results['total_zones_analyzed']
        }
        
        # Save JSON
        json_path = Path(output_path).with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Save visualizations if available
        if 'visualizations' in results:
            viz_dir = Path(output_path).parent / 'visualizations'
            viz_dir.mkdir(exist_ok=True)
            
            for name, viz_image in results['visualizations'].items():
                viz_path = viz_dir / f"{Path(output_path).stem}_{name}.jpg"
                cv2.imwrite(str(viz_path), cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))
        
        print(f"💾 Results saved: {json_path}")


def main():
    """Example usage of the neuro-symbolic landing system."""
    
    # Paths (adjust as needed)
    neural_model_path = "outputs/aerial_semantic_24/aerial_semantic_24_best.pth"
    scallop_rules_path = "scallop_integration/landing_rules.scl"
    test_image_path = "datasets/Aerial_Semantic_Segmentation_Drone_Dataset/dataset/semantic_drone_dataset/original_images/001.jpg"
    
    # Create system
    system = NeuroSymbolicLandingSystem(
        neural_model_path=neural_model_path,
        scallop_rules_path=scallop_rules_path,
        device='auto'
    )
    
    # Load test image
    if Path(test_image_path).exists():
        image = cv2.imread(str(test_image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"\n🧪 Testing on image: {test_image_path}")
        
        # Detect landing zones
        results = system.detect_landing_zones(
            image, 
            return_visualization=True,
            return_intermediate=True
        )
        
        # Print results
        print(f"\n📋 Results Summary:")
        print(f"   {results['explanations']['summary']}")
        print(f"   Processing time: {results['processing_time']:.2f}s")
        print(f"   Landing zones: {len(results['landing_recommendations'])}")
        
        for i, rec in enumerate(results['explanations']['recommendations']):
            print(f"   Zone {i+1}: {rec['explanation']}")
        
        # Save results
        system.save_results(results, "outputs/neuro_symbolic_test_results")
        
    else:
        print(f"❌ Test image not found: {test_image_path}")
        print("   Please train a model first or adjust the paths")


if __name__ == "__main__":
    main() 