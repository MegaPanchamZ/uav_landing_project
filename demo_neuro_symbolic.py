#!/usr/bin/env python3
"""
UAV Landing System - Neuro-Symbolic Demo with Scallop
====================================================

Demonstration of neuro-symbolic reasoning for UAV landing safety decisions.
Combines neural network predictions with logical safety rules using Scallop.

Features:
- Load trained segmentation model
- Process drone imagery
- Apply Scallop-based safety reasoning
- Generate comprehensive safety reports
- Visualize decision process

Usage:
    python demo_neuro_symbolic.py --image test_image.jpg
    python demo_neuro_symbolic.py --video test_video.mp4
    python demo_neuro_symbolic.py --demo_mode
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

# Scallop integration
try:
    import scallop
    SCALLOP_AVAILABLE = True
except ImportError:
    SCALLOP_AVAILABLE = False
    print("⚠️  Scallop not available. Install from: https://github.com/scallop-lang/scallop")

# Import our components
from models.mobilenetv3_edge_model import create_edge_model
from datasets.semantic_drone_dataset import create_semantic_drone_transforms


class LandingSafetyAnalyzer:
    """Neuro-symbolic landing safety analyzer using Scallop."""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """Initialize the analyzer with a trained model."""
        
        # Auto-detect device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Load model
        print(f"🧠 Loading neural network model...")
        self.model = self._load_model(model_path)
        
        # Class information
        self.class_names = ['ground', 'vegetation', 'obstacle', 'water', 'vehicle', 'other']
        self.class_colors = [
            [128, 64, 128],    # ground - purple
            [107, 142, 35],    # vegetation - olive
            [70, 70, 70],      # obstacle - dark gray
            [0, 0, 142],       # water - dark blue
            [0, 0, 70],        # vehicle - dark red
            [102, 102, 156]    # other - light purple
        ]
        
        # Safety mappings
        self.safety_levels = {
            'ground': 'safe',
            'vegetation': 'caution',
            'obstacle': 'danger',
            'water': 'danger',
            'vehicle': 'danger',
            'other': 'unknown'
        }
        
        # Initialize Scallop reasoning engine
        if SCALLOP_AVAILABLE:
            self.scallop_ctx = self._initialize_scallop()
        else:
            self.scallop_ctx = None
            print("⚠️  Scallop unavailable - using rule-based fallback")
        
        # Transform for input images
        self.transform = create_semantic_drone_transforms(
            input_size=(512, 512),
            is_training=False
        )
        
        print(f"✅ Neuro-symbolic analyzer ready")
        print(f"   Device: {self.device}")
        print(f"   Scallop integration: {'✅' if SCALLOP_AVAILABLE else '❌'}")
    
    def _load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = create_edge_model(
            model_type='enhanced',
            num_classes=6,
            use_uncertainty=True,
            pretrained=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def _initialize_scallop(self):
        """Initialize Scallop reasoning context with landing safety rules."""
        
        # Define Scallop program for landing safety reasoning
        scallop_program = '''
        // Landing Safety Knowledge Base
        
        // Basic safety classifications
        rel safe_surface = {
            "ground", "vegetation"
        }
        
        rel hazardous_surface = {
            "obstacle", "water", "vehicle"
        }
        
        rel uncertain_surface = {
            "other"
        }
        
        // Weather and environmental factors
        type weather_condition = String
        rel weather_factor = {
            ("clear", 1.0),
            ("cloudy", 0.9),
            ("windy", 0.7),
            ("rainy", 0.3),
            ("stormy", 0.1)
        }
        
        // UAV specifications
        type uav_type = String
        rel uav_landing_capability = {
            ("small_drone", 0.8),
            ("medium_drone", 0.6),
            ("large_drone", 0.4),
            ("helicopter", 0.9)
        }
        
        // Area analysis rules
        rel dominant_surface(surface) = surface_area(surface, area) and 
                                      area == max(a: surface_area(_, a))
        
        rel area_safety_score(score) = dominant_surface(surface) and
                                     safe_surface(surface) and
                                     score := 0.9
        
        rel area_safety_score(score) = dominant_surface(surface) and
                                     hazardous_surface(surface) and
                                     score := 0.1
        
        rel area_safety_score(score) = dominant_surface(surface) and
                                     uncertain_surface(surface) and
                                     score := 0.5
        
        // Landing site evaluation
        rel suitable_landing_site(x, y, confidence) = 
            local_area_safe(x, y) and
            sufficient_space(x, y) and
            confidence := 0.8
        
        rel local_area_safe(x, y) = 
            nearby_surface(x, y, surface) and
            safe_surface(surface)
        
        rel sufficient_space(x, y) = 
            landing_space_available(x, y, space) and
            space > 20  // minimum 20 pixel radius
        
        // Emergency landing assessment
        rel emergency_landing_possible(x, y, risk_level) = 
            suitable_landing_site(x, y, _) and
            risk_level := "low"
        
        rel emergency_landing_possible(x, y, risk_level) = 
            local_area_safe(x, y) and
            not sufficient_space(x, y) and
            risk_level := "medium"
        
        rel emergency_landing_possible(x, y, risk_level) = 
            not local_area_safe(x, y) and
            not hazardous_surface_nearby(x, y) and
            risk_level := "high"
        
        rel hazardous_surface_nearby(x, y) = 
            nearby_surface(x, y, surface) and
            hazardous_surface(surface)
        
        // Overall mission safety assessment
        rel mission_safety_level(level) = 
            area_safety_score(score) and
            current_weather(weather) and
            weather_factor(weather, weather_mult) and
            uav_capability(uav) and
            uav_landing_capability(uav, uav_mult) and
            final_score := score * weather_mult * uav_mult and
            final_score > 0.7 and
            level := "safe"
        
        rel mission_safety_level(level) = 
            area_safety_score(score) and
            current_weather(weather) and
            weather_factor(weather, weather_mult) and
            uav_capability(uav) and
            uav_landing_capability(uav, uav_mult) and
            final_score := score * weather_mult * uav_mult and
            final_score > 0.4 and final_score <= 0.7 and
            level := "caution"
        
        rel mission_safety_level(level) = 
            area_safety_score(score) and
            current_weather(weather) and
            weather_factor(weather, weather_mult) and
            uav_capability(uav) and
            uav_landing_capability(uav, uav_mult) and
            final_score := score * weather_mult * uav_mult and
            final_score <= 0.4 and
            level := "danger"
        '''
        
        try:
            ctx = scallop.ScallopContext()
            ctx.add_program(scallop_program)
            print("✅ Scallop reasoning engine initialized")
            return ctx
        except Exception as e:
            print(f"❌ Failed to initialize Scallop: {e}")
            return None
    
    def analyze_image(
        self, 
        image_path: str, 
        weather: str = "clear",
        uav_type: str = "small_drone",
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Analyze a single image for landing safety."""
        
        print(f"\n🔍 Analyzing image: {image_path}")
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Neural network inference
        print("🧠 Running neural network inference...")
        predictions, confidence_map = self._run_inference(image_rgb)
        
        # Extract segmentation statistics
        stats = self._compute_segmentation_stats(predictions)
        
        # Neuro-symbolic reasoning
        print("🔗 Running neuro-symbolic reasoning...")
        safety_analysis = self._run_safety_reasoning(
            predictions, stats, weather, uav_type
        )
        
        # Find potential landing sites
        landing_sites = self._identify_landing_sites(predictions, confidence_map)
        
        # Comprehensive results
        results = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'neural_predictions': {
                'segmentation_stats': stats,
                'confidence_map_available': confidence_map is not None
            },
            'safety_analysis': safety_analysis,
            'landing_sites': landing_sites,
            'parameters': {
                'weather': weather,
                'uav_type': uav_type
            }
        }
        
        # Visualization and saving
        if save_results:
            self._create_analysis_visualization(
                image_rgb, predictions, confidence_map, results, image_path
            )
        
        return results
    
    def _run_inference(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Run neural network inference on image."""
        
        # Preprocess
        if hasattr(self.transform, 'replay'):
            # Handle albumentations transform
            transformed = self.transform(image=image)
            input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        else:
            # Handle torchvision transform
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            if isinstance(outputs, dict):
                predictions = outputs['main']
                uncertainty = outputs.get('uncertainty', None)
            else:
                predictions = outputs
                uncertainty = None
        
        # Convert to numpy
        pred_probs = F.softmax(predictions, dim=1)
        pred_classes = predictions.argmax(dim=1)
        
        pred_classes_np = pred_classes[0].cpu().numpy()
        
        confidence_map = None
        if uncertainty is not None:
            # Convert uncertainty to confidence
            confidence_map = (1.0 - torch.sigmoid(uncertainty[0])).cpu().numpy()
        
        return pred_classes_np, confidence_map
    
    def _compute_segmentation_stats(self, predictions: np.ndarray) -> Dict[str, float]:
        """Compute statistics from segmentation predictions."""
        
        total_pixels = predictions.size
        stats = {}
        
        for class_id, class_name in enumerate(self.class_names):
            class_pixels = (predictions == class_id).sum()
            percentage = (class_pixels / total_pixels) * 100
            stats[class_name] = {
                'pixels': int(class_pixels),
                'percentage': float(percentage),
                'safety_level': self.safety_levels[class_name]
            }
        
        return stats
    
    def _run_safety_reasoning(
        self, 
        predictions: np.ndarray, 
        stats: Dict, 
        weather: str, 
        uav_type: str
    ) -> Dict[str, Any]:
        """Run Scallop-based safety reasoning."""
        
        if self.scallop_ctx is not None:
            return self._scallop_reasoning(predictions, stats, weather, uav_type)
        else:
            return self._fallback_reasoning(predictions, stats, weather, uav_type)
    
    def _scallop_reasoning(
        self, 
        predictions: np.ndarray, 
        stats: Dict, 
        weather: str, 
        uav_type: str
    ) -> Dict[str, Any]:
        """Use Scallop for symbolic reasoning."""
        
        try:
            # Clear previous facts
            ctx = self.scallop_ctx.clone()
            
            # Add surface area facts
            for class_name, class_stats in stats.items():
                percentage = class_stats['percentage']
                ctx.add_facts("surface_area", [(class_name, percentage)])
            
            # Add context facts
            ctx.add_facts("current_weather", [(weather,)])
            ctx.add_facts("uav_capability", [(uav_type,)])
            
            # Add spatial facts (simplified grid-based analysis)
            height, width = predictions.shape
            grid_size = 32  # Analyze in 32x32 pixel grids
            
            for y in range(0, height, grid_size):
                for x in range(0, width, grid_size):
                    # Get dominant surface in this grid cell
                    grid_region = predictions[y:y+grid_size, x:x+grid_size]
                    if grid_region.size > 0:
                        dominant_class = np.bincount(grid_region.flat).argmax()
                        surface_name = self.class_names[dominant_class]
                        
                        ctx.add_facts("nearby_surface", [(x//grid_size, y//grid_size, surface_name)])
                        
                        # Check for sufficient landing space
                        space_radius = min(grid_size // 2, 20)
                        ctx.add_facts("landing_space_available", [(x//grid_size, y//grid_size, space_radius)])
            
            # Run reasoning
            ctx.run()
            
            # Extract results
            mission_safety = list(ctx.relation("mission_safety_level"))
            area_safety = list(ctx.relation("area_safety_score"))
            landing_sites = list(ctx.relation("suitable_landing_site"))
            emergency_sites = list(ctx.relation("emergency_landing_possible"))
            
            return {
                'reasoning_engine': 'scallop',
                'mission_safety_level': mission_safety[0][0] if mission_safety else 'unknown',
                'area_safety_score': area_safety[0][0] if area_safety else 0.0,
                'suitable_landing_sites': len(landing_sites),
                'emergency_landing_sites': len(emergency_sites),
                'detailed_analysis': {
                    'landing_sites': landing_sites,
                    'emergency_sites': emergency_sites
                }
            }
            
        except Exception as e:
            print(f"⚠️  Scallop reasoning failed: {e}, falling back to rule-based")
            return self._fallback_reasoning(predictions, stats, weather, uav_type)
    
    def _fallback_reasoning(
        self, 
        predictions: np.ndarray, 
        stats: Dict, 
        weather: str, 
        uav_type: str
    ) -> Dict[str, Any]:
        """Fallback rule-based reasoning when Scallop is unavailable."""
        
        # Simple rule-based analysis
        safe_percentage = sum(
            s['percentage'] for s in stats.values() 
            if s['safety_level'] == 'safe'
        )
        danger_percentage = sum(
            s['percentage'] for s in stats.values() 
            if s['safety_level'] == 'danger'
        )
        
        # Weather factor
        weather_factors = {
            'clear': 1.0, 'cloudy': 0.9, 'windy': 0.7, 
            'rainy': 0.3, 'stormy': 0.1
        }
        weather_factor = weather_factors.get(weather, 0.8)
        
        # UAV factor
        uav_factors = {
            'small_drone': 0.8, 'medium_drone': 0.6, 
            'large_drone': 0.4, 'helicopter': 0.9
        }
        uav_factor = uav_factors.get(uav_type, 0.7)
        
        # Calculate safety score
        base_score = (safe_percentage / 100.0) - (danger_percentage / 100.0 * 0.5)
        final_score = base_score * weather_factor * uav_factor
        
        # Determine safety level
        if final_score > 0.7:
            safety_level = 'safe'
        elif final_score > 0.4:
            safety_level = 'caution'
        else:
            safety_level = 'danger'
        
        return {
            'reasoning_engine': 'rule_based_fallback',
            'mission_safety_level': safety_level,
            'area_safety_score': final_score,
            'safe_area_percentage': safe_percentage,
            'danger_area_percentage': danger_percentage,
            'weather_factor': weather_factor,
            'uav_factor': uav_factor
        }
    
    def _identify_landing_sites(
        self, 
        predictions: np.ndarray, 
        confidence_map: Optional[np.ndarray]
    ) -> List[Dict]:
        """Identify potential landing sites from predictions."""
        
        # Find safe areas (ground and vegetation)
        safe_mask = (predictions == 0) | (predictions == 1)  # ground or vegetation
        
        # Use confidence if available
        if confidence_map is not None:
            safe_mask = safe_mask & (confidence_map > 0.7)
        
        # Find connected components
        safe_mask_uint8 = safe_mask.astype(np.uint8)
        contours, _ = cv2.findContours(safe_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        landing_sites = []
        min_area = 400  # Minimum area for landing site
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area >= min_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate center
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Estimate safety score
                region = predictions[y:y+h, x:x+w]
                safe_pixels = ((region == 0) | (region == 1)).sum()
                safety_score = safe_pixels / (w * h)
                
                landing_sites.append({
                    'id': i,
                    'center': (int(center_x), int(center_y)),
                    'bounding_box': (int(x), int(y), int(w), int(h)),
                    'area': int(area),
                    'safety_score': float(safety_score),
                    'recommended': safety_score > 0.8 and area > 1000
                })
        
        # Sort by safety score
        landing_sites.sort(key=lambda x: x['safety_score'], reverse=True)
        
        return landing_sites
    
    def _create_analysis_visualization(
        self, 
        image: np.ndarray, 
        predictions: np.ndarray, 
        confidence_map: Optional[np.ndarray],
        results: Dict,
        image_path: str
    ):
        """Create comprehensive visualization of the analysis."""
        
        # Setup figure
        fig_width = 20
        fig_height = 12 if confidence_map is not None else 8
        fig, axes = plt.subplots(2 if confidence_map is not None else 1, 3, figsize=(fig_width, fig_height))
        
        if confidence_map is not None:
            axes = axes.flatten()
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Segmentation prediction
        colored_pred = self._colorize_predictions(predictions)
        axes[1].imshow(colored_pred)
        axes[1].set_title('Segmentation Prediction', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Add class legend
        self._add_class_legend(axes[1])
        
        # Landing sites overlay
        axes[2].imshow(image)
        self._overlay_landing_sites(axes[2], results['landing_sites'])
        axes[2].set_title('Identified Landing Sites', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        # Confidence map (if available)
        if confidence_map is not None:
            im = axes[3].imshow(confidence_map, cmap='viridis', vmin=0, vmax=1)
            axes[3].set_title('Confidence Map', fontsize=14, fontweight='bold')
            axes[3].axis('off')
            plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
            
            # Safety analysis text
            self._add_safety_analysis_text(axes[4], results)
            
            # Statistics pie chart
            self._create_statistics_pie_chart(axes[5], results['neural_predictions']['segmentation_stats'])
        else:
            # Safety analysis text
            self._add_safety_analysis_text(axes[3], results)
            
            # Statistics pie chart (placeholder for now)
            axes[3].text(0.5, 0.5, 'Statistics\n(pie chart)', 
                        ha='center', va='center', transform=axes[3].transAxes)
            axes[3].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = Path(image_path).stem + '_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 Analysis visualization saved: {output_path}")
        
        plt.show()
    
    def _colorize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Convert prediction mask to colored visualization."""
        colored = np.zeros((predictions.shape[0], predictions.shape[1], 3), dtype=np.uint8)
        
        for class_id in range(6):
            mask = predictions == class_id
            colored[mask] = self.class_colors[class_id]
        
        return colored
    
    def _add_class_legend(self, ax):
        """Add class legend to the plot."""
        legend_elements = []
        for i, (class_name, color) in enumerate(zip(self.class_names, self.class_colors)):
            legend_elements.append(
                patches.Patch(color=np.array(color)/255.0, label=class_name.title())
            )
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    def _overlay_landing_sites(self, ax, landing_sites: List[Dict]):
        """Overlay landing sites on the image."""
        for site in landing_sites:
            x, y, w, h = site['bounding_box']
            center_x, center_y = site['center']
            
            # Color based on recommendation
            color = 'green' if site['recommended'] else 'orange'
            
            # Draw bounding box
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Draw center point
            ax.plot(center_x, center_y, 'o', color=color, markersize=8)
            
            # Add label
            ax.text(center_x, center_y - 20, f"Site {site['id']}\n{site['safety_score']:.2f}", 
                   ha='center', va='bottom', color=color, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def _add_safety_analysis_text(self, ax, results: Dict):
        """Add safety analysis text to the plot."""
        ax.axis('off')
        
        safety_analysis = results['safety_analysis']
        
        text = f"""MISSION SAFETY ANALYSIS
        
Safety Level: {safety_analysis['mission_safety_level'].upper()}
Area Safety Score: {safety_analysis.get('area_safety_score', 'N/A'):.3f}

Weather: {results['parameters']['weather'].title()}
UAV Type: {results['parameters']['uav_type'].replace('_', ' ').title()}

Reasoning Engine: {safety_analysis['reasoning_engine'].replace('_', ' ').title()}

Landing Sites Found: {len(results['landing_sites'])}
Recommended Sites: {sum(1 for site in results['landing_sites'] if site['recommended'])}
        """
        
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    def _create_statistics_pie_chart(self, ax, stats: Dict):
        """Create pie chart of surface type statistics."""
        labels = []
        sizes = []
        colors = []
        
        for class_name, class_stats in stats.items():
            if class_stats['percentage'] > 1:  # Only show classes with >1%
                labels.append(f"{class_name.title()}\n({class_stats['percentage']:.1f}%)")
                sizes.append(class_stats['percentage'])
                
                # Get color based on class
                class_id = self.class_names.index(class_name)
                colors.append(np.array(self.class_colors[class_id]) / 255.0)
        
        if sizes:
            ax.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
            ax.set_title('Surface Type Distribution', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No significant\nsurface types', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')


def create_demo_visualization():
    """Create a demo visualization with synthetic data."""
    
    print("🎭 Creating demo visualization...")
    
    # Create synthetic segmentation result
    height, width = 512, 512
    demo_prediction = np.zeros((height, width), dtype=np.uint8)
    
    # Add some ground areas (safe landing zones)
    demo_prediction[100:200, 100:300] = 0  # ground
    demo_prediction[300:400, 200:400] = 0  # ground
    
    # Add vegetation
    demo_prediction[50:150, 350:450] = 1  # vegetation
    demo_prediction[400:500, 50:150] = 1  # vegetation
    
    # Add obstacles
    demo_prediction[200:250, 150:200] = 2  # obstacle
    demo_prediction[350:380, 350:380] = 2  # obstacle
    
    # Add water
    demo_prediction[80:120, 200:300] = 3  # water
    
    # Add vehicles
    demo_prediction[250:270, 250:280] = 4  # vehicle
    
    # Create synthetic image
    demo_image = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    
    # Color the synthetic image based on predictions
    class_colors = [
        [139, 69, 19],     # ground - brown
        [34, 139, 34],     # vegetation - forest green
        [105, 105, 105],   # obstacle - dim gray
        [0, 191, 255],     # water - deep sky blue
        [220, 20, 60],     # vehicle - crimson
        [128, 128, 128]    # other - gray
    ]
    
    for class_id in range(6):
        mask = demo_prediction == class_id
        if mask.any():
            demo_image[mask] = class_colors[class_id]
    
    # Add some noise
    noise = np.random.randint(-30, 30, demo_image.shape)
    demo_image = np.clip(demo_image.astype(int) + noise, 0, 255).astype(np.uint8)
    
    # Mock results
    mock_results = {
        'neural_predictions': {
            'segmentation_stats': {
                'ground': {'percentage': 35.2, 'safety_level': 'safe'},
                'vegetation': {'percentage': 28.1, 'safety_level': 'caution'},
                'obstacle': {'percentage': 12.5, 'safety_level': 'danger'},
                'water': {'percentage': 8.3, 'safety_level': 'danger'},
                'vehicle': {'percentage': 3.1, 'safety_level': 'danger'},
                'other': {'percentage': 12.8, 'safety_level': 'unknown'}
            }
        },
        'safety_analysis': {
            'reasoning_engine': 'demo_mode',
            'mission_safety_level': 'caution',
            'area_safety_score': 0.632,
            'safe_area_percentage': 63.3,
            'danger_area_percentage': 23.9
        },
        'landing_sites': [
            {
                'id': 0,
                'center': (200, 150),
                'bounding_box': (100, 100, 200, 100),
                'area': 20000,
                'safety_score': 0.92,
                'recommended': True
            },
            {
                'id': 1,
                'center': (300, 350),
                'bounding_box': (200, 300, 200, 100),
                'area': 20000,
                'safety_score': 0.85,
                'recommended': True
            },
            {
                'id': 2,
                'center': (400, 100),
                'bounding_box': (350, 50, 100, 100),
                'area': 10000,
                'safety_score': 0.71,
                'recommended': False
            }
        ],
        'parameters': {
            'weather': 'clear',
            'uav_type': 'small_drone'
        }
    }
    
    # Create analyzer instance (without model for demo)
    analyzer = type('DemoAnalyzer', (), {
        'class_names': ['ground', 'vegetation', 'obstacle', 'water', 'vehicle', 'other'],
        'class_colors': class_colors,
        '_colorize_predictions': lambda self, pred: np.array([class_colors[p] for p in pred.flat]).reshape(pred.shape + (3,)),
        '_add_class_legend': lambda self, ax: None,
        '_overlay_landing_sites': LandingSafetyAnalyzer._overlay_landing_sites,
        '_add_safety_analysis_text': LandingSafetyAnalyzer._add_safety_analysis_text,
        '_create_statistics_pie_chart': LandingSafetyAnalyzer._create_statistics_pie_chart
    })()
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Original synthetic image
    axes[0].imshow(demo_image)
    axes[0].set_title('Demo Aerial Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Segmentation prediction
    colored_pred = analyzer._colorize_predictions(demo_prediction)
    axes[1].imshow(colored_pred)
    axes[1].set_title('Neural Network Segmentation', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Landing sites
    axes[2].imshow(demo_image)
    analyzer._overlay_landing_sites(axes[2], mock_results['landing_sites'])
    axes[2].set_title('Neuro-Symbolic Analysis Results', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_neuro_symbolic_analysis.png', dpi=150, bbox_inches='tight')
    print("💾 Demo visualization saved: demo_neuro_symbolic_analysis.png")
    plt.show()
    
    return mock_results


def main():
    parser = argparse.ArgumentParser(description='UAV Landing Neuro-Symbolic Demo')
    
    # Input options
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--demo_mode', action='store_true', help='Run demo with synthetic data')
    
    # Model and configuration
    parser.add_argument('--model', type=str, default='outputs/stage3_best.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    # Environment parameters
    parser.add_argument('--weather', type=str, default='clear',
                        choices=['clear', 'cloudy', 'windy', 'rainy', 'stormy'],
                        help='Weather conditions')
    parser.add_argument('--uav_type', type=str, default='small_drone',
                        choices=['small_drone', 'medium_drone', 'large_drone', 'helicopter'],
                        help='UAV type')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='neuro_symbolic_results',
                        help='Output directory for results')
    parser.add_argument('--save_json', action='store_true', help='Save results as JSON')
    
    args = parser.parse_args()
    
    print("🔗 UAV Landing Neuro-Symbolic Demo")
    print("==================================")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.demo_mode:
            # Run demo mode
            results = create_demo_visualization()
            
            if args.save_json:
                json_path = output_dir / 'demo_results.json'
                with open(json_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"💾 Demo results saved: {json_path}")
            
        elif args.image:
            # Analyze single image
            if not Path(args.model).exists():
                print(f"❌ Model not found: {args.model}")
                print("   Please train a model first or use --demo_mode")
                return
            
            analyzer = LandingSafetyAnalyzer(args.model, args.device)
            results = analyzer.analyze_image(
                args.image, 
                weather=args.weather,
                uav_type=args.uav_type,
                save_results=True
            )
            
            # Print summary
            safety_level = results['safety_analysis']['mission_safety_level']
            landing_sites = len(results['landing_sites'])
            recommended_sites = sum(1 for site in results['landing_sites'] if site['recommended'])
            
            print(f"\n🎉 Analysis Complete!")
            print(f"   Safety Level: {safety_level.upper()}")
            print(f"   Landing Sites: {landing_sites} found, {recommended_sites} recommended")
            
            if args.save_json:
                json_path = output_dir / f"{Path(args.image).stem}_results.json"
                with open(json_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"   Results saved: {json_path}")
        
        elif args.video:
            print("🎬 Video analysis not yet implemented")
            print("   Use --image for single image analysis or --demo_mode for demonstration")
        
        else:
            parser.print_help()
            print("\n💡 Quick start:")
            print("   python demo_neuro_symbolic.py --demo_mode")
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 