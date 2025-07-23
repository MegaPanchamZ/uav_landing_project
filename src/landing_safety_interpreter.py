#!/usr/bin/env python3
"""
Landing Safety Interpreter
=========================

Interprets 24-class semantic segmentation results into landing safety zones.
This separates semantic understanding from safety assessment, following the
principle of working WITH the dataset structure.

Semantic Classes → Landing Safety Mapping:
- SAFE: paved-area, dirt, grass, gravel (flat, clear surfaces)
- CAUTION: vegetation, pool (manageable but need care)  
- DANGEROUS: water, rocks, roof, wall, trees, obstacles, people, vehicles
- IGNORE: unlabeled, windows, doors, fences (architectural details)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from enum import Enum

class LandingSafety(Enum):
    """Landing safety levels."""
    SAFE = 0
    CAUTION = 1  
    DANGEROUS = 2
    IGNORE = 3  # Areas not relevant for landing assessment

class LandingSafetyInterpreter:
    """
    Interprets semantic segmentation results into landing safety zones.
    
    This class takes 24-class semantic predictions and converts them to
    landing safety assessments based on aeronautical safety principles.
    """
    
    # Mapping from semantic class index to landing safety
    SEMANTIC_TO_SAFETY = {
        # SAFE - Suitable landing surfaces
        1: LandingSafety.SAFE,      # paved-area (concrete, asphalt)
        2: LandingSafety.SAFE,      # dirt (clear ground)
        3: LandingSafety.SAFE,      # grass (soft landing)
        4: LandingSafety.SAFE,      # gravel (stable surface)
        
        # CAUTION - Manageable but requires care
        7: LandingSafety.CAUTION,   # pool (water hazard, but contained)
        8: LandingSafety.CAUTION,   # vegetation (low plants, manageable)
        
        # DANGEROUS - Avoid these areas
        5: LandingSafety.DANGEROUS,  # water (crash risk)
        6: LandingSafety.DANGEROUS,  # rocks (hard/uneven surface)
        9: LandingSafety.DANGEROUS,  # roof (hard surface, property damage)
        10: LandingSafety.DANGEROUS, # wall (obstacle)
        15: LandingSafety.DANGEROUS, # person (safety hazard)
        16: LandingSafety.DANGEROUS, # dog (moving obstacle)
        17: LandingSafety.DANGEROUS, # car (valuable property)
        18: LandingSafety.DANGEROUS, # bicycle (property)
        19: LandingSafety.DANGEROUS, # tree (tall obstacle)
        20: LandingSafety.DANGEROUS, # bald-tree (still an obstacle)
        22: LandingSafety.DANGEROUS, # obstacle (explicit hazard)
        23: LandingSafety.DANGEROUS, # conflicting (uncertain area)
        
        # IGNORE - Architectural details not relevant for landing
        0: LandingSafety.IGNORE,     # unlabeled
        11: LandingSafety.IGNORE,    # window (building detail)
        12: LandingSafety.IGNORE,    # door (building detail)
        13: LandingSafety.IGNORE,    # fence (boundary marker)
        14: LandingSafety.IGNORE,    # fence-pole (boundary marker)
        21: LandingSafety.IGNORE,    # ar-marker (navigation aid)
    }
    
    # Class names for reference
    SEMANTIC_CLASS_NAMES = [
        'unlabeled', 'paved-area', 'dirt', 'grass', 'gravel', 'water',
        'rocks', 'pool', 'vegetation', 'roof', 'wall', 'window',
        'door', 'fence', 'fence-pole', 'person', 'dog', 'car',
        'bicycle', 'tree', 'bald-tree', 'ar-marker', 'obstacle', 'conflicting'
    ]
    
    SAFETY_CLASS_NAMES = ['SAFE', 'CAUTION', 'DANGEROUS', 'IGNORE']
    
    def __init__(self, apply_safety_filtering: bool = True):
        """
        Initialize the interpreter.
        
        Args:
            apply_safety_filtering: If True, applies additional safety rules
        """
        self.apply_safety_filtering = apply_safety_filtering
        
        # Create mapping tensor for efficient conversion
        self.semantic_to_safety_tensor = torch.zeros(24, dtype=torch.long)
        for sem_idx, safety in self.SEMANTIC_TO_SAFETY.items():
            self.semantic_to_safety_tensor[sem_idx] = safety.value
    
    def interpret_semantic_predictions(self, semantic_predictions: torch.Tensor) -> torch.Tensor:
        """
        Convert semantic predictions to landing safety map.
        
        Args:
            semantic_predictions: Tensor of shape [H, W] with semantic class indices
            
        Returns:
            safety_map: Tensor of shape [H, W] with safety class indices
        """
        device = semantic_predictions.device
        mapping_tensor = self.semantic_to_safety_tensor.to(device)
        
        # Convert semantic classes to safety classes
        safety_map = mapping_tensor[semantic_predictions]
        
        if self.apply_safety_filtering:
            safety_map = self._apply_safety_filtering(safety_map, semantic_predictions)
        
        return safety_map
    
    def _apply_safety_filtering(self, safety_map: torch.Tensor, 
                               semantic_predictions: torch.Tensor) -> torch.Tensor:
        """
        Apply additional safety rules based on context.
        
        Args:
            safety_map: Initial safety map
            semantic_predictions: Original semantic predictions
            
        Returns:
            filtered_safety_map: Safety map with additional filtering applied
        """
        # Convert IGNORE areas to SAFE if they're surrounded by safe areas
        # This helps with architectural details in otherwise safe zones
        kernel = torch.ones(3, 3, device=safety_map.device)
        
        # Find IGNORE areas
        ignore_mask = (safety_map == LandingSafety.IGNORE.value)
        
        if ignore_mask.sum() > 0:
            # For each IGNORE pixel, check if surrounded by SAFE areas
            safe_mask = (safety_map == LandingSafety.SAFE.value)
            
            # Use morphological operations to check neighborhood
            safe_expanded = F.conv2d(safe_mask.float().unsqueeze(0).unsqueeze(0), 
                                   kernel.unsqueeze(0).unsqueeze(0), 
                                   padding=1).squeeze()
            
            # If an IGNORE area has many SAFE neighbors, convert to CAUTION
            surrounded_by_safe = (safe_expanded >= 6) & ignore_mask
            safety_map[surrounded_by_safe] = LandingSafety.CAUTION.value
        
        return safety_map
    
    def analyze_landing_zone(self, safety_map: torch.Tensor, 
                           min_safe_area: int = 100) -> Dict[str, float]:
        """
        Analyze a landing zone and provide safety assessment.
        
        Args:
            safety_map: Safety map tensor [H, W]
            min_safe_area: Minimum number of pixels for a safe landing zone
            
        Returns:
            analysis: Dictionary with safety metrics
        """
        total_pixels = safety_map.numel()
        
        # Count pixels for each safety class
        safety_counts = {}
        for safety_class in LandingSafety:
            count = (safety_map == safety_class.value).sum().item()
            safety_counts[safety_class.name.lower()] = count
        
        # Calculate percentages
        safety_percentages = {k: (v / total_pixels) * 100 
                            for k, v in safety_counts.items()}
        
        # Find largest safe areas (connected components)
        safe_mask = (safety_map == LandingSafety.SAFE.value)
        largest_safe_area = self._find_largest_connected_component(safe_mask)
        
        # Overall safety assessment
        if largest_safe_area >= min_safe_area and safety_percentages['safe'] > 30:
            overall_safety = "SUITABLE"
        elif safety_percentages['dangerous'] > 50:
            overall_safety = "DANGEROUS" 
        elif safety_percentages['caution'] > 40:
            overall_safety = "CAUTION"
        elif safety_percentages['safe'] > 10:
            overall_safety = "MARGINAL"
        else:
            overall_safety = "UNSUITABLE"
        
        return {
            'overall_safety': overall_safety,
            'largest_safe_area': largest_safe_area,
            'percentages': safety_percentages,
            'counts': safety_counts,
            'recommendation': self._get_landing_recommendation(overall_safety, safety_percentages)
        }
    
    def _find_largest_connected_component(self, binary_mask: torch.Tensor) -> int:
        """Find the largest connected component in a binary mask."""
        # Simple implementation - could use more sophisticated algorithms
        mask_np = binary_mask.cpu().numpy().astype(np.uint8)
        
        try:
            import cv2
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np)
            if num_labels > 1:  # 0 is background
                return stats[1:, cv2.CC_STAT_AREA].max()
            else:
                return 0
        except ImportError:
            # Fallback if OpenCV not available
            return binary_mask.sum().item()
    
    def _get_landing_recommendation(self, overall_safety: str, 
                                  percentages: Dict[str, float]) -> str:
        """Generate landing recommendation based on analysis."""
        if overall_safety == "SUITABLE":
            return "Landing recommended in largest safe area"
        elif overall_safety == "MARGINAL":
            return "Landing possible with caution, avoid dangerous areas"
        elif overall_safety == "CAUTION":
            return "Landing risky, consider alternative site"
        else:
            return "Landing not recommended, find alternative site"
    
    def visualize_safety_map(self, safety_map: torch.Tensor) -> np.ndarray:
        """
        Create a color-coded visualization of the safety map.
        
        Args:
            safety_map: Safety map tensor [H, W]
            
        Returns:
            colored_map: RGB image [H, W, 3] with safety color coding
        """
        # Define colors for each safety level
        colors = {
            LandingSafety.SAFE.value: [0, 255, 0],      # Green
            LandingSafety.CAUTION.value: [255, 255, 0],  # Yellow  
            LandingSafety.DANGEROUS.value: [255, 0, 0],  # Red
            LandingSafety.IGNORE.value: [128, 128, 128]  # Gray
        }
        
        h, w = safety_map.shape
        colored_map = np.zeros((h, w, 3), dtype=np.uint8)
        
        for safety_value, color in colors.items():
            mask = (safety_map.cpu().numpy() == safety_value)
            colored_map[mask] = color
            
        return colored_map

# Example usage and testing
if __name__ == "__main__":
    # Create a sample semantic prediction
    semantic_pred = torch.randint(0, 24, (256, 256))
    
    # Initialize interpreter
    interpreter = LandingSafetyInterpreter()
    
    # Convert to safety map
    safety_map = interpreter.interpret_semantic_predictions(semantic_pred)
    
    # Analyze landing zone
    analysis = interpreter.analyze_landing_zone(safety_map)
    
    print("Landing Zone Analysis:")
    print(f"Overall Safety: {analysis['overall_safety']}")
    print(f"Largest Safe Area: {analysis['largest_safe_area']} pixels")
    print(f"Recommendation: {analysis['recommendation']}")
    print("\nSafety Distribution:")
    for safety_class, percentage in analysis['percentages'].items():
        print(f"{safety_class.capitalize():10}: {percentage:5.1f}%") 