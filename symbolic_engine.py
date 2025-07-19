# symbolic_engine.py
"""
Symbolic Engine for UAV Landing Zone Detection
Implements rule-based reasoning for safe landing zone identification.
"""

import cv2
import numpy as np
import config
from collections import deque
from typing import List, Dict, Tuple, Any
import time
import math


class LandingZone:
    """Represents a potential landing zone with its properties."""
    
    def __init__(self, zone_id: int, contour: np.ndarray, area: float, center: Tuple[int, int]):
        self.id = zone_id
        self.contour = contour
        self.area = area
        self.center = center
        self.score = 0.0
        self.clearance_score = 0.0
        self.shape_score = 0.0
        self.center_score = 0.0
        self.temporal_stability = 0
        
        # Compute additional geometric properties
        self._compute_properties()
    
    def _compute_properties(self):
        """Compute geometric properties of the zone."""
        # Bounding rectangle
        self.bounding_rect = cv2.boundingRect(self.contour)
        x, y, w, h = self.bounding_rect
        
        # Aspect ratio
        self.aspect_ratio = w / h if h > 0 else 0
        
        # Solidity (area / convex hull area)
        hull = cv2.convexHull(self.contour)
        hull_area = cv2.contourArea(hull)
        self.solidity = self.area / hull_area if hull_area > 0 else 0
        
        # Distance from image center
        img_center = (config.INPUT_RESOLUTION[0] // 2, config.INPUT_RESOLUTION[1] // 2)
        self.distance_from_center = math.sqrt(
            (self.center[0] - img_center[0])**2 + (self.center[1] - img_center[1])**2
        )


class SymbolicEngine:
    """
    Symbolic reasoning engine for landing zone detection.
    Applies rule-based logic to make safe landing decisions.
    """
    
    def __init__(self):
        """Initialize the symbolic engine."""
        # Tracks valid zones over time for temporal stability
        self.history = deque(maxlen=config.TEMPORAL_WINDOW_SIZE)
        self.zone_stability_tracker = {}  # zone_id -> count
        self.frame_count = 0
        
        # Performance tracking
        self.processing_times = []
    
    def run(self, seg_map: np.ndarray) -> Dict[str, Any]:
        """
        Main processing function that analyzes segmentation and returns decision.
        
        Args:
            seg_map: Segmentation map from neural network (H, W) with class IDs
            
        Returns:
            Dictionary containing the landing decision and associated data
        """
        start_time = time.time()
        self.frame_count += 1
        
        try:
            # 1. Clean up the segmentation map
            cleaned_seg_map = self._clean_segmentation(seg_map)
            
            # 2. Find all potential landing zones
            potential_zones = self._find_potential_zones(cleaned_seg_map)
            
            # 3. Find all obstacles
            obstacles = self._find_obstacles(cleaned_seg_map)
            
            # 4. Apply geometric and spatial rules to validate zones
            valid_zones = []
            for zone in potential_zones:
                if self._is_zone_geometrically_valid(zone) and self._is_zone_spatially_safe(zone, obstacles):
                    valid_zones.append(zone)
            
            # 5. Update temporal tracking and get confirmed zones
            confirmed_zones = self._update_temporal_tracking(valid_zones)
            
            # 6. Score and rank confirmed zones
            if not confirmed_zones:
                result = {
                    'status': 'NO_VALID_ZONE',
                    'reason': 'No confirmed safe zones found',
                    'frame_count': self.frame_count,
                    'potential_zones': len(potential_zones),
                    'valid_zones': len(valid_zones),
                    'obstacles': len(obstacles)
                }
            else:
                # Score all confirmed zones
                for zone in confirmed_zones:
                    zone.score = self._calculate_zone_score(zone)
                
                # Select the best zone
                best_zone = max(confirmed_zones, key=lambda z: z.score)
                
                result = {
                    'status': 'TARGET_ACQUIRED',
                    'zone': self._zone_to_dict(best_zone),
                    'alternatives': [self._zone_to_dict(z) for z in sorted(confirmed_zones, key=lambda z: z.score, reverse=True)[1:6]],  # Top 5 alternatives
                    'frame_count': self.frame_count,
                    'potential_zones': len(potential_zones),
                    'valid_zones': len(valid_zones),
                    'confirmed_zones': len(confirmed_zones),
                    'obstacles': len(obstacles)
                }
            
            # Track performance
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > config.PERFORMANCE_WINDOW:
                self.processing_times = self.processing_times[-config.PERFORMANCE_WINDOW:]
            
            result['processing_time'] = processing_time
            return result
            
        except Exception as e:
            return {
                'status': 'ERROR',
                'reason': f'Processing error: {str(e)}',
                'frame_count': self.frame_count
            }
    
    def _clean_segmentation(self, seg_map: np.ndarray) -> np.ndarray:
        """Clean up the segmentation map using morphological operations."""
        cleaned = seg_map.copy()
        
        # Apply Gaussian blur to reduce noise
        if config.BLUR_KERNEL_SIZE > 0:
            cleaned = cv2.GaussianBlur(cleaned, (config.BLUR_KERNEL_SIZE, config.BLUR_KERNEL_SIZE), 0)
        
        # Morphological operations for each class separately
        kernel = np.ones((config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE), np.uint8)
        
        for class_id in config.CLASS_MAP.keys():
            if class_id == 0:  # Skip background
                continue
            
            # Extract mask for this class
            mask = (cleaned == class_id).astype(np.uint8)
            
            # Apply morphological operations
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=config.MORPH_ITERATIONS)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Update the cleaned segmentation
            cleaned[mask == 1] = class_id
            cleaned[mask == 0] = 0 if (cleaned == class_id).any() else cleaned[mask == 0]
        
        return cleaned
    
    def _find_potential_zones(self, seg_map: np.ndarray) -> List[LandingZone]:
        """Find all potential landing zones from safe surface segments."""
        # Create mask for safe landing surfaces
        mask = (seg_map == config.SAFE_LANDING_CLASS_ID).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        zones = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Filter by minimum area
            if area >= config.MIN_LANDING_AREA_PIXELS:
                # Calculate center of mass
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    zone = LandingZone(i, contour, area, (cx, cy))
                    zones.append(zone)
        
        return zones
    
    def _find_obstacles(self, seg_map: np.ndarray) -> List[Dict[str, Any]]:
        """Find all obstacles in the segmentation map."""
        obstacles = []
        
        # Find high obstacles
        high_mask = (seg_map == config.HIGH_OBSTACLE_CLASS_ID).astype(np.uint8)
        high_contours, _ = cv2.findContours(high_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in high_contours:
            if cv2.contourArea(contour) > 10:  # Minimum obstacle size
                obstacles.append({
                    'type': 'high',
                    'contour': contour,
                    'clearance_required': config.HIGH_OBSTACLE_CLEARANCE
                })
        
        # Find low obstacles
        low_mask = (seg_map == config.LOW_OBSTACLE_CLASS_ID).astype(np.uint8)
        low_contours, _ = cv2.findContours(low_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in low_contours:
            if cv2.contourArea(contour) > 5:  # Minimum obstacle size
                obstacles.append({
                    'type': 'low', 
                    'contour': contour,
                    'clearance_required': config.LOW_OBSTACLE_CLEARANCE
                })
        
        return obstacles
    
    def _is_zone_geometrically_valid(self, zone: LandingZone) -> bool:
        """Check if a zone meets geometric validity criteria."""
        # Aspect ratio check
        if not (config.MIN_ASPECT_RATIO <= zone.aspect_ratio <= config.MAX_ASPECT_RATIO):
            return False
        
        # Solidity check (how filled the shape is)
        if zone.solidity < config.MIN_SOLIDITY:
            return False
        
        return True
    
    def _is_zone_spatially_safe(self, zone: LandingZone, obstacles: List[Dict[str, Any]]) -> bool:
        """Check if a zone maintains safe clearance from obstacles."""
        min_clearance = float('inf')
        
        for obstacle in obstacles:
            # Calculate minimum distance between zone contour and obstacle contour
            dist = cv2.pointPolygonTest(obstacle['contour'], zone.center, True)
            abs_dist = abs(dist)
            
            # Check if we're too close to this obstacle
            if abs_dist < obstacle['clearance_required']:
                return False
            
            min_clearance = min(min_clearance, abs_dist)
        
        # Store the clearance score for later use
        zone.clearance_score = min_clearance if min_clearance != float('inf') else 100
        
        return True
    
    def _update_temporal_tracking(self, valid_zones: List[LandingZone]) -> List[LandingZone]:
        """Update temporal stability tracking and return confirmed zones."""
        # Create current frame zone IDs (based on position similarity)
        current_zone_ids = set()
        
        # Simple position-based tracking (could be improved with more sophisticated tracking)
        for zone in valid_zones:
            # Find if this zone is similar to any previous zones
            zone_key = self._get_zone_key(zone)
            current_zone_ids.add(zone_key)
            
            # Update stability counter
            if zone_key in self.zone_stability_tracker:
                self.zone_stability_tracker[zone_key] += 1
            else:
                self.zone_stability_tracker[zone_key] = 1
        
        # Add current frame to history
        self.history.append(current_zone_ids)
        
        # Clean up old tracking data
        all_recent_zones = set()
        for frame_zones in self.history:
            all_recent_zones.update(frame_zones)
        
        # Remove zones that haven't appeared recently
        zones_to_remove = set(self.zone_stability_tracker.keys()) - all_recent_zones
        for zone_key in zones_to_remove:
            del self.zone_stability_tracker[zone_key]
        
        # Filter zones by temporal stability
        confirmed_zones = []
        for zone in valid_zones:
            zone_key = self._get_zone_key(zone)
            stability_count = self.zone_stability_tracker.get(zone_key, 0)
            
            if stability_count >= config.TEMPORAL_CONFIRMATION_THRESHOLD:
                zone.temporal_stability = stability_count
                confirmed_zones.append(zone)
        
        return confirmed_zones
    
    def _get_zone_key(self, zone: LandingZone) -> str:
        """Generate a key for zone tracking based on position and size."""
        # Discretize position and size for tracking
        x_bin = zone.center[0] // 20  # 20-pixel bins
        y_bin = zone.center[1] // 20
        area_bin = int(zone.area // 1000)  # 1000-pixel area bins
        
        return f"{x_bin}_{y_bin}_{area_bin}"
    
    def _calculate_zone_score(self, zone: LandingZone) -> float:
        """Calculate comprehensive score for a landing zone."""
        # Normalize area score (0-1)
        max_possible_area = config.INPUT_RESOLUTION[0] * config.INPUT_RESOLUTION[1]
        area_score = min(zone.area / max_possible_area, 1.0)
        
        # Normalize center score (0-1, higher for zones closer to center)
        max_distance = math.sqrt(config.INPUT_RESOLUTION[0]**2 + config.INPUT_RESOLUTION[1]**2) / 2
        center_score = 1.0 - (zone.distance_from_center / max_distance)
        
        # Shape score (0-1, based on solidity and aspect ratio)
        aspect_score = 1.0 - abs(zone.aspect_ratio - 1.0) / max(zone.aspect_ratio, 1.0/zone.aspect_ratio)
        shape_score = (zone.solidity + aspect_score) / 2.0
        
        # Clearance score (0-1)
        clearance_score = min(zone.clearance_score / 100.0, 1.0)
        
        # Store individual scores for debugging
        zone.area_score = area_score
        zone.center_score = center_score
        zone.shape_score = shape_score
        zone.clearance_score = clearance_score
        
        # Weighted final score
        final_score = (
            config.W_AREA * area_score +
            config.W_CENTER * center_score +
            config.W_SHAPE * shape_score +
            config.W_CLEARANCE * clearance_score
        )
        
        return final_score
    
    def _zone_to_dict(self, zone: LandingZone) -> Dict[str, Any]:
        """Convert LandingZone object to dictionary for output."""
        return {
            'id': zone.id,
            'center': zone.center,
            'area': zone.area,
            'score': zone.score,
            'temporal_stability': zone.temporal_stability,
            'bounding_rect': zone.bounding_rect,
            'aspect_ratio': zone.aspect_ratio,
            'solidity': zone.solidity,
            'distance_from_center': zone.distance_from_center,
            'individual_scores': {
                'area': getattr(zone, 'area_score', 0),
                'center': getattr(zone, 'center_score', 0),
                'shape': getattr(zone, 'shape_score', 0),
                'clearance': getattr(zone, 'clearance_score', 0)
            }
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the symbolic engine."""
        if not self.processing_times:
            return {"avg_processing_time": 0, "fps": 0, "samples": 0}
        
        avg_time = np.mean(self.processing_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            "avg_processing_time": avg_time,
            "fps": fps,
            "samples": len(self.processing_times),
            "min_time": np.min(self.processing_times),
            "max_time": np.max(self.processing_times),
            "tracked_zones": len(self.zone_stability_tracker)
        }
    
    def reset_performance_stats(self):
        """Reset performance tracking statistics."""
        self.processing_times = []
    
    def reset_temporal_tracking(self):
        """Reset temporal stability tracking."""
        self.history.clear()
        self.zone_stability_tracker.clear()
        self.frame_count = 0
