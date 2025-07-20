#!/usr/bin/env python3
"""
Neurosymbolic Memory System for UAV Landing
Clean, production-ready implementation
"""

import numpy as np
import cv2
import math
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from collections import deque

from .types import MemoryZone, SpatialMemoryState, TemporalMemoryState, SemanticMemoryState


class NeuroSymbolicMemory:
    """
    Production-ready neurosymbolic memory system for UAV landing.
    
    Maintains spatial, temporal, and semantic memory for robust landing decisions
    even when visual context is temporarily lost.
    """
    
    def __init__(self, 
                 memory_horizon: float = 300.0,
                 spatial_resolution: float = 0.5,
                 confidence_decay_rate: float = 0.98,
                 min_observations: int = 3,
                 grid_size: int = 100):
        """
        Initialize memory system.
        
        Args:
            memory_horizon: How long to retain memory (seconds)
            spatial_resolution: Memory grid resolution (meters per cell) 
            confidence_decay_rate: How fast confidence decays per frame
            min_observations: Minimum observations to trust a zone
            grid_size: Size of confidence grid (grid_size x grid_size)
        """
        
        self.memory_horizon = memory_horizon
        self.spatial_resolution = spatial_resolution
        self.confidence_decay_rate = confidence_decay_rate
        self.min_observations = min_observations
        self.grid_size = grid_size
        
        # Initialize memory states
        self.spatial_memory = SpatialMemoryState(grid_size=grid_size, grid_resolution=spatial_resolution)
        self.temporal_memory = TemporalMemoryState()
        self.semantic_memory = SemanticMemoryState()
        
        # Current state tracking
        self.current_position = np.array([0.0, 0.0])
        self.current_altitude = 0.0
        self.current_heading = 0.0
        self.last_update_time = time.time()
        self.frame_count = 0
        
        # Initialize spatial grid
        self._initialize_spatial_grid()
        
    def _initialize_spatial_grid(self):
        """Initialize the probabilistic spatial memory grid"""
        self.spatial_memory.confidence_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
    
    def update_drone_state(self, world_position: Tuple[float, float], altitude: float, heading: float = 0.0):
        """Update current drone state for spatial reasoning"""
        
        if self.spatial_memory.reference_position is None:
            self.spatial_memory.reference_position = world_position
            self.current_position = np.array([0.0, 0.0])
        else:
            ref_x, ref_y = self.spatial_memory.reference_position
            self.current_position = np.array([
                world_position[0] - ref_x,
                world_position[1] - ref_y
            ])
        
        self.current_altitude = altitude
        self.current_heading = heading
        
        # Add to trajectory (keep last 100 positions)
        trajectory_entry = {
            'position': self.current_position.copy(),
            'altitude': altitude,
            'timestamp': time.time(),
            'heading': heading
        }
        
        self.spatial_memory.drone_trajectory.append(trajectory_entry)
        if len(self.spatial_memory.drone_trajectory) > 100:
            self.spatial_memory.drone_trajectory.pop(0)
    
    def observe_zones(self, zones: List[Dict], world_positions: List[Tuple[float, float]], 
                     confidences: List[float], environment_context: Dict = None):
        """Process observed landing zones and update memory"""
        
        current_time = time.time()
        
        # Update each observed zone
        for zone, world_pos, confidence in zip(zones, world_positions, confidences):
            self._update_spatial_memory(zone, world_pos, confidence, current_time)
            self._update_temporal_memory(zone, confidence, current_time)
        
        # Update semantic memory with context
        if environment_context:
            self._update_semantic_memory(environment_context, zones, confidences)
        
        # Decay confidence in unobserved areas
        self._decay_memory_confidence()
        
        # Update confidence grid
        self._update_confidence_grid()
        
        self.frame_count += 1
        self.last_update_time = current_time
    
    def _update_spatial_memory(self, zone: Dict, world_pos: Tuple[float, float], 
                              confidence: float, timestamp: float):
        """Update spatial memory with new zone observation"""
        
        if self.spatial_memory.reference_position is None:
            return
        
        # Convert to relative coordinates
        ref_x, ref_y = self.spatial_memory.reference_position
        rel_pos = (world_pos[0] - ref_x, world_pos[1] - ref_y)
        
        # Find existing zone or create new one
        existing_zone = self._find_matching_zone(rel_pos)
        
        if existing_zone:
            self._update_existing_zone(existing_zone, rel_pos, confidence, timestamp)
        else:
            new_zone = MemoryZone(
                world_position=rel_pos,
                estimated_size=math.sqrt(zone.get('area', 1000)) * 0.1,
                first_seen=timestamp,
                last_seen=timestamp,
                observation_count=1,
                max_confidence=confidence,
                avg_confidence=confidence,
                spatial_stability=1.0,
                position_uncertainty=0.5
            )
            self.spatial_memory.memory_zones.append(new_zone)
    
    def _find_matching_zone(self, position: Tuple[float, float], max_distance: float = 2.0) -> Optional[MemoryZone]:
        """Find existing memory zone near the given position"""
        
        for zone in self.spatial_memory.memory_zones:
            distance = math.sqrt(
                (zone.world_position[0] - position[0])**2 + 
                (zone.world_position[1] - position[1])**2
            )
            if distance <= max_distance:
                return zone
        return None
    
    def _update_existing_zone(self, zone: MemoryZone, new_position: Tuple[float, float], 
                            confidence: float, timestamp: float):
        """Update an existing memory zone with new observation"""
        
        # Update position with weighted average
        weight = min(confidence, 0.3)
        zone.world_position = (
            zone.world_position[0] * (1 - weight) + new_position[0] * weight,
            zone.world_position[1] * (1 - weight) + new_position[1] * weight
        )
        
        # Update metrics
        zone.last_seen = timestamp
        zone.observation_count += 1
        zone.max_confidence = max(zone.max_confidence, confidence)
        zone.avg_confidence = (zone.avg_confidence * (zone.observation_count - 1) + confidence) / zone.observation_count
        
        # Update spatial stability
        position_error = math.sqrt(
            (zone.world_position[0] - new_position[0])**2 + 
            (zone.world_position[1] - new_position[1])**2
        )
        stability_update = max(0.1, 1.0 - position_error / 2.0)
        zone.spatial_stability = zone.spatial_stability * 0.9 + stability_update * 0.1
        zone.position_uncertainty = max(0.1, zone.position_uncertainty * 0.95 + position_error * 0.05)
    
    def _update_temporal_memory(self, zone: Dict, confidence: float, timestamp: float):
        """Update temporal patterns and sequences"""
        
        detection_info = {
            'timestamp': timestamp,
            'confidence': confidence,
            'zone_center': zone.get('center', (0, 0)),
            'zone_area': zone.get('area', 0)
        }
        
        self.temporal_memory.recent_detections.append(detection_info)
        
        # Keep only recent detections
        if len(self.temporal_memory.recent_detections) > 30:
            self.temporal_memory.recent_detections.pop(0)
        
        # Analyze patterns
        if len(self.temporal_memory.recent_detections) >= 5:
            self._analyze_temporal_patterns()
    
    def _analyze_temporal_patterns(self):
        """Analyze temporal patterns in detections"""
        
        recent = self.temporal_memory.recent_detections[-10:]
        
        # Confidence stability pattern
        confidences = [d['confidence'] for d in recent]
        if len(confidences) >= 5:
            confidence_stability = 1.0 - np.std(confidences)
            self.temporal_memory.detection_patterns['confidence_stability'] = confidence_stability
        
        # Spatial convergence pattern  
        if len(recent) >= 5:
            positions = [d['zone_center'] for d in recent]
            distances = []
            for i in range(1, len(positions)):
                dist = math.sqrt(
                    (positions[i][0] - positions[i-1][0])**2 + 
                    (positions[i][1] - positions[i-1][1])**2
                )
                distances.append(dist)
            
            if distances:
                convergence = 1.0 / (1.0 + np.mean(distances) / 10.0)
                self.temporal_memory.detection_patterns['spatial_convergence'] = convergence
    
    def _update_semantic_memory(self, context: Dict, zones: List[Dict], confidences: List[float]):
        """Update high-level semantic understanding"""
        
        env_type = context.get('environment_type', 'unknown')
        if env_type != 'unknown':
            if env_type not in self.semantic_memory.environment_templates:
                self.semantic_memory.environment_templates[env_type] = {
                    'typical_zone_count': 0,
                    'typical_confidences': [],
                    'success_rate': 0.5
                }
            
            template = self.semantic_memory.environment_templates[env_type]
            template['typical_zone_count'] = (template['typical_zone_count'] * 0.9 + len(zones) * 0.1)
            template['typical_confidences'].extend(confidences)
            
            # Keep only recent confidences
            if len(template['typical_confidences']) > 100:
                template['typical_confidences'] = template['typical_confidences'][-50:]
    
    def _decay_memory_confidence(self):
        """Decay confidence in memory zones not recently observed"""
        
        current_time = time.time()
        
        for zone in self.spatial_memory.memory_zones:
            time_since_seen = current_time - zone.last_seen
            decay_factor = self.confidence_decay_rate ** (time_since_seen / 1.0)
            zone.avg_confidence *= decay_factor
            zone.position_uncertainty = min(5.0, zone.position_uncertainty * 1.01)
    
    def _update_confidence_grid(self):
        """Update the probabilistic confidence grid"""
        
        if self.spatial_memory.confidence_grid is None:
            return
        
        self.spatial_memory.confidence_grid.fill(0.0)
        
        for zone in self.spatial_memory.memory_zones:
            if zone.observation_count >= self.min_observations:
                grid_x = int((zone.world_position[0] + self.grid_size * self.spatial_resolution / 2) / self.spatial_resolution)
                grid_y = int((zone.world_position[1] + self.grid_size * self.spatial_resolution / 2) / self.spatial_resolution)
                
                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    sigma = zone.position_uncertainty / self.spatial_resolution
                    self._add_gaussian_to_grid(grid_x, grid_y, zone.avg_confidence, sigma)
    
    def _add_gaussian_to_grid(self, center_x: int, center_y: int, confidence: float, sigma: float):
        """Add a Gaussian confidence blob to the grid"""
        
        grid = self.spatial_memory.confidence_grid
        kernel_size = min(int(3 * sigma), self.grid_size // 4)
        if kernel_size < 1:
            kernel_size = 1
        
        for dy in range(-kernel_size, kernel_size + 1):
            for dx in range(-kernel_size, kernel_size + 1):
                gx, gy = center_x + dx, center_y + dy
                
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    distance_sq = dx**2 + dy**2
                    gaussian_value = confidence * math.exp(-distance_sq / (2 * sigma**2))
                    grid[gy, gx] = max(grid[gy, gx], gaussian_value)
    
    def predict_zones_from_memory(self, min_confidence: float = 0.3, max_zones: int = 5) -> List[Dict]:
        """Predict likely landing zones based on memory when visual input is poor"""
        
        predicted_zones = []
        current_time = time.time()
        
        for zone in self.spatial_memory.memory_zones:
            if (zone.avg_confidence >= min_confidence and 
                zone.observation_count >= self.min_observations):
                
                time_since_seen = current_time - zone.last_seen
                current_confidence = zone.avg_confidence * (self.confidence_decay_rate ** time_since_seen)
                
                if current_confidence >= min_confidence:
                    if self.spatial_memory.reference_position:
                        ref_x, ref_y = self.spatial_memory.reference_position
                        world_x = zone.world_position[0] + ref_x
                        world_y = zone.world_position[1] + ref_y
                        
                        # Estimate pixel position and size
                        pixel_center = self._world_to_pixel(zone.world_position)
                        pixel_size = int(zone.estimated_size * 10)  # Convert meters to pixels
                        
                        predicted_zones.append({
                            'center': pixel_center,
                            'world_position': (world_x, world_y),
                            'confidence': current_confidence,
                            'area': (zone.estimated_size * 10)**2,
                            'bbox': (pixel_center[0] - pixel_size//2, 
                                   pixel_center[1] - pixel_size//2, 
                                   pixel_size, pixel_size),
                            'aspect_ratio': 1.0,  # Assume square zones from memory
                            'solidity': 0.9,      # High solidity for memory zones
                            'source': 'memory',
                            'uncertainty': zone.position_uncertainty,
                            'last_seen': zone.last_seen,
                            'observation_count': zone.observation_count
                        })
        
        predicted_zones.sort(key=lambda x: x['confidence'], reverse=True)
        return predicted_zones[:max_zones]
    
    def _world_to_pixel(self, relative_world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert relative world coordinates to approximate pixel coordinates"""
        scale = 50.0  # Assume 1 meter = 50 pixels at typical altitude
        center_x, center_y = 320, 240  # Assume 640x480 image
        
        pixel_x = int(center_x + relative_world_pos[0] * scale)
        pixel_y = int(center_y + relative_world_pos[1] * scale)
        
        return (pixel_x, pixel_y)
    
    def get_memory_confidence(self, position: Tuple[float, float]) -> float:
        """Get memory-based confidence for a given world position"""
        
        if self.spatial_memory.reference_position is None:
            return 0.0
        
        ref_x, ref_y = self.spatial_memory.reference_position
        rel_pos = (position[0] - ref_x, position[1] - ref_y)
        
        grid = self.spatial_memory.confidence_grid
        if grid is not None:
            grid_x = int((rel_pos[0] + self.grid_size * self.spatial_resolution / 2) / self.spatial_resolution)
            grid_y = int((rel_pos[1] + self.grid_size * self.spatial_resolution / 2) / self.spatial_resolution)
            
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                return float(grid[grid_y, grid_x])
        
        return 0.0
    
    def get_active_zones(self) -> List[MemoryZone]:
        """Get currently active memory zones"""
        current_time = time.time()
        return [z for z in self.spatial_memory.memory_zones 
                if (current_time - z.last_seen) < 60.0]
    
    def get_memory_status(self) -> Dict:
        """Get current memory system status for debugging"""
        
        current_time = time.time()
        active_zones = [z for z in self.spatial_memory.memory_zones 
                       if (current_time - z.last_seen) < 60.0]
        
        return {
            'total_memory_zones': len(self.spatial_memory.memory_zones),
            'active_memory_zones': len(active_zones),
            'avg_zone_confidence': np.mean([z.avg_confidence for z in active_zones]) if active_zones else 0.0,
            'temporal_patterns': dict(self.temporal_memory.detection_patterns),
            'frame_count': self.frame_count,
            'memory_age': current_time - (self.spatial_memory.drone_trajectory[0]['timestamp'] 
                                        if self.spatial_memory.drone_trajectory else current_time),
            'grid_coverage': np.mean(self.spatial_memory.confidence_grid > 0.1) if self.spatial_memory.confidence_grid is not None else 0.0
        }
    
    def save_memory(self, filepath: str):
        """Save memory state to file for persistence across flights"""
        
        memory_data = {
            'spatial_memory': {
                'reference_position': self.spatial_memory.reference_position,
                'memory_zones': [
                    {
                        'world_position': zone.world_position,
                        'estimated_size': zone.estimated_size,
                        'first_seen': zone.first_seen,
                        'last_seen': zone.last_seen,
                        'observation_count': zone.observation_count,
                        'max_confidence': zone.max_confidence,
                        'avg_confidence': zone.avg_confidence,
                        'spatial_stability': zone.spatial_stability,
                        'environment_type': zone.environment_type,
                        'nearby_features': zone.nearby_features,
                        'position_uncertainty': zone.position_uncertainty
                    }
                    for zone in self.spatial_memory.memory_zones
                ]
            },
            'semantic_memory': {
                'environment_templates': self.semantic_memory.environment_templates,
                'learned_associations': self.semantic_memory.learned_associations
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(memory_data, f, indent=2)
    
    def load_memory(self, filepath: str):
        """Load memory state from file"""
        
        if not Path(filepath).exists():
            return
        
        try:
            with open(filepath, 'r') as f:
                memory_data = json.load(f)
            
            # Restore spatial memory
            if 'spatial_memory' in memory_data:
                spatial_data = memory_data['spatial_memory']
                self.spatial_memory.reference_position = spatial_data.get('reference_position')
                
                self.spatial_memory.memory_zones = []
                for zone_data in spatial_data.get('memory_zones', []):
                    zone = MemoryZone(
                        world_position=tuple(zone_data['world_position']),
                        estimated_size=zone_data['estimated_size'],
                        first_seen=zone_data['first_seen'],
                        last_seen=zone_data['last_seen'],
                        observation_count=zone_data['observation_count'],
                        max_confidence=zone_data['max_confidence'],
                        avg_confidence=zone_data['avg_confidence'],
                        spatial_stability=zone_data['spatial_stability'],
                        environment_type=zone_data.get('environment_type', 'unknown'),
                        nearby_features=zone_data.get('nearby_features', []),
                        position_uncertainty=zone_data.get('position_uncertainty', 1.0)
                    )
                    self.spatial_memory.memory_zones.append(zone)
            
            # Restore semantic memory
            if 'semantic_memory' in memory_data:
                semantic_data = memory_data['semantic_memory']
                self.semantic_memory.environment_templates = semantic_data.get('environment_templates', {})
                self.semantic_memory.learned_associations = semantic_data.get('learned_associations', {})
                
        except Exception as e:
            print(f"Error loading memory: {e}")
    
    def reset_memory(self):
        """Reset all memory"""
        self.spatial_memory = SpatialMemoryState(grid_size=self.grid_size, grid_resolution=self.spatial_resolution)
        self.temporal_memory = TemporalMemoryState()
        self.semantic_memory = SemanticMemoryState()
        self._initialize_spatial_grid()
        self.frame_count = 0
