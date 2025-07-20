#!/usr/bin/env python3
"""
Neurosymbolic Memory System for UAV Landing
Maintains spatial, temporal, and semantic memory for robust landing decisions
"""

import numpy as np
import cv2
import math
import time
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from collections import deque
import json
from pathlib import Path


@dataclass
class MemoryZone:
    """A landing zone stored in memory with spatial and temporal information"""
    # Spatial properties (world coordinates relative to first detection)
    world_position: Tuple[float, float]  # (x, y) in meters from reference point
    estimated_size: float  # Estimated zone radius in meters
    
    # Temporal properties
    first_seen: float  # Timestamp when first detected
    last_seen: float   # Timestamp when last confirmed
    observation_count: int  # Number of times observed
    
    # Quality metrics
    max_confidence: float  # Best confidence score ever achieved
    avg_confidence: float  # Average confidence over all observations
    spatial_stability: float  # How consistent the position has been (0-1)
    
    # Contextual information
    environment_type: str = "unknown"  # grass, concrete, mixed, etc.
    nearby_features: List[str] = field(default_factory=list)  # landmarks, obstacles
    
    # Uncertainty tracking
    position_uncertainty: float = 1.0  # Standard deviation in meters
    confidence_decay_rate: float = 0.95  # How fast confidence decays without observation


@dataclass
class SpatialMemoryState:
    """Current state of spatial memory"""
    reference_position: Optional[Tuple[float, float]] = None  # World reference point
    drone_trajectory: deque = field(default_factory=lambda: deque(maxlen=100))  # Recent positions
    memory_zones: List[MemoryZone] = field(default_factory=list)
    confidence_grid: Optional[np.ndarray] = None  # Probabilistic occupancy grid
    grid_resolution: float = 0.5  # meters per grid cell
    grid_size: int = 100  # 50x50 meter area around drone


@dataclass  
class TemporalMemoryState:
    """Temporal patterns and sequences"""
    recent_detections: deque = field(default_factory=lambda: deque(maxlen=30))
    detection_patterns: Dict[str, float] = field(default_factory=dict)  # Pattern -> confidence
    phase_transitions: List[Tuple[str, str, float]] = field(default_factory=list)  # (from, to, time)


@dataclass
class SemanticMemoryState:
    """High-level environmental understanding"""
    environment_templates: Dict[str, Dict] = field(default_factory=dict)
    learned_associations: Dict[str, List[str]] = field(default_factory=dict)
    success_patterns: List[Dict] = field(default_factory=list)
    failure_patterns: List[Dict] = field(default_factory=list)


class NeuroSymbolicMemory:
    """
    Advanced memory system for neurosymbolic UAV landing.
    
    Maintains multiple memory types:
    - Spatial: Where landing zones are in world coordinates
    - Temporal: Patterns and sequences over time  
    - Semantic: High-level environmental understanding
    """
    
    def __init__(self, 
                 memory_horizon: float = 300.0,  # seconds
                 spatial_resolution: float = 0.5,  # meters per grid cell
                 confidence_decay_rate: float = 0.98,  # per frame decay
                 min_observations: int = 3):  # minimum observations to trust a zone
        
        self.memory_horizon = memory_horizon
        self.spatial_resolution = spatial_resolution
        self.confidence_decay_rate = confidence_decay_rate
        self.min_observations = min_observations
        
        # Memory states
        self.spatial_memory = SpatialMemoryState()
        self.temporal_memory = TemporalMemoryState()
        self.semantic_memory = SemanticMemoryState()
        
        # Current drone state
        self.current_position = np.array([0.0, 0.0])  # Relative to reference
        self.current_altitude = 0.0
        self.current_heading = 0.0
        
        # Memory management
        self.last_update_time = time.time()
        self.frame_count = 0
        
        # Initialize confidence grid
        self._initialize_spatial_grid()
        
        print("🧠 NeuroSymbolic Memory System initialized")
    
    def _initialize_spatial_grid(self):
        """Initialize the probabilistic spatial memory grid"""
        grid_size = self.spatial_memory.grid_size
        self.spatial_memory.confidence_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    def update_drone_state(self, 
                          world_position: Tuple[float, float], 
                          altitude: float,
                          heading: float = 0.0):
        """Update current drone state for spatial reasoning"""
        
        if self.spatial_memory.reference_position is None:
            # Set first position as reference
            self.spatial_memory.reference_position = world_position
            self.current_position = np.array([0.0, 0.0])
        else:
            # Calculate relative position from reference
            ref_x, ref_y = self.spatial_memory.reference_position
            self.current_position = np.array([
                world_position[0] - ref_x,
                world_position[1] - ref_y
            ])
        
        self.current_altitude = altitude
        self.current_heading = heading
        
        # Add to trajectory
        self.spatial_memory.drone_trajectory.append({
            'position': self.current_position.copy(),
            'altitude': altitude,
            'timestamp': time.time(),
            'heading': heading
        })
    
    def observe_zones(self, 
                     zones: List[Dict], 
                     world_positions: List[Tuple[float, float]],
                     confidences: List[float],
                     environment_context: Dict = None):
        """
        Process observed landing zones and update memory
        
        Args:
            zones: List of detected zone dictionaries
            world_positions: World coordinates of each zone
            confidences: Confidence scores for each zone
            environment_context: Additional context (lighting, terrain type, etc.)
        """
        
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
        
        # Update grid
        self._update_confidence_grid()
        
        self.frame_count += 1
        self.last_update_time = current_time
    
    def _update_spatial_memory(self, 
                              zone: Dict, 
                              world_pos: Tuple[float, float], 
                              confidence: float, 
                              timestamp: float):
        """Update spatial memory with new zone observation"""
        
        # Convert to relative coordinates
        if self.spatial_memory.reference_position is None:
            return
        
        ref_x, ref_y = self.spatial_memory.reference_position
        rel_pos = (world_pos[0] - ref_x, world_pos[1] - ref_y)
        
        # Find existing zone or create new one
        existing_zone = self._find_matching_memory_zone(rel_pos)
        
        if existing_zone:
            # Update existing zone
            self._update_existing_zone(existing_zone, rel_pos, confidence, timestamp)
        else:
            # Create new memory zone
            new_zone = MemoryZone(
                world_position=rel_pos,
                estimated_size=math.sqrt(zone.get('area', 1000)) * 0.1,  # Rough size estimate
                first_seen=timestamp,
                last_seen=timestamp,
                observation_count=1,
                max_confidence=confidence,
                avg_confidence=confidence,
                spatial_stability=1.0,
                position_uncertainty=0.5
            )
            self.spatial_memory.memory_zones.append(new_zone)
    
    def _find_matching_memory_zone(self, position: Tuple[float, float], 
                                  max_distance: float = 2.0) -> Optional[MemoryZone]:
        """Find existing memory zone near the given position"""
        
        for zone in self.spatial_memory.memory_zones:
            distance = math.sqrt(
                (zone.world_position[0] - position[0])**2 + 
                (zone.world_position[1] - position[1])**2
            )
            
            if distance <= max_distance:
                return zone
        
        return None
    
    def _update_existing_zone(self, 
                            zone: MemoryZone, 
                            new_position: Tuple[float, float], 
                            confidence: float, 
                            timestamp: float):
        """Update an existing memory zone with new observation"""
        
        # Update position with weighted average
        weight = min(confidence, 0.3)  # Limit influence of single observation
        zone.world_position = (
            zone.world_position[0] * (1 - weight) + new_position[0] * weight,
            zone.world_position[1] * (1 - weight) + new_position[1] * weight
        )
        
        # Update temporal info
        zone.last_seen = timestamp
        zone.observation_count += 1
        
        # Update confidence metrics
        zone.max_confidence = max(zone.max_confidence, confidence)
        zone.avg_confidence = (zone.avg_confidence * (zone.observation_count - 1) + confidence) / zone.observation_count
        
        # Update spatial stability
        position_error = math.sqrt(
            (zone.world_position[0] - new_position[0])**2 + 
            (zone.world_position[1] - new_position[1])**2
        )
        stability_update = max(0.1, 1.0 - position_error / 2.0)
        zone.spatial_stability = zone.spatial_stability * 0.9 + stability_update * 0.1
        
        # Update uncertainty
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
        
        # Analyze patterns (simplified version)
        if len(self.temporal_memory.recent_detections) >= 5:
            self._analyze_temporal_patterns()
    
    def _analyze_temporal_patterns(self):
        """Analyze temporal patterns in detections"""
        
        recent = list(self.temporal_memory.recent_detections)[-10:]  # Last 10 detections
        
        # Pattern: Consistent detection confidence
        confidences = [d['confidence'] for d in recent]
        if len(confidences) >= 5:
            confidence_stability = 1.0 - np.std(confidences)
            self.temporal_memory.detection_patterns['confidence_stability'] = confidence_stability
        
        # Pattern: Spatial convergence (zones getting closer together)
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
        
        # Environment type learning
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
            
            # Apply exponential decay
            decay_factor = self.confidence_decay_rate ** (time_since_seen / 1.0)  # Decay per second
            zone.avg_confidence *= decay_factor
            zone.position_uncertainty = min(5.0, zone.position_uncertainty * 1.01)  # Increase uncertainty over time
    
    def _update_confidence_grid(self):
        """Update the probabilistic confidence grid"""
        
        if self.spatial_memory.confidence_grid is None:
            return
        
        # Clear grid
        self.spatial_memory.confidence_grid.fill(0.0)
        
        grid_size = self.spatial_memory.grid_size
        resolution = self.spatial_resolution
        
        # Add confidence from memory zones
        for zone in self.spatial_memory.memory_zones:
            if zone.observation_count >= self.min_observations:
                # Convert world position to grid coordinates
                grid_x = int((zone.world_position[0] + grid_size * resolution / 2) / resolution)
                grid_y = int((zone.world_position[1] + grid_size * resolution / 2) / resolution)
                
                if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                    # Add Gaussian blob around zone position
                    sigma = zone.position_uncertainty / resolution
                    self._add_gaussian_to_grid(grid_x, grid_y, zone.avg_confidence, sigma)
    
    def _add_gaussian_to_grid(self, center_x: int, center_y: int, confidence: float, sigma: float):
        """Add a Gaussian confidence blob to the grid"""
        
        grid = self.spatial_memory.confidence_grid
        grid_size = grid.shape[0]
        
        # Create Gaussian kernel
        kernel_size = min(int(3 * sigma), grid_size // 4)
        if kernel_size < 1:
            kernel_size = 1
        
        for dy in range(-kernel_size, kernel_size + 1):
            for dx in range(-kernel_size, kernel_size + 1):
                gx, gy = center_x + dx, center_y + dy
                
                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    distance_sq = dx**2 + dy**2
                    gaussian_value = confidence * math.exp(-distance_sq / (2 * sigma**2))
                    grid[gy, gx] = max(grid[gy, gx], gaussian_value)
    
    def predict_zones_from_memory(self, 
                                 min_confidence: float = 0.3,
                                 max_zones: int = 5) -> List[Dict]:
        """
        Predict likely landing zones based on memory when visual input is poor
        
        Returns:
            List of predicted zone dictionaries with world coordinates
        """
        
        predicted_zones = []
        current_time = time.time()
        
        # Get zones from spatial memory
        for zone in self.spatial_memory.memory_zones:
            if (zone.avg_confidence >= min_confidence and 
                zone.observation_count >= self.min_observations):
                
                # Calculate current confidence (with temporal decay)
                time_since_seen = current_time - zone.last_seen
                current_confidence = zone.avg_confidence * (self.confidence_decay_rate ** time_since_seen)
                
                if current_confidence >= min_confidence:
                    # Convert back to world coordinates
                    if self.spatial_memory.reference_position:
                        ref_x, ref_y = self.spatial_memory.reference_position
                        world_x = zone.world_position[0] + ref_x
                        world_y = zone.world_position[1] + ref_y
                        
                        predicted_zones.append({
                            'center': self._world_to_pixel(zone.world_position),
                            'world_position': (world_x, world_y),
                            'confidence': current_confidence,
                            'area': (zone.estimated_size * 10)**2,  # Rough area estimate
                            'source': 'memory',
                            'uncertainty': zone.position_uncertainty,
                            'last_seen': zone.last_seen,
                            'observation_count': zone.observation_count
                        })
        
        # Sort by confidence
        predicted_zones.sort(key=lambda x: x['confidence'], reverse=True)
        
        return predicted_zones[:max_zones]
    
    def _world_to_pixel(self, relative_world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert relative world coordinates to approximate pixel coordinates"""
        # This is a simplified conversion - in practice you'd use camera parameters
        # For now, assume 1 meter = 50 pixels at typical altitude
        scale = 50.0
        center_x, center_y = 320, 240  # Assume 640x480 image
        
        pixel_x = int(center_x + relative_world_pos[0] * scale)
        pixel_y = int(center_y + relative_world_pos[1] * scale)
        
        return (pixel_x, pixel_y)
    
    def get_memory_confidence(self, position: Tuple[float, float]) -> float:
        """Get memory-based confidence for a given world position"""
        
        if self.spatial_memory.reference_position is None:
            return 0.0
        
        # Convert to relative coordinates
        ref_x, ref_y = self.spatial_memory.reference_position
        rel_pos = (position[0] - ref_x, position[1] - ref_y)
        
        # Check confidence grid
        grid = self.spatial_memory.confidence_grid
        if grid is not None:
            grid_size = self.spatial_memory.grid_size
            resolution = self.spatial_resolution
            
            grid_x = int((rel_pos[0] + grid_size * resolution / 2) / resolution)
            grid_y = int((rel_pos[1] + grid_size * resolution / 2) / resolution)
            
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                return float(grid[grid_y, grid_x])
        
        return 0.0
    
    def fuse_perception_memory(self, 
                              perception_zones: List[Dict], 
                              perception_confidence: float = 1.0) -> List[Dict]:
        """
        Fuse current perception with memory predictions
        
        Args:
            perception_zones: Zones detected in current frame
            perception_confidence: Overall confidence in current perception
            
        Returns:
            Fused list of zones with combined confidence scores
        """
        
        memory_zones = self.predict_zones_from_memory()
        fused_zones = []
        
        # Weight between perception and memory
        perception_weight = perception_confidence
        memory_weight = 1.0 - perception_confidence
        
        # Add perception zones with high weight
        for zone in perception_zones:
            zone_copy = zone.copy()
            zone_copy['confidence'] *= perception_weight
            zone_copy['source'] = 'perception'
            fused_zones.append(zone_copy)
        
        # Add memory zones that don't overlap with perception
        for mem_zone in memory_zones:
            # Check for overlap with perception zones
            overlaps = False
            mem_center = mem_zone.get('center', (0, 0))
            
            for perc_zone in perception_zones:
                perc_center = perc_zone.get('center', (0, 0))
                distance = math.sqrt(
                    (mem_center[0] - perc_center[0])**2 + 
                    (mem_center[1] - perc_center[1])**2
                )
                
                if distance < 50:  # pixels
                    overlaps = True
                    break
            
            if not overlaps:
                mem_zone_copy = mem_zone.copy()
                mem_zone_copy['confidence'] *= memory_weight
                fused_zones.append(mem_zone_copy)
        
        return fused_zones
    
    def get_memory_status(self) -> Dict:
        """Get current memory system status for debugging"""
        
        current_time = time.time()
        
        # Count active memory zones
        active_zones = [z for z in self.spatial_memory.memory_zones 
                       if (current_time - z.last_seen) < 60.0]  # Active if seen within last minute
        
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
        
        print(f"💾 Memory saved to {filepath}")
    
    def load_memory(self, filepath: str):
        """Load memory state from file"""
        
        if not Path(filepath).exists():
            print(f"⚠️  Memory file {filepath} not found, starting fresh")
            return
        
        try:
            with open(filepath, 'r') as f:
                memory_data = json.load(f)
            
            # Restore spatial memory
            if 'spatial_memory' in memory_data:
                spatial_data = memory_data['spatial_memory']
                self.spatial_memory.reference_position = spatial_data.get('reference_position')
                
                # Restore memory zones
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
            
            print(f"🧠 Memory loaded from {filepath}")
            print(f"   Restored {len(self.spatial_memory.memory_zones)} memory zones")
            
        except Exception as e:
            print(f"❌ Error loading memory: {e}")
    
    def reset_memory(self):
        """Reset all memory (use carefully!)"""
        
        self.spatial_memory = SpatialMemoryState()
        self.temporal_memory = TemporalMemoryState() 
        self.semantic_memory = SemanticMemoryState()
        self._initialize_spatial_grid()
        self.frame_count = 0
        
        print("🔄 Memory system reset")
