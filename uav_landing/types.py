#!/usr/bin/env python3
"""
Data types and structures for UAV Landing System
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class LandingResult:
    """Complete landing detection result with memory enhancement"""
    
    # Core detection
    status: str  # 'TARGET_ACQUIRED', 'NO_TARGET', 'UNSAFE', 'SEARCHING'
    confidence: float  # 0.0-1.0
    
    # Target information
    target_pixel: Optional[Tuple[int, int]] = None  # (x, y) in image
    target_world: Optional[Tuple[float, float]] = None  # (x, y) in meters
    distance: Optional[float] = None  # meters from camera
    bearing: Optional[float] = None  # radians
    
    # Movement commands
    forward_velocity: float = 0.0  # m/s (positive = forward)
    right_velocity: float = 0.0   # m/s (positive = right) 
    descent_rate: float = 0.0     # m/s (positive = down)
    yaw_rate: float = 0.0         # rad/s (positive = clockwise)
    
    # Performance metrics
    processing_time: float = 0.0  # milliseconds
    fps: float = 0.0
    
    # Memory information
    memory_zones: List[Dict] = field(default_factory=list)
    memory_confidence: float = 0.0
    perception_memory_fusion: str = "perception_only"  # perception_only, memory_only, fused
    memory_status: Dict = field(default_factory=dict)
    
    # Recovery information
    recovery_mode: bool = False
    search_pattern: Optional[str] = None
    
    # Visualization (optional)
    annotated_image: Optional[np.ndarray] = None


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
    drone_trajectory: List[Dict] = field(default_factory=list)  # Recent positions
    memory_zones: List[MemoryZone] = field(default_factory=list)
    confidence_grid: Optional[np.ndarray] = None  # Probabilistic occupancy grid
    grid_resolution: float = 0.5  # meters per grid cell
    grid_size: int = 100  # 50x50 meter area around drone


@dataclass
class TemporalMemoryState:
    """Temporal patterns and sequences"""
    recent_detections: List[Dict] = field(default_factory=list)
    detection_patterns: Dict[str, float] = field(default_factory=dict)  # Pattern -> confidence
    phase_transitions: List[Tuple[str, str, float]] = field(default_factory=list)  # (from, to, time)


@dataclass
class SemanticMemoryState:
    """High-level environmental understanding"""
    environment_templates: Dict[str, Dict] = field(default_factory=dict)
    learned_associations: Dict[str, List[str]] = field(default_factory=dict)
    success_patterns: List[Dict] = field(default_factory=list)
    failure_patterns: List[Dict] = field(default_factory=list)
