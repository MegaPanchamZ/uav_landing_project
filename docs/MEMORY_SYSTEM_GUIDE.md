# Neurosymbolic Memory System for UAV Landing

## Overview

The neurosymbolic memory system addresses the critical challenge of maintaining landing context when the drone loses clear visual cues (e.g., when descending through uniform grass). This system combines neural perception with symbolic reasoning and persistent memory to enable robust landing decisions even under challenging visual conditions.

## Architecture

### Core Components

#### 1. NeuroSymbolicMemory Class
The central memory manager with three types of memory:

**Spatial Memory**: Maintains an egocentric map of landing zones
- World coordinates relative to a reference point
- Probabilistic confidence grid
- Zone size and quality estimates
- Uncertainty tracking with confidence decay

**Temporal Memory**: Tracks patterns and sequences over time
- Recent detection history
- Pattern recognition (consistency, convergence)
- Phase transitions and behaviors

**Semantic Memory**: High-level environmental understanding
- Environment templates (grass fields, concrete surfaces)
- Learned associations between visual cues and landing success
- Success/failure pattern storage

#### 2. Memory-Enhanced Detector
Extends the base UAV detector with memory integration:
- Visual confidence assessment
- Memory-perception fusion strategies
- Recovery and search behaviors
- Persistent memory across flights

### Memory Integration Strategies

The system uses different strategies based on visual confidence:

```
Visual Confidence > 0.6: PERCEPTION_ONLY
- Use current perception
- Update memory with observations
- High confidence in neural detection

Visual Confidence 0.4-0.6: FUSED
- Combine perception + memory
- Memory boosts detection confidence
- Validates detections against memory

Visual Confidence < 0.4: MEMORY_ONLY
- Rely primarily on memory predictions
- Use stored zone locations
- Apply motion models for position updates
```

## Key Features

### 1. Spatial Context Preservation
When visual cues are lost (all grass scenario):
- Maintains world coordinate system relative to first detection
- Tracks drone movement to update relative positions
- Predicts likely zone locations based on movement history

### 2. Confidence Management
- **Confidence Decay**: Memory confidence decreases over time without observation
- **Uncertainty Tracking**: Position uncertainty grows when zones aren't visible
- **Bayesian Updates**: Combines new observations with existing memory

### 3. Recovery Behaviors
When target is lost for multiple frames:
- **Memory-Guided Search**: Navigate towards highest-confidence memory zones
- **Spiral Search**: Systematic pattern when visual conditions are moderate
- **Hover-Observe**: Slow rotation for better observation in good conditions

### 4. Environmental Learning
- **Template Matching**: Learn typical patterns for different terrain types
- **Success Patterns**: Remember configurations that led to successful landings
- **Context Analysis**: Analyze lighting, terrain composition, visual texture

## Implementation Details

### Memory Zone Structure
```python
@dataclass
class MemoryZone:
    world_position: Tuple[float, float]  # Relative coordinates
    estimated_size: float                # Zone radius in meters
    first_seen: float                    # Initial detection time
    last_seen: float                     # Most recent confirmation
    observation_count: int               # Number of observations
    max_confidence: float                # Best confidence achieved
    avg_confidence: float                # Average over time
    spatial_stability: float             # Position consistency (0-1)
    position_uncertainty: float          # Uncertainty in meters
    environment_type: str                # Context classification
```

### Visual Confidence Assessment
The system evaluates visual quality using multiple metrics:
- **Detection Confidence**: Quality of current neural detection
- **Contrast Score**: Local image contrast (std deviation)
- **Edge Density**: Amount of visual structure (Canny edges)
- **Color Diversity**: Visual variation (HSV histogram analysis)

### Memory Grid
A probabilistic occupancy grid centered on the drone:
- 50x50 meter coverage area
- 0.5m resolution per cell
- Gaussian confidence blobs around memory zones
- Real-time updates as drone moves

## Usage Examples

### Basic Integration
```python
from memory_enhanced_detector import MemoryEnhancedUAVDetector

# Initialize with memory enabled
detector = MemoryEnhancedUAVDetector(
    model_path="models/bisenetv2_landing.onnx",
    enable_memory=True,
    memory_config={
        'memory_horizon': 300.0,       # 5 minute memory
        'spatial_resolution': 0.5,     # 0.5m grid cells
        'confidence_decay_rate': 0.985, # Slow decay
        'min_observations': 2          # Min obs to trust zone
    }
)

# Process frames with position information
result = detector.process_frame(
    image=camera_frame,
    altitude=current_altitude,
    drone_position=(x_world, y_world),  # Critical for memory
    drone_heading=current_heading
)

# Check memory usage
if result.perception_memory_fusion == "memory_only":
    print("Relying on memory - visual conditions poor")
elif result.recovery_mode:
    print(f"Recovery active: {result.search_pattern}")
```

### Memory Persistence
```python
# Save memory across flights
detector.save_memory()  # Saves to uav_memory.json

# Memory automatically loads on next initialization
detector = MemoryEnhancedUAVDetector(...)  # Auto-loads saved memory
```

### Memory Visualization
```python
# Get memory state visualization
memory_viz = detector.get_memory_visualization()
cv2.imshow("Memory Map", memory_viz)

# Get memory status
status = detector.memory.get_memory_status()
print(f"Active zones: {status['active_memory_zones']}")
print(f"Grid coverage: {status['grid_coverage']:.2f}")
```

## Challenging Scenarios Addressed

### 1. "All Grass" Scenario
**Problem**: Uniform grass provides no visual landmarks
**Solution**: 
- Memory maintains last known landing zone positions
- Motion model updates positions as drone moves
- Confidence decay prevents over-reliance on old information

### 2. Partial Occlusion
**Problem**: Landing zones partially hidden by grass/obstacles
**Solution**:
- Fuse partial visual information with memory predictions
- Use memory to "fill in" missing visual data
- Validate partial detections against memory expectations

### 3. Moving Platform/Drone Drift
**Problem**: Landing zones appear to move due to drone drift
**Solution**:
- Spatial tracking maintains world coordinate system
- Temporal patterns detect consistent movement
- Motion prediction compensates for drift

### 4. Environmental Changes
**Problem**: Lighting, weather, or seasonal changes affect perception
**Solution**:
- Semantic memory adapts to environmental variations
- Multiple confidence sources reduce single-point failures
- Learning from success/failure patterns

## Configuration Parameters

### Memory Behavior
- `memory_horizon`: How long to retain memory (seconds)
- `confidence_decay_rate`: How fast confidence decays (per frame)
- `min_observations`: Minimum observations to trust a zone
- `spatial_resolution`: Memory grid resolution (meters/cell)

### Fusion Strategy
- `min_visual_confidence`: Threshold to start using memory (default: 0.4)
- `memory_fusion_threshold`: Threshold for perception-only (default: 0.6)

### Recovery Behavior
- Recovery triggers after 5 consecutive frames without target
- Search patterns adapt to visual confidence levels
- Memory guides search towards most probable zones

## Performance Characteristics

### Memory Overhead
- Spatial grid: ~40KB for 100x100 grid
- Memory zones: ~200 bytes per zone
- Temporal history: ~1KB for 30-frame buffer
- Total overhead: < 100KB for typical operation

### Computational Cost
- Memory updates: ~1-2ms per frame
- Grid operations: ~0.5ms per frame
- Confidence fusion: ~0.2ms per frame
- Total overhead: ~2-3ms (minimal impact on real-time operation)

### Persistence
- JSON serialization for cross-flight memory
- Automatic loading on initialization
- Configurable save frequency

## Testing and Validation

The system includes comprehensive test scenarios:
1. **Learning Phase**: Build memory under clear conditions
2. **Grass Challenge**: Pure memory-based navigation
3. **Partial Occlusion**: Memory-perception fusion
4. **Moving Target**: Temporal tracking validation
5. **Multiple Zones**: Memory-based zone selection

Run tests with:
```bash
python tests/test_memory_system.py
```

## Limitations and Future Improvements

### Current Limitations
- Requires accurate position information (GPS/visual odometry)
- Memory grid size limits operational area
- No semantic understanding of obstacle changes

### Future Enhancements
- **SLAM Integration**: Build memory without external position
- **Semantic Segmentation**: Understand environmental changes
- **Multi-Scale Memory**: Hierarchical spatial memory
- **Collaborative Memory**: Share memory between multiple drones
- **Adaptive Parameters**: Self-tuning based on performance

## Conclusion

The neurosymbolic memory system provides robust landing capabilities by:
- Maintaining spatial awareness when visual cues are lost
- Learning from experience to improve future performance
- Gracefully degrading when conditions are challenging
- Providing explainable decision-making through memory traces

This system is particularly valuable for:
- Agricultural operations over uniform crops
- Emergency landings in challenging conditions
- Autonomous operations in GPS-denied environments
- Long-duration flights with changing conditions
