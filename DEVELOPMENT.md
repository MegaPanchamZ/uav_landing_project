# Development Guide - UAV Landing Zone Detection System

This guide helps developers understand, extend, and maintain the UAV Landing Zone Detection System.

## Architecture Overview

```
┌─────────────────────┐    ┌─────────────────────┐    ┌──────────────────────┐
│   Camera Input      │────│   Neural Engine     │────│   Symbolic Engine    │
│                     │    │                     │    │                      │
│ • Webcam            │    │ • BiSeNetV2 Model   │    │ • Rule-based Logic   │
│ • Video File        │    │ • Segmentation      │    │ • Temporal Tracking  │
│ • Image Stream      │    │ • Class Prediction  │    │ • Zone Validation    │
└─────────────────────┘    └─────────────────────┘    └──────────────────────┘
                                      │                           │
                                      ▼                           ▼
                           ┌─────────────────────────────────────────────┐
                           │           Main Application                  │
                           │                                             │
                           │ • Visualization                             │
                           │ • Performance Monitoring                    │
                           │ • User Interface                            │
                           │ • Data Logging                              │
                           └─────────────────────────────────────────────┘
```

## Code Organization

### Core Components

1. **`config.py`** - Centralized configuration
2. **`neural_engine.py`** - Neural network wrapper
3. **`symbolic_engine.py`** - Rule-based reasoning
4. **`main.py`** - Application orchestration

### Utility Components

5. **`test_system.py`** - Comprehensive test suite
6. **`generate_test_video.py`** - Synthetic test data generation
7. **`manage.py`** - Project management utility

## Key Design Principles

### 1. Modularity
- Each component has a single responsibility
- Clean interfaces between modules
- Easy to test and modify individual parts

### 2. Configuration-Driven
- All parameters externalized to `config.py`
- Easy to tune without code changes
- Support for different deployment scenarios

### 3. Performance-Aware
- Real-time processing considerations
- Performance monitoring built-in
- Efficient algorithms and data structures

### 4. Robust Error Handling
- Graceful degradation when models are unavailable
- Comprehensive error reporting
- Fallback modes for development

## Extending the System

### Adding New Rules

1. **Geometric Rules** (in `SymbolicEngine._is_zone_geometrically_valid`)
```python
def _is_zone_geometrically_valid(self, zone: LandingZone) -> bool:
    # Existing rules...
    
    # Add new rule - example: minimum perimeter
    if zone.perimeter < config.MIN_ZONE_PERIMETER:
        return False
    
    # Add new rule - example: convexity check
    hull_perimeter = cv2.arcLength(cv2.convexHull(zone.contour), True)
    convexity_ratio = zone.perimeter / hull_perimeter if hull_perimeter > 0 else 0
    if convexity_ratio < config.MIN_CONVEXITY_RATIO:
        return False
    
    return True
```

2. **Spatial Rules** (in `SymbolicEngine._is_zone_spatially_safe`)
```python
def _is_zone_spatially_safe(self, zone: LandingZone, obstacles: List[Dict]) -> bool:
    # Existing clearance checks...
    
    # Add new rule - example: check for water bodies
    water_mask = (seg_map == config.WATER_CLASS_ID).astype(np.uint8)
    water_distance = cv2.distanceTransform(1 - water_mask, cv2.DIST_L2, 5)
    min_water_distance = np.min(water_distance[zone.mask])
    
    if min_water_distance < config.MIN_WATER_CLEARANCE:
        return False
    
    return True
```

### Adding New Scoring Criteria

1. **Extend the LandingZone class**:
```python
class LandingZone:
    def _compute_properties(self):
        # Existing properties...
        
        # Add new property - example: texture analysis
        self.texture_score = self._analyze_texture()
        
        # Add new property - example: slope estimation
        self.estimated_slope = self._estimate_slope()
    
    def _analyze_texture(self):
        # Implement texture analysis using local binary patterns
        # or other computer vision techniques
        return texture_score
```

2. **Update scoring function**:
```python
def _calculate_zone_score(self, zone: LandingZone) -> float:
    # Existing scores...
    
    # Add new score component
    texture_score = zone.texture_score
    
    # Update weights in config.py
    final_score = (
        config.W_AREA * area_score +
        config.W_CENTER * center_score +
        config.W_SHAPE * shape_score +
        config.W_CLEARANCE * clearance_score +
        config.W_TEXTURE * texture_score  # New weight
    )
    
    return final_score
```

### Adding New Neural Network Backends

1. **TensorRT Support**:
```python
class TensorRTEngine(NeuralEngine):
    def _load_model(self):
        import tensorrt as trt
        import pycuda.driver as cuda
        
        # TensorRT-specific loading code
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        
        with open(self.model_path, "rb") as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.trt_logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
```

2. **OpenVINO Support**:
```python
class OpenVINOEngine(NeuralEngine):
    def _load_model(self):
        from openvino.runtime import Core
        
        ie = Core()
        model = ie.read_model(self.model_path)
        self.compiled_model = ie.compile_model(model, "CPU")
        self.infer_request = self.compiled_model.create_infer_request()
```

### Adding New Visualization Features

1. **3D Visualization**:
```python
def visualize_3d_zones(self, frame, seg_map, decision):
    # Use OpenCV's stereo vision or depth estimation
    # to create 3D visualization of landing zones
    pass
```

2. **Augmented Reality Overlay**:
```python
def create_ar_overlay(self, frame, zones, camera_params):
    # Project zone coordinates to 3D space
    # Add AR markers and information
    pass
```

## Testing Guidelines

### Unit Tests

Create test files for each component:

```python
# test_neural_engine.py
class TestNeuralEngine:
    def test_preprocessing(self):
        engine = NeuralEngine()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        preprocessed = engine._preprocess_frame(frame)
        assert preprocessed.shape == (1, 3, 512, 512)
```

### Integration Tests

Test component interactions:

```python
def test_end_to_end_pipeline():
    neural_engine = NeuralEngine()
    symbolic_engine = SymbolicEngine()
    
    for scenario in ["empty", "single_zone", "multiple_zones", "obstacles"]:
        frame = create_test_scenario(scenario)
        seg_map = neural_engine.process_frame(frame)
        decision = symbolic_engine.run(seg_map)
        validate_decision(decision, scenario)
```

### Performance Tests

Monitor performance continuously:

```python
def test_performance_regression():
    # Ensure new changes don't degrade performance
    # beyond acceptable thresholds
    assert avg_processing_time < 50  # milliseconds
    assert memory_usage < 500  # MB
```

## Deployment Considerations

### Model Optimization

1. **Convert to ONNX**: `torch.onnx.export()`
2. **Optimize with TensorRT**: `trtexec --onnx=model.onnx --saveEngine=model.engine --fp16`
3. **Quantization**: Use INT8 quantization for edge devices

### Edge Deployment

1. **NVIDIA Jetson**:
   - Use TensorRT for maximum performance
   - Enable CUDA memory pooling
   - Optimize memory allocation patterns

2. **Raspberry Pi**:
   - Use CPU-optimized ONNX Runtime
   - Consider model pruning/quantization
   - Implement frame skipping if necessary

### Flight Controller Integration

1. **MAVLink Integration**:
```python
from pymavlink import mavutil

def send_landing_command(connection, zone_coords):
    # Convert pixel coordinates to GPS coordinates
    lat, lon = pixel_to_gps(zone_coords)
    
    # Send MAVLink command
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, lat, lon, 0
    )
```

## Debugging Guide

### Common Issues

1. **No zones detected**:
   - Check `MIN_LANDING_AREA_PIXELS` setting
   - Verify segmentation quality
   - Review obstacle clearance parameters

2. **Poor temporal stability**:
   - Adjust `TEMPORAL_CONFIRMATION_THRESHOLD`
   - Check zone tracking algorithm
   - Review camera shake compensation

3. **Performance issues**:
   - Profile with `cProfile`
   - Check memory usage with `memory_profiler`
   - Optimize hot paths identified in profiling

### Debug Visualization

Enable comprehensive debugging:

```python
# In config.py
DEBUG_MODE = True
SAVE_DEBUG_FRAMES = True
DEBUG_OUTPUT_DIR = "debug_output"

# Add debug information to visualization
def visualize_debug_info(self, frame, seg_map, zones, obstacles):
    # Draw all potential zones (not just confirmed ones)
    # Show obstacle clearance circles
    # Display zone scores and properties
    # Add performance metrics overlay
    pass
```

## Contributing Guidelines

### Code Style

- Follow PEP 8
- Use type hints
- Document all public methods
- Keep functions focused and small

### Pull Request Process

1. Create feature branch from `main`
2. Implement changes with tests
3. Update documentation
4. Run full test suite
5. Submit PR with clear description

### Performance Monitoring

Always include performance impact analysis:
- Benchmark before/after changes
- Monitor memory usage
- Test on target hardware
- Include performance metrics in PR

---

For questions or contributions, please refer to the main README.md or create an issue in the project repository.
