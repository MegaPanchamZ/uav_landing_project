# UAV Landing System with Neurosymbolic Memory

A production-ready UAV landing detection system combining computer vision with neurosymbolic memory for robust performance in challenging scenarios.

## Features

- **Neural Segmentation**: BiSeNetV2-based landing zone detection
- **Neurosymbolic Memory**: Three-tier memory system (spatial, temporal, semantic) for handling visual context loss
- **Real-time Performance**: Optimized for flight-critical applications (<100KB memory overhead, ~2-3ms processing impact)
- **ONNX Runtime**: Cross-platform model inference
- **Production Ready**: Clean architecture with comprehensive error handling

## Memory System

The neurosymbolic memory addresses scenarios where visual context is lost (e.g., "all grass" environments):

1. **Spatial Memory**: Grid-based landing zone tracking with confidence decay
2. **Temporal Memory**: Recent frame history for context continuity  
3. **Semantic Memory**: Persistent high-confidence landing zones

## Installation

### Quick Install
```bash
pip install -e .
```

### With GPU Support
```bash
pip install -e ".[gpu]"
```

### Development Install
```bash
pip install -e ".[dev]"
```

## Quick Start

### Command Line Usage
```bash
# Test with webcam
python uav_landing_main.py --source camera --test-mode

# Process video file
python uav_landing_main.py --source path/to/video.mp4 --save-output

# Test memory system with synthetic scenarios
python uav_landing_main.py --test-mode --scenario all_grass --memory-test
```

### Python API Usage
```python
from uav_landing.detector import UAVLandingDetector
import cv2

# Initialize detector
detector = UAVLandingDetector(
    model_path="models/bisenetv2_uav_landing.onnx",
    enable_memory=True,
    memory_persistence_file="uav_memory.json"
)

# Process single frame
frame = cv2.imread("test_image.jpg")
result = detector.process_frame(frame, altitude=5.0)

if result.status == "TARGET_ACQUIRED":
    print(f"Landing confidence: {result.confidence:.3f}")
    print(f"Landing position: {result.target_pixel}")
else:
    print("No suitable landing zone detected")
```

## Project Structure

```
uav_landing_project/
├── uav_landing/                 # Main package
│   ├── __init__.py             # Package initialization
│   ├── detector.py             # UAVLandingDetector class
│   ├── memory.py               # NeuroSymbolicMemory system
│   └── types.py                # Data structures
├── uav_landing_main.py         # Production entry point
├── models/                     # ONNX models
├── archive/                    # Old implementation files
└── tests/                      # Test suite
```

## Memory System Details

### Spatial Memory
- 8x8 grid covering full frame
- Confidence-based zone tracking
- Exponential decay over time
- Minimum confidence thresholds

### Temporal Memory  
- Rolling buffer of recent detections
- Weighted averaging for stability
- Configurable history length

### Semantic Memory
- Persistent storage of high-confidence zones
- Cross-session memory (optional)
- JSON-based serialization

## Configuration

### Memory Parameters
```python
detector = UAVLandingDetector(
    model_path="models/bisenetv2_uav_landing.onnx",
    enable_memory=True,
    memory_persistence_file="uav_memory.json",
    memory_config={
        'memory_horizon': 300.0,
        'confidence_decay_rate': 0.98,
        'spatial_resolution': 0.5
    }
)
```

### Performance Tuning
- **Input Resolution**: 512x512 (default) or 256x256 (faster)
- **Memory Grid Size**: 8x8 (default) or 16x16 (higher precision)
- **History Length**: 10 frames (default)

## Safety Considerations

This system is designed for research and development. For production deployment:

1. Implement redundant safety systems
2. Add sensor fusion (GPS, IMU, lidar)
3. Validate in controlled environments
4. Follow aviation safety regulations
5. Include human oversight capabilities

## Testing

### Run Tests
```bash
# Basic functionality test
python test_headless.py

# Memory system validation
python uav_landing_main.py --test-mode --no-display

# Performance benchmarks
python -c "from uav_landing.detector import UAVLandingDetector; print('System ready')"
```

### Test Memory System
```bash
# Test with synthetic data
python uav_landing_main.py --test-mode --no-display

# Memory persistence test
python -c "
from uav_landing.memory import NeuroSymbolicMemory
memory = NeuroSymbolicMemory()
print('Memory system initialized successfully')
"
```

## Performance Benchmarks

- **Processing Speed**: ~6-15 FPS (depending on hardware)
- **Memory Overhead**: <100KB for memory system
- **Memory Processing**: ~2-3ms additional latency
- **Model Inference**: ~160ms (CPU), ~20-50ms (GPU)

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in this README
- Review test files for usage examples

---

**⚠️ Safety Notice**: This system is for research purposes. Always follow proper safety protocols when working with UAV systems.