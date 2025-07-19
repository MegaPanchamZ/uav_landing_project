# UAV Landing Zone Detector 🚁

**Production-ready, single-class implementation for real-time UAV landing zone detection**

## 🎯 Core Implementation

### Essential Files
```
📁 uav_landing_project/
├── uav_landing_detector.py    # 🎯 Main detector class (CORE)
├── demo.py                    # 🎮 Interactive demo & testing
├── convert_to_onnx.py         # 🔄 PyTorch → ONNX converter
├── requirements.txt           # 📦 Dependencies
└── FINAL_STATUS.py           # 📊 Status checker
```

### Supporting Directories
```
📁 models/                     # 🧠 ONNX model files
├── bisenetv2_uav_landing.onnx # Converted BiSeNetV2 model

📁 tests/                      # 🧪 Test scripts
├── quick_test.py             # Basic functionality test
└── test_real_model.py        # ONNX model test

📁 training_tools/            # 🏗️ Training pipeline (optional)
├── training_pipeline.py     # Complete training orchestration
├── fine_tuning_pipeline.py  # BiSeNetV2 implementation
└── dataset_preparation.py   # Dataset tools
```

## 🚀 Quick Start

```python
from uav_landing_detector import UAVLandingDetector

# Initialize detector (warm-up)
detector = UAVLandingDetector(
    model_path="models/bisenetv2_uav_landing.onnx",
    enable_visualization=True
)

# Process camera frame
result = detector.process_frame(image, altitude=5.0)

# Get navigation commands
if result.status == "TARGET_ACQUIRED":
    print(f"Target at {result.distance:.1f}m")
    print(f"Commands: F={result.forward_velocity:.1f}, R={result.right_velocity:.1f}, D={result.descent_rate:.1f}")
```

## 📋 Installation

```bash
# Core dependencies only
pip install opencv-python numpy onnxruntime

# Or install all
pip install -r requirements.txt
```

## 🎮 Testing & Demo

```bash
# Quick functionality test
python tests/quick_test.py

# Test with real ONNX model
python tests/test_real_model.py

# Interactive demo
python demo.py
```

## 🔄 Model Conversion

Convert your BiSeNetV2 `.pth` to ONNX:

```bash
python convert_to_onnx.py --input ../model_pths/your_model.pth
```

## ⚡ Performance

- **17+ FPS** on CPU (BiSeNetV2 ONNX)
- **<50ms** processing time per frame  
- **Memory**: ~200MB model loaded
- **GPU**: 60+ FPS expected with CUDA

## 🏗️ Architecture

**Single Class Design**: Everything in `UAVLandingDetector`
- Neural segmentation (BiSeNetV2)
- Symbolic reasoning (rule-based)
- Landing zone detection
- Navigation command generation
- Performance tracking
- Visualization (optional)

**Landing Classes**:
- 0: Background
- 1: Suitable (landing zones)
- 2: Marginal (rough terrain)
- 3: Obstacles (buildings, trees)
- 4: Unsafe (water, slopes)
- 5: Unknown

## 📊 Results Structure

```python
@dataclass
class LandingResult:
    status: str                 # "TARGET_ACQUIRED", "NO_TARGET", "UNSAFE"
    confidence: float           # 0.0-1.0
    target_pixel: tuple         # (x, y) image coordinates
    target_world: tuple         # (x, y) world coordinates (meters)
    distance: float             # Distance to target (meters)
    bearing: float              # Bearing (radians)
    
    # Navigation commands (ready for flight controller)
    forward_velocity: float     # m/s
    right_velocity: float       # m/s  
    descent_rate: float         # m/s
    yaw_rate: float            # rad/s
    
    # Performance
    processing_time: float      # milliseconds
    fps: float                 # frames per second
    annotated_image: ndarray    # Visualization (if enabled)
```

---

**🎯 Ready for extreme real-time UAV landing detection!**
- **Real-time Performance**: Optimized for real-time processing with performance monitoring

## Features

- ✅ Real-time semantic segmentation of landing zones
- ✅ Rule-based safety validation with configurable parameters
- ✅ Temporal stability tracking across multiple frames
- ✅ Comprehensive scoring system for zone ranking
- ✅ Performance monitoring and statistics
- ✅ Debug visualization and data logging
- ✅ Flexible configuration system
- ✅ Test video generation for development
- ✅ Support for webcam and video file input

## Installation

### Prerequisites

- Python 3.8 or higher
- UV package manager (recommended) or pip
- OpenCV-compatible camera or test videos

### Using UV (Recommended)

```bash
# Clone or create the project
cd uav_landing_project

# Install dependencies (already done if you followed the setup)
uv sync

# Activate the environment
source .venv/bin/activate
```

### Using Traditional pip

```bash
pip install -r requirements.txt
```

### Optional: GPU Acceleration

For NVIDIA GPU support, install additional packages:

```bash
pip install onnxruntime-gpu tensorrt
```

## Usage

### Basic Usage

```bash
# Run with webcam (default)
python main.py

# Run with a video file
python main.py --video path/to/your/video.mp4

# Run with debug output
python main.py --video 0 --output debug_frames/
```

### Generating Test Videos

```bash
# Generate test videos for development
python generate_test_video.py

# Generate specific scenarios
python generate_test_video.py --scenarios mixed urban --duration 60

# Custom resolution and frame rate
python generate_test_video.py --resolution 1280x720 --fps 30
```

### Command Line Options

```bash
python main.py --help

Options:
  --video, -v    Video source (0 for webcam, path for file)
  --output, -o   Output directory for debug files
  --config, -c   Path to configuration file
```

## System Architecture

### Neural Engine (`neural_engine.py`)

- **Purpose**: Semantic segmentation of camera frames
- **Model**: BiSeNetV2 (ONNX format for cross-platform compatibility)
- **Classes**:
  - `background` (Class 0)
  - `safe_flat_surface` (Class 1)
  - `unsafe_uneven_surface` (Class 2)
  - `low_obstacle` (Class 3)
  - `high_obstacle` (Class 4)

### Symbolic Engine (`symbolic_engine.py`)

- **Purpose**: Rule-based reasoning and decision making
- **Key Rules**:
  - Minimum area requirements
  - Obstacle clearance validation
  - Geometric shape analysis
  - Temporal stability tracking
  - Multi-criteria scoring

### Configuration (`config.py`)

All system parameters are centralized in the configuration file:

- Neural network settings
- Class definitions and mappings
- Rule parameters (clearances, thresholds)
- Temporal stability settings
- Scoring weights
- Visualization parameters

## Key Rules and Logic

### Geometric Validation

- **Minimum Area**: Zones must be at least 3000 pixels
- **Aspect Ratio**: Between 0.3 and 3.0 to avoid overly elongated zones
- **Solidity**: At least 0.6 to ensure zones are reasonably filled

### Spatial Safety

- **High Obstacle Clearance**: 30 pixels minimum
- **Low Obstacle Clearance**: 15 pixels minimum
- **Distance calculation**: Uses OpenCV's point-to-polygon distance

### Temporal Stability

- **Window Size**: Tracks last 15 frames
- **Confirmation Threshold**: Zone must be valid in at least 10 of the last 15 frames
- **Position-based Tracking**: Groups zones by discretized position and size

### Scoring System

Final score is weighted combination of:
- **Area Score** (40%): Larger zones preferred
- **Center Score** (30%): Zones closer to image center preferred
- **Shape Score** (20%): Well-formed shapes preferred
- **Clearance Score** (10%): Greater clearance from obstacles preferred

## Controls

While running the application, use these keyboard shortcuts:

- **'q'**: Quit the application
- **'s'**: Save the current frame as an image
- **'r'**: Reset temporal tracking
- **'p'**: Print current performance statistics

## Performance Optimization

### CPU Optimization

- Morphological operations are minimized
- Efficient contour analysis
- Vectorized numerical operations with NumPy

### GPU Acceleration (Future)

The system is designed to support TensorRT optimization:

```bash
# Convert ONNX to TensorRT engine (requires TensorRT installation)
trtexec --onnx=bisenetv2_uav.onnx \
        --saveEngine=bisenetv2_uav.engine \
        --fp16
```

Then update `config.py`:
```python
MODEL_PATH = "bisenetv2_uav.engine"
```

## Development and Testing

### Test Scenarios

The test video generator creates four scenarios:

1. **Mixed**: Varied terrain with moderate obstacles
2. **Urban**: Pavement-heavy with buildings and vehicles
3. **Rural**: Grass-heavy with trees and natural obstacles
4. **Challenging**: Dense obstacles with limited safe zones

### Debug Mode

Enable debug mode in `config.py`:

```python
DEBUG_MODE = True
SAVE_DEBUG_FRAMES = True
```

### Performance Monitoring

The system tracks:
- Neural network inference time and FPS
- Symbolic processing time and FPS
- Zone detection statistics
- Memory usage (planned)

## Model Training (Future Work)

To train your own BiSeNetV2 model:

1. Collect UAV footage from your specific use case
2. Annotate frames using tools like CVAT or Labelbox
3. Fine-tune BiSeNetV2 on your dataset
4. Export to ONNX format
5. Optionally convert to TensorRT for maximum performance

### Annotation Guidelines

- **Class 1 (safe_flat_surface)**: Short grass, pavement, dirt roads, landing pads
- **Class 2 (unsafe_uneven_surface)**: Tall grass, bushes, water, rocky terrain
- **Class 3 (low_obstacle)**: Curbs, small debris, low fences
- **Class 4 (high_obstacle)**: Trees, buildings, people, vehicles, tall structures

## Integration with Flight Controllers

The system outputs structured decision data that can be integrated with flight controllers:

```python
decision = {
    'status': 'TARGET_ACQUIRED',
    'zone': {
        'center': (x, y),  # Pixel coordinates
        'score': 0.85,     # Confidence score
        'area': 5000,      # Zone area in pixels
        # ... additional metadata
    }
}
```

### Coordinate Transformation

To convert pixel coordinates to real-world coordinates, implement:

```python
def pixel_to_world(pixel_coords, altitude, camera_fov, image_resolution):
    """Convert pixel coordinates to meter offsets from current position."""
    # Implementation depends on camera calibration and mounting
    pass
```

## Troubleshooting

### Common Issues

1. **No camera found**: Check camera permissions and connection
2. **Poor performance**: Verify OpenCV installation and consider GPU acceleration
3. **No valid zones detected**: Adjust rule parameters in `config.py`
4. **Model not found**: Ensure ONNX model file exists or use placeholder mode

### Performance Tuning

Adjust these parameters in `config.py`:
- `MIN_LANDING_AREA_PIXELS`: Reduce for smaller zones
- `*_CLEARANCE`: Reduce for tighter obstacle tolerance
- `TEMPORAL_CONFIRMATION_THRESHOLD`: Reduce for faster decisions

## Contributing

1. Follow the existing code structure and documentation patterns
2. Add unit tests for new functionality
3. Update the configuration file for new parameters
4. Test with multiple scenarios using the test video generator

## License

This project is intended for research and development purposes. Ensure compliance with local regulations for UAV operations.

## Future Enhancements

- [ ] Integration with ROS/ROS2
- [ ] MAVLink protocol support
- [ ] Advanced tracking algorithms (Kalman filtering)
- [ ] Machine learning-based rule optimization
- [ ] Multi-camera fusion
- [ ] Weather condition adaptation
- [ ] Emergency landing protocols
- [ ] Integration with obstacle avoidance systems

## Acknowledgments

- BiSeNetV2 architecture for efficient semantic segmentation
- OpenCV community for computer vision tools
- ONNX Runtime for cross-platform model deployment
