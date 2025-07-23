# API Reference

Complete API documentation for the UAV Landing System with Neurosymbolic Memory.

## Core Classes

### UAVLandingDetector

Main detector class for UAV landing detection with memory enhancement.

```python
class UAVLandingDetector:
    """Production-ready UAV Landing Detector with Neurosymbolic Memory."""
    
    def __init__(self,
                 model_path: str = "models/bisenetv2_uav_landing.onnx",
                 input_resolution: Tuple[int, int] = (512, 512),
                 camera_fx: float = 800,
                 camera_fy: float = 800,
                 enable_memory: bool = True,
                 enable_visualization: bool = False,
                 memory_config: Optional[Dict] = None,
                 memory_persistence_file: str = "uav_memory.json",
                 device: str = "auto"):
        """
        Initialize UAV Landing Detector.
        
        Args:
            model_path: Path to ONNX segmentation model
            input_resolution: Model input size (width, height)
            camera_fx: Camera focal length in x direction
            camera_fy: Camera focal length in y direction
            enable_memory: Enable neurosymbolic memory system
            enable_visualization: Generate debug visualizations
            memory_config: Configuration for memory system
            memory_persistence_file: File to save/load memory state
            device: Inference device ("auto", "cuda", "cpu")
        """
```

#### Methods

##### `process_frame(image, altitude, current_velocity, drone_position, drone_heading) -> LandingResult`

Main method to process a single frame for landing zone detection.

```python
def process_frame(self,
                 image: np.ndarray,
                 altitude: float,
                 current_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 drone_position: Optional[Tuple[float, float]] = None,
                 drone_heading: float = 0.0) -> LandingResult:
    """
    Process single frame for landing zone detection.
    
    Args:
        image: Input BGR image from camera
        altitude: Current altitude above ground (meters)
        current_velocity: Current velocity [vx, vy, vz] (m/s)
        drone_position: Current drone position in world coordinates
        drone_heading: Current drone heading in radians
        
    Returns:
        LandingResult with detection, navigation, and memory information
    """
```

**Example:**
```python
detector = UAVLandingDetector(
    model_path="models/bisenetv2_uav_landing.onnx",
    enable_memory=True
)

frame = cv2.imread('drone_view.jpg')
result = detector.process_frame(frame, altitude=5.0)

if result.status == "TARGET_ACQUIRED":
    print(f"Landing target at {result.target_pixel}")
    print(f"Confidence: {result.confidence:.3f}")
```

##### `visualize_result(image, result) -> np.ndarray`

Generate visualization of detection results.

```python
def visualize_result(self, 
                    image: np.ndarray, 
                    result: LandingResult) -> np.ndarray:
    """
    Create visualization of detection results.
    
    Args:
        image: Original input image
        result: Detection result from process_frame()
        
    Returns:
        Annotated image with detection overlays
    """
```

##### `get_memory_status() -> Dict`

Get current memory system status.

```python
def get_memory_status(self) -> Dict:
    """
    Get memory system status for debugging.
    
    Returns:
        Dict with memory statistics and active zones
    """
```

## Data Types

### LandingResult

Main result object from frame processing.

```python
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
```

### MemoryZone

Represents a landing zone stored in memory.

```python
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
```

## Memory System

### NeuroSymbolicMemory

Memory system for persistent landing zone tracking.

```python
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
```

#### Key Methods

```python
def observe_zones(self, zones: List[Dict], world_positions: List[Tuple[float, float]], 
                 confidences: List[float], timestamp: float = None):
    """Update memory with new zone observations."""

def predict_zones_from_memory(self, min_confidence: float = 0.3, max_zones: int = 5) -> List[Dict]:
    """Predict likely landing zones based on memory when visual input is poor."""

def get_memory_confidence(self, position: Tuple[float, float]) -> float:
    """Get memory confidence for a specific position."""

def get_active_zones(self) -> List[MemoryZone]:
    """Get currently active memory zones."""

def save_memory(self, filepath: str):
    """Save memory state to file."""

def load_memory(self, filepath: str):
    """Load memory state from file."""
```

## Status Codes

### Landing Status

| Status | Description | Action |
|--------|-------------|---------|
| `TARGET_ACQUIRED` | Safe landing zone found | Can proceed with landing |
| `NO_TARGET` | No suitable zones detected | Continue searching |
| `UNSAFE` | Only unsafe zones found | Abort landing attempt |
| `SEARCHING` | Actively searching for zones | Continue flight pattern |

### Memory Fusion Modes

| Mode | Description | Usage |
|------|-------------|-------|
| `perception_only` | Using only current frame | Normal visual conditions |
| `memory_only` | Using only memory data | Poor visual conditions |
| `fused` | Combining perception + memory | Mixed conditions |

## Usage Examples

### Basic Detection

```python
from uav_landing.detector import UAVLandingDetector
import cv2

# Initialize detector
detector = UAVLandingDetector(
    model_path="models/bisenetv2_uav_landing.onnx"
)

# Process single frame
frame = cv2.imread("drone_view.jpg")
result = detector.process_frame(frame, altitude=5.0)

print(f"Status: {result.status}")
print(f"Confidence: {result.confidence:.3f}")
```

### Memory-Enhanced Detection

```python
# Initialize with memory enabled
detector = UAVLandingDetector(
    model_path="models/bisenetv2_uav_landing.onnx",
    enable_memory=True,
    memory_persistence_file="landing_memory.json"
)

# Process video stream
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    
    result = detector.process_frame(
        frame, 
        altitude=5.0,
        drone_position=(0.0, 0.0)  # GPS coordinates
    )
    
    # Check if memory is being used
    if result.perception_memory_fusion != "perception_only":
        print("Using memory assistance")
    
    # Visualize results
    vis_frame = detector.visualize_result(frame, result)
    cv2.imshow('Landing Detection', vis_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### Real-time Performance Monitoring

```python
detector = UAVLandingDetector()
frame_times = []

for i in range(100):
    frame = get_camera_frame()  # Your camera interface
    
    start_time = time.time()
    result = detector.process_frame(frame, altitude=5.0)
    frame_times.append(time.time() - start_time)
    
    print(f"Frame {i}: {result.processing_time:.1f}ms, "
          f"Status: {result.status}")

print(f"Average FPS: {1.0/np.mean(frame_times):.1f}")
```

## Configuration

### Default Configuration

```python
DEFAULT_CONFIG = {
    'model_path': 'models/bisenetv2_uav_landing.onnx',
    'input_resolution': (512, 512),
    'enable_memory': True,
    'memory_config': {
        'memory_horizon': 300.0,
        'confidence_decay_rate': 0.98,
        'spatial_resolution': 0.5
    }
}
```

### Memory Configuration

```python
memory_config = {
    'memory_horizon': 600.0,        # 10 minutes
    'confidence_decay_rate': 0.99,  # Slower decay
    'min_observations': 5,          # More observations needed
    'grid_size': 200               # Higher resolution grid
}

detector = UAVLandingDetector(
    enable_memory=True,
    memory_config=memory_config
)
```

## Error Handling

### Common Exceptions

```python
# Model loading errors
try:
    detector = UAVLandingDetector("invalid_path.onnx")
except FileNotFoundError:
    print("Model file not found")

# Processing errors
try:
    result = detector.process_frame(frame, altitude=5.0)
except ValueError as e:
    print(f"Invalid input: {e}")
```

### Robust Processing

```python
def safe_process_frame(detector, frame, altitude):
    """Safely process frame with error handling."""
    try:
        return detector.process_frame(frame, altitude)
    except Exception as e:
        print(f"Processing error: {e}")
        # Return safe default
        return LandingResult(
            status="ERROR",
            confidence=0.0,
            processing_time=0.0
        )
```

## Performance Notes

- **Input Resolution**: 512x512 (default) balances accuracy and speed
- **Memory Overhead**: <100KB for memory system
- **Processing Time**: ~160ms (CPU), ~20-50ms (GPU)
- **Memory Processing**: ~2-3ms additional latency

## Support

For API questions:
- Check examples in the main repository
- Review test files for usage patterns
- Refer to the main README for installation and setup

#### Methods

##### `detect_landing_site(image: np.ndarray) -> LandingResult`

Detect landing sites in an image.

```python
def detect_landing_site(self, image: np.ndarray) -> LandingResult:
    """
    Detect optimal landing sites in the given image.
    
    Args:
        image: Input RGB image (H, W, 3)
        
    Returns:
        LandingResult object containing detection results
        
    Raises:
        ValueError: If image format is invalid
        RuntimeError: If inference fails
    """
```

**Example:**
```python
detector = UAVLandingDetector()
image = cv2.imread('aerial_view.jpg')
result = detector.detect_landing_site(image)

if result.safe_landing:
    print(f"Safe landing at {result.best_site.center}")
```

##### `process_video_stream(video_path: str) -> Iterator[LandingResult]`

Process video stream for continuous detection.

```python
def process_video_stream(self, video_path: str) -> Iterator[LandingResult]:
    """
    Process video stream for real-time landing detection.
    
    Args:
        video_path: Path to video file or camera index (0, 1, ...)
        
    Yields:
        LandingResult for each frame
    """
```

##### `get_performance_stats() -> Dict[str, float]`

Get performance statistics.

```python
def get_performance_stats(self) -> Dict[str, float]:
    """
    Get performance statistics.
    
    Returns:
        Dict with keys: 'avg_inference_time', 'fps', 'total_detections'
    """
```

## 📊 Data Types

### LandingResult

```python
@dataclass
class LandingResult:
    """Result from landing site detection."""
    
    # Detection status
    safe_landing: bool              # Whether safe landing is possible
    confidence: float               # Overall confidence (0.0-1.0)
    
    # Best landing site
    best_site: Optional[LandingSite] # Best detected site
    all_sites: List[LandingSite]    # All detected sites
    
    # Performance metrics
    inference_time: float           # Inference time in milliseconds
    processing_time: float          # Total processing time
    
    # Visualization
    segmentation_mask: np.ndarray   # Segmentation result (H, W)
    annotated_image: np.ndarray     # Image with annotations
```

### LandingSite

```python
@dataclass
class LandingSite:
    """Detected landing site information."""
    
    # Location
    center: Tuple[int, int]         # Center coordinates (x, y)
    bbox: Tuple[int, int, int, int] # Bounding box (x, y, w, h)
    area: int                       # Area in pixels
    
    # Quality metrics
    safety_score: float             # Safety score (0.0-1.0)
    size_score: float              # Size adequacy (0.0-1.0)
    shape_score: float             # Shape quality (0.0-1.0)
    
    # Classification
    landing_class: int             # 0=Background, 1=Safe, 2=Caution, 3=Danger
    class_name: str                # Human-readable class name
```

##  Landing Classes

### Class Definitions

| ID | Name | Color | RGB | Description |
|----|------|-------|-----|-------------|
| 0 | Background | Black | (0,0,0) | Non-landing areas |
| 1 | Safe | Green | (0,255,0) | Optimal landing zones |
| 2 | Caution | Yellow | (255,255,0) | Possible with care |
| 3 | Danger | Red | (255,0,0) | No landing - hazards |

### Class Mapping Functions

```python
def get_class_color(class_id: int) -> Tuple[int, int, int]:
    """Get RGB color for class visualization."""
    colors = {
        0: (0, 0, 0),      # Black
        1: (0, 255, 0),    # Green  
        2: (255, 255, 0),  # Yellow
        3: (255, 0, 0),    # Red
    }
    return colors.get(class_id, (128, 128, 128))

def get_class_name(class_id: int) -> str:
    """Get human-readable class name."""
    names = {
        0: "Background",
        1: "Safe Landing",
        2: "Caution",
        3: "Danger"
    }
    return names.get(class_id, "Unknown")
```

## 🔧 Utility Functions

### Image Processing

```python
def preprocess_image(image: np.ndarray, 
                    target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Preprocess image for model inference.
    
    Args:
        image: Input RGB image (H, W, 3)
        target_size: Target size (width, height)
        
    Returns:
        Preprocessed image tensor (1, 3, H, W)
    """

def postprocess_segmentation(output: np.ndarray, 
                           original_size: Tuple[int, int]) -> np.ndarray:
    """
    Postprocess segmentation output.
    
    Args:
        output: Model output (1, 4, H, W)
        original_size: Original image size (width, height)
        
    Returns:
        Segmentation mask (H, W) with class IDs
    """
```

### Visualization

```python
def visualize_segmentation(image: np.ndarray, 
                          mask: np.ndarray,
                          alpha: float = 0.5) -> np.ndarray:
    """
    Overlay segmentation mask on image.
    
    Args:
        image: Original RGB image
        mask: Segmentation mask with class IDs
        alpha: Transparency (0.0-1.0)
        
    Returns:
        Visualized image with overlay
    """

def draw_landing_sites(image: np.ndarray, 
                      sites: List[LandingSite]) -> np.ndarray:
    """
    Draw detected landing sites on image.
    
    Args:
        image: Input RGB image
        sites: List of detected landing sites
        
    Returns:
        Annotated image
    """
```

## 🚀 Performance Optimization

### Model Optimization

```python
class OptimizedDetector(UAVLandingDetector):
    """Performance-optimized detector."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Enable TensorRT optimization if available
        if torch.cuda.is_available():
            self._enable_tensorrt()
    
    def _enable_tensorrt(self):
        """Enable TensorRT optimization for NVIDIA GPUs."""
        try:
            import tensorrt as trt
            # TensorRT optimization code
        except ImportError:
            print("TensorRT not available")
```

### Batch Processing

```python
def detect_batch(self, 
                images: List[np.ndarray]) -> List[LandingResult]:
    """
    Process multiple images in a batch for better throughput.
    
    Args:
        images: List of RGB images
        
    Returns:
        List of LandingResult objects
    """
```

## 🔍 Error Handling

### Exception Types

```python
class UAVDetectionError(Exception):
    """Base exception for UAV detection errors."""
    pass

class ModelLoadError(UAVDetectionError):
    """Error loading the model."""
    pass

class InferenceError(UAVDetectionError):
    """Error during model inference."""
    pass

class ImageFormatError(UAVDetectionError):
    """Error with input image format."""
    pass
```

### Error Handling Example

```python
try:
    detector = UAVLandingDetector('model.onnx')
    result = detector.detect_landing_site(image)
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
except ImageFormatError as e:
    print(f"Invalid image format: {e}")
except InferenceError as e:
    print(f"Inference failed: {e}")
```

## ⚙️ Configuration

### Default Configuration

```python
DEFAULT_CONFIG = {
    # Model settings
    'model_path': 'trained_models/ultra_fast_uav_landing.onnx',
    'input_size': (256, 256),
    'device': 'auto',
    
    # Detection thresholds
    'confidence_threshold': 0.5,
    'min_area': 1000,  # pixels
    'max_sites': 10,
    
    # Visualization
    'enable_visualization': True,
    'overlay_alpha': 0.5,
    'show_confidence': True,
    
    # Performance
    'enable_profiling': False,
    'warmup_frames': 5,
}
```

### Custom Configuration

```python
config = {
    'confidence_threshold': 0.7,  # Higher threshold
    'min_area': 2000,            # Larger minimum area
    'device': 'cuda',            # Force GPU
}

detector = UAVLandingDetector(**config)
```

## 🧪 Testing Interface

### Unit Testing

```python
def test_detector_initialization():
    """Test detector initialization."""
    detector = UAVLandingDetector()
    assert detector.model is not None
    assert detector.input_size == (256, 256)

def test_detection_output():
    """Test detection output format."""
    detector = UAVLandingDetector()
    image = np.random.uint8((480, 640, 3)) * 255
    result = detector.detect_landing_site(image)
    
    assert isinstance(result, LandingResult)
    assert result.inference_time > 0
    assert 0 <= result.confidence <= 1
```

### Performance Testing

```python
def benchmark_detector(num_frames: int = 100):
    """Benchmark detector performance."""
    detector = UAVLandingDetector()
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    times = []
    for _ in range(num_frames):
        start = time.time()
        result = detector.detect_landing_site(image)
        times.append(time.time() - start)
    
    print(f"Average inference time: {np.mean(times)*1000:.1f}ms")
    print(f"FPS: {1/np.mean(times):.1f}")
```

---

## 📞 Support

For API questions or issues:
- Check the examples in `examples/`
- Run tests with `python -m pytest tests/`
- Review documentation in `docs/`

**Happy Coding!** 🚁⚡
