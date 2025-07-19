# 📖 API Reference

Complete API documentation for the Ultra-Fast UAV Landing Detection system.

## 🏗️ Core Classes

### UAVLandingDetector

Main detector class for landing site detection.

```python
class UAVLandingDetector:
    """Ultra-fast UAV landing site detector using semantic segmentation."""
    
    def __init__(self, 
                 model_path: str = "trained_models/ultra_fast_uav_landing.onnx",
                 device: str = "auto",
                 input_size: Tuple[int, int] = (256, 256),
                 confidence_threshold: float = 0.5):
        """
        Initialize the UAV landing detector.
        
        Args:
            model_path: Path to ONNX model file
            device: Device to run inference ('cpu', 'cuda', 'auto')
            input_size: Input image size (width, height)
            confidence_threshold: Minimum confidence for landing sites
        """
```

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

## 🎯 Landing Classes

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
