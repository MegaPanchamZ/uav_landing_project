# GPU Support Setup Guide

## Current Status
✅ NVIDIA GPU detected: RTX 4060 Ti
✅ TensorRT provider available (Optimal)
✅ CUDA provider available (Fast)  
✅ ONNX Runtime GPU support available
⚠️  Runtime libraries may need installation

## Performance Hierarchy

1. **TensorRT** (~60-120 FPS) - Optimal performance with graph optimization
2. **CUDA** (~40-60 FPS) - Standard GPU acceleration  
3. **CPU** (~20-25 FPS) - Compatible fallback (Current)

## Quick Fix Options

### Option 1: Use CPU (Current - Working)
No action needed. The system runs efficiently on CPU.

### Option 2: Enable TensorRT (Optimal Performance)

1. **Install TensorRT 8.6+ for CUDA 12.x:**
   ```bash
   # Download TensorRT from NVIDIA Developer
   # https://developer.nvidia.com/tensorrt
   
   # Or install via conda/pip (if available):
   pip install tensorrt
   ```

2. **Install CUDA Runtime (if not present):**
   ```bash
   sudo apt update
   sudo apt install nvidia-cuda-toolkit
   ```

3. **Test TensorRT:**
   ```bash
   python3 -c "
   from src.uav_landing_detector import UAVLandingDetector
   detector = UAVLandingDetector(device='tensorrt')
   print('TensorRT Status:', detector.actual_device)
   "
   ```

### Option 3: Enable CUDA Only

1. **Install CUDA 12.x runtime:**
   ```bash
   # Ubuntu/Debian:
   sudo apt update
   sudo apt install nvidia-cuda-toolkit
   
   # Or download from: https://developer.nvidia.com/cuda-downloads
   ```

2. **Install cuDNN (if needed):**
   ```bash
   # Download from: https://developer.nvidia.com/cudnn
   # Follow NVIDIA installation guide
   ```

3. **Verify installation:**
   ```bash
   cd /home/mpz/development/playground/uav_landing_project
   source .venv/bin/activate
   python3 -c "
   import onnxruntime as ort
   print('Providers:', ort.get_available_providers())
   
   # Test GPU
   from src.uav_landing_detector import UAVLandingDetector  
   detector = UAVLandingDetector(device='auto')
   "
   ```

## Performance Comparison

- **TensorRT Mode**: ~60-120 FPS (Optimal - requires TensorRT installation)  
- **CUDA Mode**: ~40-60 FPS (Fast - requires CUDA libraries)
- **CPU Mode**: ~20-25 FPS (Compatible - current fallback)

Both GPU modes support the full neuro-symbolic UAV landing system with Scallop integration.

## Quick Setup Commands

```bash
# Check current system
python3 -c "from src.uav_landing_detector import UAVLandingDetector; d=UAVLandingDetector(); print(f'Current: {d.actual_device}')"

# Install TensorRT (optimal performance)  
./setup_tensorrt.sh

# Benchmark performance
python3 benchmark_gpu.py

# Test specific modes
python3 -c "from src.uav_landing_detector import UAVLandingDetector; UAVLandingDetector(device='tensorrt')"
```

## Alternative: Install CPU-only version

If you prefer to remove GPU warnings entirely:

```bash
cd /home/mpz/development/playground/uav_landing_project
source .venv/bin/activate
pip uninstall onnxruntime-gpu
pip install onnxruntime==1.22.0
```

## Status Check

Run this to check your current setup:
```bash
python3 -c "
import onnxruntime as ort
from src.uav_landing_detector import UAVLandingDetector
print('ONNX Version:', ort.__version__)
print('Providers:', ort.get_available_providers())
detector = UAVLandingDetector()
print('System ready!')
"
```
