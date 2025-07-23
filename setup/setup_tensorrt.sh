#!/bin/bash

# TensorRT Setup Script for UAV Landing System
# Optimizes NVIDIA GPU performance with TensorRT acceleration

set -e

echo "🚀 TensorRT Setup for UAV Landing System"
echo "========================================"

# Check NVIDIA GPU
echo "1. Checking NVIDIA GPU..."
if ! nvidia-smi > /dev/null 2>&1; then
    echo "❌ No NVIDIA GPU detected. TensorRT requires NVIDIA GPU."
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
echo " Detected GPU: $GPU_INFO"

# Check CUDA version
echo ""
echo "2. Checking CUDA version..."
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9.]*\).*/\1/')
echo " CUDA Driver: $CUDA_VERSION"

# Determine CUDA major version for TensorRT compatibility
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)

echo ""
echo "3. TensorRT Installation Options:"
echo "   A) Auto-install via pip (recommended for CUDA 12.x)"
echo "   B) Manual download from NVIDIA"
echo "   C) Cancel"
echo ""
read -p "Choose option (A/B/C): " choice

case $choice in
    [Aa]* )
        echo ""
        echo "Installing TensorRT via pip..."
        
        # Activate virtual environment
        cd /home/mpz/development/playground/uav_landing_project
        source .venv/bin/activate
        
        # Install TensorRT based on CUDA version
        if [ "$CUDA_MAJOR" -ge "12" ]; then
            echo "Installing TensorRT for CUDA 12.x..."
            pip install tensorrt
        elif [ "$CUDA_MAJOR" -eq "11" ]; then
            echo "Installing TensorRT for CUDA 11.x..."
            pip install tensorrt-cu11
        else
            echo "⚠️  CUDA version $CUDA_VERSION may not be supported"
            pip install tensorrt
        fi
        
        # Install additional dependencies
        pip install pycuda
        
        echo ""
        echo "4. Testing TensorRT installation..."
        python3 -c "
import tensorrt as trt
print(f'TensorRT version: {trt.__version__}')

# Test UAV system with TensorRT
from src.uav_landing_detector import UAVLandingDetector
detector = UAVLandingDetector(device='auto')
print(f'Detector using: {detector.actual_device}')
"
        
        if [ $? -eq 0 ]; then
            echo " TensorRT installation successful!"
            echo ""
            echo "Performance comparison:"
            echo "  🐢 CPU: ~20-25 FPS"
            echo "  🚀 CUDA: ~40-60 FPS"  
            echo "  🏎️  TensorRT: ~60-120 FPS"
        else
            echo "❌ TensorRT installation failed"
        fi
        ;;
    [Bb]* )
        echo ""
        echo "Manual TensorRT Installation:"
        echo "1. Download TensorRT from: https://developer.nvidia.com/tensorrt"
        echo "2. Choose TensorRT 8.6+ for CUDA $CUDA_VERSION"
        echo "3. Extract and follow NVIDIA installation guide"
        echo "4. Set LD_LIBRARY_PATH to include TensorRT libraries"
        echo ""
        echo "Example:"
        echo "  export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/path/to/TensorRT/lib"
        ;;
    [Cc]* )
        echo "Installation cancelled."
        ;;
    * )
        echo "Invalid option. Installation cancelled."
        ;;
esac

echo ""
echo "5. System Status Check:"
cd /home/mpz/development/playground/uav_landing_project
source .venv/bin/activate > /dev/null 2>&1 || true

python3 -c "
import onnxruntime as ort
print(f'Available providers: {ort.get_available_providers()[:3]}')

try:
    from src.uav_landing_detector import UAVLandingDetector
    detector = UAVLandingDetector(device='auto')
    print(f'UAV System Status: {detector.actual_device} acceleration')
except Exception as e:
    print(f'UAV System Error: {e}')
"

echo ""
echo " Setup Complete!"
echo ""
echo "Next steps:"
echo "  - Run 'python3 uav_landing_main.py' to test the system"
echo "  - Use 'python3 examples/demo.py' for interactive demo" 
echo "  - Check performance with different device modes"
