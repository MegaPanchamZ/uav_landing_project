#!/bin/bash

# UAV Landing Project - GPU Support Setup
echo "🚀 Setting up GPU support for UAV Landing System"

# Check if NVIDIA GPU is available
if ! nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA GPU not detected. GPU support not available."
    echo "   Using CPU-only mode."
    exit 0
fi

echo " NVIDIA GPU detected"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits

# Check CUDA installation
CUDA_VERSION=""
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo " CUDA toolkit found: $CUDA_VERSION"
else
    echo "⚠️  CUDA toolkit not found. Installing..."
    
    # Check if running on Ubuntu/Debian
    if command -v apt &> /dev/null; then
        echo "📦 Installing CUDA toolkit via apt..."
        sudo apt update
        sudo apt install -y nvidia-cuda-toolkit nvidia-cuda-dev
    else
        echo "❌ Automatic CUDA installation not supported on this system."
        echo "   Please install CUDA toolkit manually from: https://developer.nvidia.com/cuda-downloads"
        echo "   Required version: CUDA 11.8 or 12.x"
        exit 1
    fi
fi

# Set up environment variables
echo "🔧 Setting up CUDA environment..."
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Add to bashrc for persistence
if ! grep -q "CUDA_HOME" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# CUDA Environment" >> ~/.bashrc
    echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
    echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
fi

# Install GPU-enabled ONNX Runtime
echo "📦 Installing ONNX Runtime with GPU support..."
source .venv/bin/activate
pip uninstall -y onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.22.0

# Test GPU support
echo "🧪 Testing GPU support..."
python3 -c "
import onnxruntime as ort
print('Available providers:', ort.get_available_providers())
if 'CUDAExecutionProvider' in ort.get_available_providers():
    print(' CUDA support is available!')
else:
    print('❌ CUDA support not available')
"

echo ""
echo "🎉 GPU setup complete!"
echo "   Restart your terminal or run: source ~/.bashrc"
echo "   Then test with: python3 demo_complete_system.py"
