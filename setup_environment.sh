#!/bin/bash
# Environment Setup Script for UAV Landing Training
# ================================================
#
# Sets up virtual environment and installs all dependencies

set -e  # Exit on any error

echo "🚁 UAV Landing Training Environment Setup"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo "📂 Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Python version check
echo ""
echo "🐍 Python Environment Check"
echo "==========================="
python3 --version || {
    echo "❌ Python3 not found! Please install Python 3.8+"
    exit 1
}

# Check if we're in a container (RunPod) or need virtual environment
if [ -f "/.dockerenv" ] || [ "$CONTAINER" = "docker" ] || [ "$RUNPOD_POD_ID" != "" ]; then
    echo "🐳 Container environment detected - installing globally"
    USE_VENV=false
else
    echo "💻 Local environment detected - using virtual environment"
    USE_VENV=true
fi

# Create and activate virtual environment (if not in container)
if [ "$USE_VENV" = true ]; then
    echo ""
    echo "🌐 Virtual Environment Setup"
    echo "============================"
    
    if [ ! -d "venv" ]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv venv
    else
        echo "✅ Virtual environment already exists"
    fi
    
    echo "🔌 Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    echo "📈 Upgrading pip..."
    pip install --upgrade pip
else
    echo "🐳 Using container Python environment"
fi

# Install packages
echo ""
echo "📦 Package Installation"
echo "======================="

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found!"
    exit 1
fi

echo "📋 Installing packages from requirements.txt..."
pip install -r requirements.txt

# Verify key packages
echo ""
echo "✅ Package Verification"
echo "======================"

echo "🔍 Verifying core packages..."

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || {
    echo "❌ PyTorch installation failed!"
    exit 1
}

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
    python -c "import torch; print(f'Current device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
else
    echo "⚠️  CUDA not available - training will be slower on CPU"
fi

# Check other critical packages
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')" || {
    echo "❌ OpenCV installation failed!"
    exit 1
}

python -c "import numpy; print(f'NumPy: {numpy.__version__}')" || {
    echo "❌ NumPy installation failed!"
    exit 1
}

python -c "import pandas; print(f'Pandas: {pandas.__version__}')" || {
    echo "❌ Pandas installation failed!"
    exit 1
}

python -c "import albumentations; print(f'Albumentations: {albumentations.__version__}')" || {
    echo "❌ Albumentations installation failed!"
    exit 1
}

# Check optional packages
echo ""
echo "🔍 Optional packages:"
python -c "import wandb; print(f'Weights & Biases: {wandb.__version__}')" 2>/dev/null || echo "⚠️  W&B not available"
python -c "import timm; print(f'TIMM: {timm.__version__}')" 2>/dev/null || echo "⚠️  TIMM not available"

# Test dataset loaders
echo ""
echo "🧪 Testing Dataset Loaders"
echo "=========================="

if [ -d "datasets" ]; then
    echo "📂 datasets/ directory found"
    
    # Test imports
    python -c "
try:
    from datasets.semantic_drone_dataset import SemanticDroneDataset
    print('✅ SemanticDroneDataset import OK')
except Exception as e:
    print(f'⚠️  SemanticDroneDataset import failed: {e}')

try:
    from datasets.dronedeploy_1024_dataset import DroneDeploy1024Dataset
    print('✅ DroneDeploy1024Dataset import OK')
except Exception as e:
    print(f'⚠️  DroneDeploy1024Dataset import failed: {e}')

try:
    from datasets.udd6_dataset import UDD6Dataset
    print('✅ UDD6Dataset import OK')
except Exception as e:
    print(f'⚠️  UDD6Dataset import failed: {e}')
"
else
    echo "⚠️  datasets/ directory not found - download datasets first"
fi

# Memory and GPU info
echo ""
echo "💾 System Information"
echo "===================="

echo "🖥️  CPU cores: $(nproc)"
echo "💾 RAM: $(free -h | awk '/^Mem:/ {print $2}')"

if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | while read line; do
        echo "   $line"
    done
else
    echo "⚠️  No GPU detected"
fi

# Environment summary
echo ""
echo "🎉 Environment Setup Complete!"
echo "=============================="

if [ "$USE_VENV" = true ]; then
    echo "📋 To activate environment in future sessions:"
    echo "   source venv/bin/activate"
    echo ""
fi

echo "🚀 Ready for training! Next steps:"
echo "1. Download datasets (if not done):"
echo "   bash scripts/download_datasets_fixed.sh"
echo ""
echo "2. Start progressive training:"
echo "   python scripts/train_a100_progressive_multi_dataset.py --stage 1 --sdd_data_root ./datasets/semantic_drone_dataset --use_wandb"
echo ""
echo "3. Monitor training:"
echo "   - Check W&B dashboard (if configured)"
echo "   - Monitor GPU: watch nvidia-smi"

echo ""
echo "✅ Environment ready for UAV landing training! 🚁" 