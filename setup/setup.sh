#!/bin/bash

# GPS-Free UAV Landing System - Quick Setup Script
# This script sets up the complete development environment

set -e  # Exit on any error

echo "🚁 GPS-Free UAV Landing System - Quick Setup"
echo "============================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed."
    echo "Please install uv first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo " uv found"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
if [ "$PYTHON_VERSION" != "3.12" ]; then
    echo "⚠️  Python 3.12 not found, creating with uv..."
    uv python install 3.12
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating Python 3.12 virtual environment..."
    uv venv --python 3.12
fi

echo "🔧 Activating environment and installing dependencies..."
source .venv/bin/activate

# Install all required packages
uv pip install --upgrade pip
uv pip install opencv-python>=4.11.0
uv pip install numpy>=1.24.0
uv pip install onnxruntime>=1.16.0
uv pip install torch>=2.0.0
uv pip install matplotlib>=3.7.0
uv pip install Pillow>=10.0.0
uv pip install scipy>=1.11.0

echo " Making scripts executable..."
chmod +x *.py

echo "🧪 Running system tests..."
python test_gps_free.py

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📋 Next Steps:"
echo "1. Activate environment: source .venv/bin/activate"
echo "2. Calibrate camera: python gps_free_main.py --calibrate"
echo "3. Test with webcam: python gps_free_main.py --video 0"
echo "4. Read full guide: cat GPS_FREE_GUIDE.md"
echo ""
echo "🔧 Available Commands:"
echo "  python gps_free_main.py --video test_videos/test_video_mixed.mp4"
echo "  python gps_free_main.py --video 0  # Webcam"
echo "  python gps_free_main.py --calibrate"
echo "  python test_gps_free.py  # Run tests"
echo ""
echo "📖 Controls during operation:"
echo "  'l' - Start landing sequence"
echo "  's' - Stop/Emergency hover"
echo "  'r' - Reset visual odometry"  
echo "  'c' - Camera calibration mode"
echo "  'q' - Quit"
echo ""
echo " System ready for GPS-free UAV landing operations!"
