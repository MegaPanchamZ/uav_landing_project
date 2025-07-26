#!/bin/bash

# =============================================================================
# A100 GPU Pod Setup Script for UAV Landing Detection Training
# =============================================================================
# This script sets up the complete environment on your A100 pod instance
# Run this script first after SSH-ing into your pod

set -e  # Exit on any error

echo "🚀 Setting up A100 GPU Pod for UAV Landing Training..."

# =============================================================================
# 1. Basic Dependencies (RunPod usually has these pre-installed)
# =============================================================================
echo "📦 Checking system packages..."
# Most packages are pre-installed on RunPod, so we skip apt commands

# =============================================================================
# 2. NVIDIA Drivers and CUDA Verification
# =============================================================================
echo "🔧 Verifying NVIDIA GPU setup..."
nvidia-smi
nvcc --version || echo "⚠️  NVCC not found - CUDA toolkit may need installation"

# =============================================================================
# 3. Project Setup
# =============================================================================
echo "📁 Setting up project directory..."
cd ~
if [ ! -d "uav_landing_system" ]; then
    mkdir -p uav_landing_system
fi
cd uav_landing_system

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# =============================================================================
# 4. PyTorch and GPU Dependencies
# =============================================================================
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# =============================================================================
# 5. Core Training Dependencies
# =============================================================================
echo "📚 Installing training dependencies..."

# Create requirements file
cat > requirements.txt << 'EOF'
# Core ML libraries
opencv-python>=4.11.0
numpy>=2.0.0
albumentations>=1.4.0
matplotlib>=3.8.0
seaborn>=0.12.0
Pillow>=10.0.0

# Training utilities
tqdm>=4.64.0
wandb>=0.16.0
tensorboard>=2.15.0
scikit-learn>=1.3.0
pandas>=2.0.0

# GPU optimization
onnxruntime-gpu>=1.19.0

# Kaggle API
kaggle>=1.6.0

# System monitoring
psutil>=5.9.0
gpustat>=1.1.0

# Development tools
jupyter>=1.0.0
ipywidgets>=8.0.0
EOF

# Install from requirements file
pip install -r requirements.txt

# =============================================================================
# 6. Kaggle API Setup
# =============================================================================
echo "🔑 Setting up Kaggle API..."
echo "Please ensure you have your kaggle.json file ready to upload!"
echo "You can download it from: https://www.kaggle.com/settings -> Create New API Token"

# Create kaggle directory
mkdir -p ~/.kaggle
chmod 700 ~/.kaggle

echo "📋 Place your kaggle.json file in ~/.kaggle/ directory"
echo "Run: chmod 600 ~/.kaggle/kaggle.json"

# =============================================================================
# 7. Create Directory Structure
# =============================================================================
echo "📂 Creating project directory structure..."
mkdir -p {datasets,models,outputs,logs,checkpoints,results}

# =============================================================================
# 8. GPU Memory and Performance Setup
# =============================================================================
echo "⚡ Configuring GPU settings for A100..."

# Create GPU optimization script
cat > gpu_setup.py << 'EOF'
import torch
import gc

def setup_a100_optimizations():
    """Optimize PyTorch for A100 GPU training"""
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Enable memory fraction usage
        torch.cuda.empty_cache()
        gc.collect()
        
        # Set memory optimization
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # A100 specific optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        print("✅ A100 optimizations applied!")
        return True
    else:
        print("❌ No GPU detected!")
        return False

if __name__ == "__main__":
    setup_a100_optimizations()
EOF

python gpu_setup.py

# =============================================================================
# 9. Training Environment Verification
# =============================================================================
echo "🧪 Testing training environment..."
python -c "
import torch
import torchvision
import cv2
import numpy as np
import albumentations
print('✅ PyTorch:', torch.__version__)
print('✅ TorchVision:', torchvision.__version__)
print('✅ CUDA Available:', torch.cuda.is_available())
print('✅ CUDA Version:', torch.version.cuda)
print('✅ GPU Count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('✅ GPU Name:', torch.cuda.get_device_name(0))
    print('✅ GPU Memory:', f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
print('✅ OpenCV:', cv2.__version__)
print('✅ NumPy:', np.__version__)
print('✅ Albumentations:', albumentations.__version__)
"

# =============================================================================
# 10. Create tmux session for long training
# =============================================================================
echo "📺 Setting up tmux for long training sessions..."
cat > start_training_session.sh << 'EOF'
#!/bin/bash
# Create a tmux session for training
tmux new-session -d -s uav_training
tmux send-keys -t uav_training "cd ~/uav_landing_system && source venv/bin/activate" Enter
tmux send-keys -t uav_training "echo 'UAV Training Session Ready!'" Enter
tmux send-keys -t uav_training "clear" Enter
echo "🎯 Training session created! Connect with: tmux attach -t uav_training"
EOF
chmod +x start_training_session.sh

# =============================================================================
# 11. System Monitoring Setup
# =============================================================================
echo "📊 Setting up system monitoring..."
cat > monitor_training.sh << 'EOF'
#!/bin/bash
# Monitor GPU and training progress
echo "🔍 GPU Monitoring Dashboard"
echo "=========================="
watch -n 2 '
echo "🚀 GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
echo ""
echo "💾 Memory Usage:"
free -h
echo ""
echo "🔥 Top GPU Processes:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
'
EOF
chmod +x monitor_training.sh

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "🎉 A100 GPU Pod Setup Complete!"
echo "================================"
echo ""
echo "📋 Next Steps:"
echo "1. Upload your kaggle.json to ~/.kaggle/ and set permissions:"
echo "   chmod 600 ~/.kaggle/kaggle.json"
echo ""
echo "2. Download datasets:"
echo "   source venv/bin/activate"
echo "   ./download_datasets.sh"
echo ""
echo "3. Start training:"
echo "   ./start_training_session.sh"
echo "   tmux attach -t uav_training"
echo "   python train_a100.py"
echo ""
echo "4. Monitor training (in another terminal):"
echo "   ./monitor_training.sh"
echo ""
echo "🔗 Useful Commands:"
echo "   tmux list-sessions    # List active sessions"
echo "   tmux attach -t uav_training  # Attach to training session"
echo "   Ctrl+B then D         # Detach from tmux session"
echo "   nvidia-smi           # Check GPU status"
echo "   htop                 # Check CPU usage"
echo ""
echo "✅ Environment ready for UAV landing detection training!" 