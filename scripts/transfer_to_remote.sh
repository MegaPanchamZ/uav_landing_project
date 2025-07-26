#!/bin/bash
# Transfer Script for A100 Remote Machine
# =======================================
#
# Transfers only the essential files needed for progressive training:
# - Training scripts
# - Dataset loaders  
# - Model architectures
# - Loss functions
# - Download scripts
#
# Excludes large files, caches, and unnecessary components.

set -e

echo "🚁 UAV Landing System - Remote Transfer Script"
echo "=============================================="

# Configuration
REMOTE_HOST_PORT="${1:-your-a100-server:22}"
REMOTE_USER="${2:-username}"
REMOTE_PATH="${3:-~/uav_landing_training}"
SSH_KEY="${4:-~/.ssh/runpod_key}"

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <remote_host:port> <remote_user> <remote_path> [ssh_key]"
    echo "Example: $0 192.168.1.100:22 ubuntu ~/uav_landing_training"
    echo "Example: $0 216.81.248.126:18310 root ~/uav_landing/ ~/.ssh/runpod_key"
    exit 1
fi

# Parse host and port
IFS=':' read -r REMOTE_HOST REMOTE_PORT <<< "$REMOTE_HOST_PORT"
REMOTE_PORT="${REMOTE_PORT:-22}"

# SSH options for RunPod/cloud providers
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
if [ -f "$SSH_KEY" ]; then
    SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
    echo "🔑 Using SSH key: $SSH_KEY"
else
    echo "⚠️  SSH key not found: $SSH_KEY (will use password auth)"
fi

echo "🎯 Target: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PORT:$REMOTE_PATH"

# Create temporary transfer directory
TEMP_DIR="./transfer_package"
echo "📦 Creating transfer package in: $TEMP_DIR"

rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# Create directory structure
mkdir -p "$TEMP_DIR/scripts"
mkdir -p "$TEMP_DIR/models"
mkdir -p "$TEMP_DIR/datasets"
mkdir -p "$TEMP_DIR/losses"
mkdir -p "$TEMP_DIR/configs"

echo "📋 Copying essential files..."

# Core training scripts
echo "  📄 Training scripts..."
cp "uav_landing_project/scripts/train_a100_progressive_multi_dataset.py" "$TEMP_DIR/scripts/"
cp "uav_landing_project/scripts/download_datasets.sh" "$TEMP_DIR/scripts/"

# Model architectures
echo "  🧠 Model architectures..."
cp "uav_landing_project/models/mobilenetv3_edge_model.py" "$TEMP_DIR/models/"

# Check if enhanced_architectures exists and copy if available
if [ -f "uav_landing_project/models/enhanced_architectures.py" ]; then
    cp "uav_landing_project/models/enhanced_architectures.py" "$TEMP_DIR/models/"
fi

# Dataset loaders
echo "  📊 Dataset loaders..."
cp "uav_landing_project/datasets/semantic_drone_dataset.py" "$TEMP_DIR/datasets/"
cp "uav_landing_project/datasets/dronedeploy_1024_dataset.py" "$TEMP_DIR/datasets/"
cp "uav_landing_project/datasets/udd6_dataset.py" "$TEMP_DIR/datasets/"

# Check for additional dataset files
for dataset_file in "edge_landing_dataset.py" "aerial_semantic_24_dataset.py"; do
    if [ -f "uav_landing_project/datasets/$dataset_file" ]; then
        cp "uav_landing_project/datasets/$dataset_file" "$TEMP_DIR/datasets/"
    fi
done

# Loss functions
echo "  💢 Loss functions..."
if [ -f "uav_landing_project/losses/safety_aware_losses.py" ]; then
    cp "uav_landing_project/losses/safety_aware_losses.py" "$TEMP_DIR/losses/"
else
    # Create a basic loss function if it doesn't exist
    cat > "$TEMP_DIR/losses/safety_aware_losses.py" << 'EOF'
import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedSafetyLoss(nn.Module):
    """Basic combined loss for UAV landing safety."""
    
    def __init__(self, num_classes=6, alpha=0.25, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.focal_loss = self._focal_loss
        self.dice_loss = self._dice_loss
    
    def _focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def _dice_loss(self, pred, target):
        smooth = 1e-5
        pred_soft = F.softmax(pred, dim=1)
        dice_loss = 0
        for i in range(self.num_classes):
            pred_i = pred_soft[:, i]
            target_i = (target == i).float()
            intersection = (pred_i * target_i).sum()
            union = pred_i.sum() + target_i.sum()
            dice_coeff = (2 * intersection + smooth) / (union + smooth)
            dice_loss += 1 - dice_coeff
        return dice_loss / self.num_classes
    
    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return {'total_loss': focal + dice, 'focal': focal, 'dice': dice}
EOF
    echo "  💢 Created basic safety_aware_losses.py"
fi

# Configuration files
echo "  ⚙️  Configuration files..."
if [ -d "uav_landing_project/configs" ]; then
    cp -r uav_landing_project/configs/* "$TEMP_DIR/configs/" 2>/dev/null || true
fi

# Documentation
echo "  📚 Documentation..."
cp "uav_landing_project/EDGE_OPTIMIZED_STRATEGY.md" "$TEMP_DIR/" 2>/dev/null || true
cp "uav_landing_project/LATEST_MULTI_DATASET_STRATEGY.md" "$TEMP_DIR/" 2>/dev/null || true

# Create requirements.txt
echo "  📦 Creating requirements.txt..."
cat > "$TEMP_DIR/requirements.txt" << 'EOF'
# Core ML libraries
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
opencv-python>=4.5.0
pillow>=8.0.0

# Data processing
albumentations>=1.3.0
tqdm>=4.60.0
scikit-learn>=1.0.0

# Monitoring and logging
wandb>=0.15.0
tensorboard>=2.8.0

# Dataset downloads
kaggle>=1.5.0
gdown>=4.6.0

# Utilities
pathlib
argparse
json5
pyyaml
EOF

# Create setup script for remote machine
echo "  🔧 Creating setup script..."
cat > "$TEMP_DIR/setup_remote.sh" << 'EOF'
#!/bin/bash
echo "🚁 Setting up UAV Landing Training Environment"
echo "============================================="

# Install requirements
echo "📦 Installing Python requirements..."
pip install -r requirements.txt

# Make scripts executable
chmod +x scripts/*.sh

# Create output directories
mkdir -p outputs/a100_progressive
mkdir -p datasets

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Set up Kaggle credentials: ~/.kaggle/kaggle.json"
echo "2. Run: bash scripts/download_datasets.sh"
echo "3. Start training: python scripts/train_a100_progressive_multi_dataset.py --stage 1 ..."
EOF

chmod +x "$TEMP_DIR/setup_remote.sh"

# Create main __init__.py files
echo "  🐍 Creating __init__.py files..."
touch "$TEMP_DIR/models/__init__.py"
touch "$TEMP_DIR/datasets/__init__.py" 
touch "$TEMP_DIR/losses/__init__.py"

# Create README for remote setup
echo "  📖 Creating remote README..."
cat > "$TEMP_DIR/README_REMOTE.md" << 'EOF'
# UAV Landing A100 Training

## Quick Start

1. **Setup environment:**
   ```bash
   bash setup_remote.sh
   ```

2. **Configure Kaggle (required):**
   ```bash
   # Get your kaggle.json from https://www.kaggle.com/settings
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Download datasets:**
   ```bash
   bash scripts/download_datasets.sh
   ```

4. **Run progressive training:**
   ```bash
   # Stage 1: Semantic Foundation (SDD)
   python scripts/train_a100_progressive_multi_dataset.py \
       --stage 1 \
       --sdd_data_root ./datasets/semantic_drone_dataset \
       --use_wandb

   # Stage 2: Landing Specialization (DroneDeploy)  
   python scripts/train_a100_progressive_multi_dataset.py \
       --stage 2 \
       --dronedeploy_data_root ./datasets/drone_deploy_dataset \
       --use_wandb

   # Stage 3: Domain Adaptation (UDD6)
   python scripts/train_a100_progressive_multi_dataset.py \
       --stage 3 \
       --udd6_data_root ./datasets/udd6_dataset \
       --use_wandb
   ```

## Features

- ✅ Progressive 3-dataset training strategy
- ✅ MobileNetV3 edge-optimized architecture  
- ✅ A100 GPU optimizations (large batches, mixed precision)
- ✅ Automatic dataset downloading
- ✅ W&B logging and monitoring
- ✅ Large image → chip conversion

## Architecture

- **Stage 1**: Semantic foundation with rich 24→6 class mapping
- **Stage 2**: Landing specialization with native 6 classes
- **Stage 3**: Domain adaptation for high-altitude scenarios
- **Model**: MobileNetV3-Small backbone + lightweight segmentation head
- **Classes**: 6 unified landing classes (ground, vegetation, obstacle, water, vehicle, other)

## Monitoring

Training progress available in Weights & Biases dashboard.
Use `nvidia-smi` to monitor GPU utilization.
EOF

echo "📊 Transfer package summary:"
find "$TEMP_DIR" -type f | wc -l | xargs echo "  Files:"
du -sh "$TEMP_DIR" | cut -f1 | xargs echo "  Size:"

echo ""
echo "🚀 Transferring to remote machine..."

# Create remote directory
ssh -p "$REMOTE_PORT" $SSH_OPTS "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_PATH"

# Check if rsync is available, otherwise use scp
echo "📤 Checking transfer method..."
if ssh -p "$REMOTE_PORT" $SSH_OPTS "$REMOTE_USER@$REMOTE_HOST" "command -v rsync" > /dev/null 2>&1; then
    echo "   Using rsync (faster)..."
    rsync -avz --progress -e "ssh -p $REMOTE_PORT $SSH_OPTS" "$TEMP_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/"
else
    echo "   rsync not available, installing it first..."
    ssh -p "$REMOTE_PORT" $SSH_OPTS "$REMOTE_USER@$REMOTE_HOST" "apt-get update && apt-get install -y rsync" || true
    
    if ssh -p "$REMOTE_PORT" $SSH_OPTS "$REMOTE_USER@$REMOTE_HOST" "command -v rsync" > /dev/null 2>&1; then
        echo "   Using rsync (after install)..."
        rsync -avz --progress -e "ssh -p $REMOTE_PORT $SSH_OPTS" "$TEMP_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/"
    else
        echo "   Using scp (fallback)..."
        scp -P "$REMOTE_PORT" $SSH_OPTS -r "$TEMP_DIR/"* "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/"
    fi
fi

# Run remote setup
echo "🔧 Running remote setup..."
ssh -p "$REMOTE_PORT" $SSH_OPTS "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_PATH && bash setup_remote.sh"

echo ""
echo "✅ Transfer complete!"
echo "=============================="
echo "🎯 Remote location: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
echo ""
echo "📋 Next steps on remote machine:"
echo "1. SSH to remote: ssh -p $REMOTE_PORT $SSH_OPTS $REMOTE_USER@$REMOTE_HOST"
echo "2. Navigate: cd $REMOTE_PATH"
echo "3. Set up Kaggle credentials (see README_REMOTE.md)"
echo "4. Download datasets: bash scripts/download_datasets.sh"
echo "5. Start training: python scripts/train_a100_progressive_multi_dataset.py --stage 1 ..."
echo ""
echo "💡 Use tmux/screen for long training sessions!"

# Cleanup
rm -rf "$TEMP_DIR"
echo "🧹 Cleaned up temporary files" 