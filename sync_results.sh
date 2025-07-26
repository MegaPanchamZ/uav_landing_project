#!/bin/bash

# =============================================================================
# Results Sync Script for A100 Pod Training
# =============================================================================
# This script syncs trained models, logs, and results from A100 pod to local machine
# Usage: ./sync_results.sh [pod_ip] [pod_user] [local_destination]

set -e

# =============================================================================
# Configuration
# =============================================================================
POD_IP="${1:-your_pod_ip}"
POD_USER="${2:-root}"
LOCAL_DEST="${3:-./a100_results}"
POD_PROJECT_DIR="~/uav_landing_system"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔄 A100 Pod Results Sync${NC}"
echo "========================="

# =============================================================================
# Validation
# =============================================================================
if [ "$POD_IP" = "your_pod_ip" ]; then
    echo -e "${RED}❌ Please provide your pod IP address:${NC}"
    echo "Usage: ./sync_results.sh <pod_ip> [pod_user] [local_destination]"
    echo "Example: ./sync_results.sh 192.168.1.100 root ./a100_results"
    exit 1
fi

# Test SSH connection
echo -e "${YELLOW}🔍 Testing SSH connection to $POD_USER@$POD_IP...${NC}"
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$POD_USER@$POD_IP" "echo 'Connection successful'" 2>/dev/null; then
    echo -e "${RED}❌ Cannot connect to $POD_USER@$POD_IP${NC}"
    echo "Please ensure:"
    echo "1. Pod is running and accessible"
    echo "2. SSH key is properly configured"
    echo "3. IP address is correct"
    exit 1
fi
echo -e "${GREEN}✅ SSH connection established${NC}"

# =============================================================================
# Create Local Directory Structure
# =============================================================================
echo -e "${YELLOW}📁 Creating local directory structure...${NC}"
mkdir -p "$LOCAL_DEST"/{models,checkpoints,logs,outputs,datasets,wandb,results}

# =============================================================================
# Sync Functions
# =============================================================================
sync_directory() {
    local remote_dir="$1"
    local local_dir="$2"
    local description="$3"
    
    echo -e "${YELLOW}📥 Syncing $description...${NC}"
    
    # Check if remote directory exists
    if ssh "$POD_USER@$POD_IP" "[ -d '$POD_PROJECT_DIR/$remote_dir' ]"; then
        rsync -avz --progress \
            "$POD_USER@$POD_IP:$POD_PROJECT_DIR/$remote_dir/" \
            "$LOCAL_DEST/$local_dir/"
        echo -e "${GREEN}✅ $description synced${NC}"
    else
        echo -e "${YELLOW}⚠️  Remote directory $remote_dir not found${NC}"
    fi
}

sync_file() {
    local remote_file="$1"
    local local_file="$2"
    local description="$3"
    
    echo -e "${YELLOW}📥 Syncing $description...${NC}"
    
    if ssh "$POD_USER@$POD_IP" "[ -f '$POD_PROJECT_DIR/$remote_file' ]"; then
        rsync -avz --progress \
            "$POD_USER@$POD_IP:$POD_PROJECT_DIR/$remote_file" \
            "$LOCAL_DEST/$local_file"
        echo -e "${GREEN}✅ $description synced${NC}"
    else
        echo -e "${YELLOW}⚠️  Remote file $remote_file not found${NC}"
    fi
}

# =============================================================================
# Primary Sync Operations
# =============================================================================
echo -e "${BLUE}🚀 Starting sync operations...${NC}"

# 1. Sync trained models
sync_directory "outputs" "models" "Trained Models"

# 2. Sync checkpoints
sync_directory "checkpoints" "checkpoints" "Training Checkpoints"

# 3. Sync logs
sync_directory "logs" "logs" "Training Logs"

# 4. Sync Weights & Biases logs
sync_directory "wandb" "wandb" "W&B Logs"

# 5. Sync training outputs
sync_directory "results" "results" "Training Results"

# 6. Sync specific important files
echo -e "${YELLOW}📄 Syncing important files...${NC}"
important_files=(
    "training_history.json:training_history.json"
    "dataset_info.txt:datasets/dataset_info.txt"
    "gpu_setup.py:gpu_setup.py"
)

for file_pair in "${important_files[@]}"; do
    remote_file="${file_pair%%:*}"
    local_file="${file_pair##*:}"
    sync_file "$remote_file" "$local_file" "$(basename $remote_file)"
done

# =============================================================================
# Sync Training Metrics and Performance Data
# =============================================================================
echo -e "${YELLOW}📊 Collecting training metrics...${NC}"

# Create a comprehensive training report
ssh "$POD_USER@$POD_IP" "cd $POD_PROJECT_DIR && python3 -c \"
import json
import os
from pathlib import Path
from datetime import datetime

# Collect system info
import torch
import psutil

report = {
    'sync_timestamp': datetime.now().isoformat(),
    'system_info': {
        'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'None',
        'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
        'cpu_cores': psutil.cpu_count(),
        'ram_gb': psutil.virtual_memory().total / 1e9
    },
    'training_status': {},
    'file_sizes': {}
}

# Check for training files and their sizes
training_files = [
    'outputs/final_model.pth',
    'outputs/training_history.json',
    'checkpoints/best_checkpoint.pth',
    'checkpoints/latest_checkpoint.pth'
]

for file_path in training_files:
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / 1e6
        report['file_sizes'][file_path] = f'{size_mb:.1f} MB'
        report['training_status'][file_path] = 'exists'
    else:
        report['training_status'][file_path] = 'missing'

# Load training history if available
if os.path.exists('outputs/training_history.json'):
    try:
        with open('outputs/training_history.json', 'r') as f:
            history = json.load(f)
            if history:
                last_epoch = history[-1]
                report['last_epoch_metrics'] = last_epoch
                report['total_epochs'] = len(history)
                
                # Find best metrics
                best_val_miou = max([epoch.get('val_miou', 0) for epoch in history])
                report['best_val_miou'] = best_val_miou
    except:
        pass

# Save report
with open('sync_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print('Training report generated')
\"" 2>/dev/null || echo "Could not generate training report"

# Sync the report
sync_file "sync_report.json" "sync_report.json" "Training Report"

# =============================================================================
# Create Local Summary
# =============================================================================
echo -e "${YELLOW}📋 Creating local summary...${NC}"

cat > "$LOCAL_DEST/README.md" << EOF
# A100 Pod Training Results

Synced on: $(date)
Pod: $POD_USER@$POD_IP

## Directory Structure

\`\`\`
$(find "$LOCAL_DEST" -type d | head -20 | sed "s|$LOCAL_DEST|.|g")
\`\`\`

## Files Summary

\`\`\`
$(find "$LOCAL_DEST" -name "*.pth" -o -name "*.json" -o -name "*.log" | wc -l) total files synced
$(find "$LOCAL_DEST" -name "*.pth" | wc -l) model files
$(find "$LOCAL_DEST" -name "*.json" | wc -l) JSON files
$(find "$LOCAL_DEST" -name "*.log" | wc -l) log files
\`\`\`

## Model Files

$(find "$LOCAL_DEST" -name "*.pth" -exec ls -lh {} \; | awk '{print $9, $5}' || echo "No model files found")

## Quick Commands

\`\`\`bash
# View training history
cat models/training_history.json | jq '.[-1]'  # Last epoch

# Load best model in Python
python3 -c "import torch; model = torch.load('checkpoints/best_checkpoint.pth')"

# Check W&B logs
ls wandb/

# View sync report
cat sync_report.json | jq '.'
\`\`\`

## Next Steps

1. Analyze training results: \`python analyze_results.py\`
2. Evaluate model performance: \`python evaluate_model.py\`
3. Deploy model: \`python deploy_model.py\`
EOF

# =============================================================================
# Performance Analysis
# =============================================================================
echo -e "${YELLOW}⚡ Analyzing sync performance...${NC}"

# Calculate total synced data
TOTAL_SIZE=$(du -sh "$LOCAL_DEST" 2>/dev/null | cut -f1 || echo "Unknown")
FILE_COUNT=$(find "$LOCAL_DEST" -type f | wc -l)

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${GREEN}🎉 Sync Complete!${NC}"
echo "=================="
echo ""
echo -e "${BLUE}📊 Sync Summary:${NC}"
echo "   📁 Local destination: $LOCAL_DEST"
echo "   📦 Total size: $TOTAL_SIZE"
echo "   📄 Files synced: $FILE_COUNT"
echo "   🎯 Pod: $POD_USER@$POD_IP"
echo ""
echo -e "${BLUE}📋 What was synced:${NC}"
echo "   ✅ Trained models (outputs/)"
echo "   ✅ Training checkpoints (checkpoints/)"
echo "   ✅ Training logs (logs/)"
echo "   ✅ W&B experiment data (wandb/)"
echo "   ✅ Training metrics and history"
echo "   ✅ System and performance reports"
echo ""
echo -e "${BLUE}🔗 Next Steps:${NC}"
echo "   1. Review: cat $LOCAL_DEST/README.md"
echo "   2. Check training: cat $LOCAL_DEST/sync_report.json | jq '.'"
echo "   3. Load model: python3 -c \"import torch; torch.load('$LOCAL_DEST/checkpoints/best_checkpoint.pth')\""
echo "   4. Analyze results: cd $LOCAL_DEST && python analyze_results.py"
echo ""
echo -e "${GREEN}✅ All results successfully synced from A100 pod!${NC}"

# =============================================================================
# Optional: Create analysis script
# =============================================================================
cat > "$LOCAL_DEST/analyze_results.py" << 'EOF'
#!/usr/bin/env python3
"""
Quick analysis script for A100 training results
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_training():
    """Analyze training results"""
    
    # Load training history
    history_file = Path("models/training_history.json")
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
        
        print(f"📊 Training completed: {len(history)} epochs")
        
        if history:
            last_epoch = history[-1]
            print(f"🏆 Final validation mIoU: {last_epoch.get('val_miou', 'N/A')}")
            print(f"📉 Final loss: {last_epoch.get('loss', 'N/A')}")
            
            # Plot training curves
            epochs = range(len(history))
            losses = [epoch.get('loss', 0) for epoch in history]
            val_mious = [epoch.get('val_miou', 0) for epoch in history]
            
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(epochs, losses)
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            
            plt.subplot(1, 2, 2)
            plt.plot(epochs, val_mious)
            plt.title('Validation mIoU')
            plt.xlabel('Epoch')
            plt.ylabel('mIoU')
            
            plt.tight_layout()
            plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
            print("📈 Training curves saved to training_curves.png")
    
    # Load sync report
    report_file = Path("sync_report.json")
    if report_file.exists():
        with open(report_file) as f:
            report = json.load(f)
        
        print(f"\n🖥️  System Info:")
        sys_info = report.get('system_info', {})
        print(f"   GPU: {sys_info.get('gpu_name', 'Unknown')}")
        print(f"   GPU Memory: {sys_info.get('gpu_memory_gb', 0):.1f} GB")
        print(f"   CPU Cores: {sys_info.get('cpu_cores', 'Unknown')}")
        print(f"   RAM: {sys_info.get('ram_gb', 0):.1f} GB")

if __name__ == "__main__":
    analyze_training()
EOF

chmod +x "$LOCAL_DEST/analyze_results.py"

echo ""
echo -e "${GREEN}📊 Analysis script created: $LOCAL_DEST/analyze_results.py${NC}" 