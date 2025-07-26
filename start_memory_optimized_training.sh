#!/bin/bash
# Memory-Optimized A100 Training Launcher
# ======================================
# Hardware: 1x A100 SXM + 32 vCPU + 251GB RAM
# Strategy: Load entire dataset to memory to eliminate CPU bottleneck

echo "🚁 MEMORY-OPTIMIZED A100 Training"
echo "================================="
echo "Hardware: 1x A100 SXM + 32 vCPU + 251GB RAM"
echo "Strategy: ENTIRE DATASET IN MEMORY (CPU bottleneck eliminated!)"

# Aggressive cleanup
echo "1. 🧹 Cleaning all processes..."
pkill -f train_a100_progressive_multi_dataset.py
pkill -f python
pkill -f pt_data  # Kill any remaining DataLoader workers
sleep 5

# Memory-optimized system settings
echo "2. 🧠 Memory optimization settings..."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# Minimal CPU threads since data is in memory
export OMP_NUM_THREADS=4  # Minimal for memory-based training
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# A100 SXM + memory optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096,expandable_segments:True
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=0

# Python optimizations
export PYTHONOPTIMIZE=1
export PYTHONUNBUFFERED=1

echo "3. 📊 System status:"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
echo "CPU cores: $(nproc)"
echo "Total RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "Available RAM: $(free -h | grep Mem | awk '{print $7}')"

echo ""
echo "4. 🚀 Starting MEMORY-OPTIMIZED Training:"
echo "   🧠 Datasets: LOADED TO MEMORY (eliminate I/O)"
echo "   🔥 Batch size: 256 (massive GPU utilization)"
echo "   ⚡ Workers: 2 (minimal CPU overhead)"
echo "   🎯 Expected: 95%+ GPU utilization"
echo "   🏎️ Expected speed: 10-15x faster than disk-based"

cd /workspace/uav_landing

# Monitor memory before training
echo ""
echo "5. 📈 Memory status before training:"
free -h

# Start memory-optimized training
echo ""
echo "6. 🚀 Launching training..."
python scripts/train_a100_progressive_multi_dataset.py \
    --stage 1 \
    --sdd_data_root ./datasets/semantic_drone_dataset \
    --dronedeploy_data_root ./datasets/drone_deploy_dataset \
    --udd6_data_root ./datasets/udd6_dataset \
    --use_wandb \
    --stage1_epochs 30 &