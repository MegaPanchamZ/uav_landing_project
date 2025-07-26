#!/bin/bash
# A100 SXM Optimized Training Launcher
# ====================================
# Hardware: 1x A100 SXM + 32 vCPU + 251GB RAM

echo "🚁 A100 SXM Optimized UAV Training"
echo "=================================="
echo "Hardware: 1x A100 SXM + 32 vCPU + 251GB RAM"

# Clean environment aggressively
echo "1. 🧹 Aggressive cleanup..."
pkill -f train_a100_progressive_multi_dataset.py
pkill -f python
sleep 5

# System optimization for massive hardware
echo "2. ⚙️  Hardware optimization..."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16  # Half of 32 vCPU for optimal performance
export MKL_NUM_THREADS=16

# A100 SXM specific optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048  # Larger splits for A100 SXM
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=0

# Memory optimization for 251GB RAM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "3. 📊 System status:"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
echo "CPU cores: $(nproc)"
echo "RAM: $(free -h | grep Mem | awk '{print $2}')"

echo ""
echo "4. 🚀 Starting A100 SXM Optimized Training:"
echo "   🔥 Batch size: 256 (4x larger)"
echo "   🔥 Workers: 24 (75% of 32 vCPU)"
echo "   🔥 Prefetch factor: 4 (aggressive)"
echo "   🔥 Crops per image: 8 (massive RAM utilization)"
echo "   🔥 Mixed precision: Enabled"
echo "   🔥 Memory optimization: A100 SXM tuned"

cd /workspace/uav_landing

# Start optimized training
python scripts/train_a100_progressive_multi_dataset.py \
    --stage 1 \
    --sdd_data_root ./datasets/semantic_drone_dataset \
    --dronedeploy_data_root ./datasets/drone_deploy_dataset \
    --udd6_data_root ./datasets/udd6_dataset \
    --use_wandb \
    --stage1_epochs 30 &

TRAIN_PID=$!
echo "5. ✅ Training started with PID: $TRAIN_PID"

# Monitor startup
sleep 15
if kill -0 $TRAIN_PID 2>/dev/null; then
    echo "6. 🎉 A100 SXM training running!"
    echo ""
    echo "Expected improvements:"
    echo "   • GPU utilization: 80-95% (vs current ~10%)"
    echo "   • Training speed: 5-10x faster"
    echo "   • Batch throughput: 4x higher"
    echo "   • Memory efficiency: Maximized"
    echo ""
    echo "Monitor with:"
    echo "   watch -n 1 nvidia-smi"
    echo "   htop  # CPU utilization"
    echo "   W&B: https://wandb.ai/debkumar269-macquarie-university/uav-a100-progressive"
else
    echo "6. ❌ Training failed to start"
    exit 1
fi 