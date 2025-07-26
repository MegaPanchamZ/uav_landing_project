#!/bin/bash
# A100 Optimized Training Launcher
# ================================

echo "🚁 A100 Progressive Multi-Dataset Training Launcher"
echo "=================================================="

# Kill any existing training processes
echo "1. 🧹 Cleaning up existing processes..."
pkill -f train_a100_progressive_multi_dataset.py
sleep 3

# Check GPU status
echo "2. 🔍 Checking A100 status..."
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits

# Set optimal environment variables for A100
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=0

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_USE_CUDA_DSA=1

echo "3. 🚀 Starting Stage 1 training with A100 optimizations..."
echo "   - Batch size: 64 (4x larger)"
echo "   - Workers: 16 (2x more)"
echo "   - Mixed precision: Enabled"
echo "   - Memory optimization: Enabled"

# Start training with nohup for background execution
cd /workspace/uav_landing

nohup python scripts/train_a100_progressive_multi_dataset.py \
    --stage 1 \
    --sdd_data_root ./datasets/semantic_drone_dataset \
    --dronedeploy_data_root ./datasets/drone_deploy_dataset \
    --udd6_data_root ./datasets/udd6_dataset \
    --use_wandb \
    --stage1_epochs 30 \
    > training_output.log 2>&1 &

TRAIN_PID=$!
echo "4. ✅ Training started with PID: $TRAIN_PID"
echo "   Monitor with: nvidia-smi -l 1"
echo "   Check logs: tail -f training_output.log"
echo "   Check W&B: https://wandb.ai/debkumar269-macquarie-university/uav-a100-progressive"

# Wait a moment and check if training started successfully
sleep 5
if kill -0 $TRAIN_PID 2>/dev/null; then
    echo "5. 🎉 Training is running successfully!"
    echo "   GPU utilization should increase in ~30 seconds"
else
    echo "5. ❌ Training failed to start. Check training_output.log"
    exit 1
fi

echo ""
echo "🔄 To monitor training:"
echo "   nvidia-smi -l 1                    # GPU monitoring"
echo "   tail -f training_output.log        # Training logs"
echo "   ps aux | grep train_a100           # Process status"
echo ""
echo "🛑 To stop training:"
echo "   pkill -f train_a100_progressive_multi_dataset.py" 