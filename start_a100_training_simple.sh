#!/bin/bash
# A100 Training Launcher - CPU Optimized
# =======================================

echo "🚁 A100 Training - CPU Optimized Launch"
echo "======================================="

# Kill any existing processes
echo "1. 🧹 Cleaning up all training processes..."
pkill -f train_a100_progressive_multi_dataset.py
pkill -f pt_data
sleep 5

# Check clean state
echo "2. 🔍 Verifying clean state..."
REMAINING=$(ps aux | grep -c train_a100)
if [ "$REMAINING" -gt 1 ]; then
    echo "⚠️  Warning: $REMAINING training processes still running"
    killall python3
    sleep 3
fi

nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader,nounits

# Set environment for single process training
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "3. 🚀 Starting SINGLE training process..."
echo "   - Batch size: 128 (maximum for A100)"
echo "   - Workers: 4 (CPU optimized)"
echo "   - Single process only"

cd /workspace/uav_landing

# Start single training process
python scripts/train_a100_progressive_multi_dataset.py \
    --stage 1 \
    --sdd_data_root ./datasets/semantic_drone_dataset \
    --dronedeploy_data_root ./datasets/drone_deploy_dataset \
    --udd6_data_root ./datasets/udd6_dataset \
    --use_wandb \
    --stage1_epochs 30 2>&1 | tee training_simple.log &

TRAIN_PID=$!
echo "4. ✅ Training started with PID: $TRAIN_PID"

# Monitor startup
sleep 10
if kill -0 $TRAIN_PID 2>/dev/null; then
    echo "5. 🎉 Training running successfully!"
    echo "   Monitor GPU: watch -n 1 nvidia-smi"
    echo "   Check logs: tail -f training_simple.log"
else
    echo "5. ❌ Training failed - check training_simple.log"
    exit 1
fi 