#!/bin/bash
# Enhanced A100 Training with W&B Tracking
# ========================================

echo "🚁 Enhanced A100 Training with Comprehensive W&B Tracking"
echo "========================================================="

# Clean environment
pkill -f train_a100_progressive_multi_dataset.py
sleep 3

# System info
echo "📊 System Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

# Start enhanced training
echo ""
echo "🚀 Starting Enhanced Training with:"
echo "   ✅ Comprehensive W&B metrics"
echo "   ✅ Per-class mIoU tracking" 
echo "   ✅ Regular model checkpointing"
echo "   ✅ Hardware monitoring"
echo "   ✅ Batch time optimization"
echo "   ✅ CPU-optimized data loading"

cd /workspace/uav_landing

python scripts/train_a100_progressive_multi_dataset.py \
    --stage 1 \
    --sdd_data_root ./datasets/semantic_drone_dataset \
    --dronedeploy_data_root ./datasets/drone_deploy_dataset \
    --udd6_data_root ./datasets/udd6_dataset \
    --use_wandb \
    --stage1_epochs 30

echo "🎯 Training complete! Check W&B dashboard for detailed metrics." 