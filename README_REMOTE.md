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
