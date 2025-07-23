# UAV Landing Detection Fine-Tuning

This directory contains practical tools for fine-tuning a BiSeNetV2 model for UAV landing zone detection using your DroneDeploy dataset.

## Quick Start

### 1. **Quick Setup and Training**
```bash
cd training_tools/
python quick_setup.py --data_path ../datasets/drone_deploy_dataset_intermediate/dataset-medium
```

This will:
- Install required packages 
- Check your dataset structure
- Find your pre-trained BiSeNetV2 model automatically
- Start fine-tuning with sensible defaults

### 2. **Custom Training**
```bash
python practical_fine_tuning.py \
    --data_path ../datasets/drone_deploy_dataset_intermediate/dataset-medium \
    --pretrained_model ../model_pths/bisenetv2_fcn_fp16_4x4_1024x1024_160k_cityscapes_20210902_045942-b979777b.pth \
    --epochs 50 \
    --batch_size 8 \
    --export_onnx
```

### 3. **Visualize Your Dataset**
```bash
# View dataset samples
python visualize.py --action dataset --data_path ../datasets/drone_deploy_dataset_intermediate/dataset-medium

# Analyze dataset statistics  
python visualize.py --action stats --data_path ../datasets/drone_deploy_dataset_intermediate/dataset-medium

# View training progress
python visualize.py --action history --history_path ./fine_tuned_models/training_history.json

# See model predictions
python visualize.py --action predictions --model_path ./fine_tuned_models/best_model.pth
```

## What This Does

### Class Mapping
The system converts DroneDeploy's 7 RGB color classes into 4 landing-relevant classes:

| DroneDeploy Class | RGB Color | → | Landing Class | Description |
|------------------|-----------|---|---------------|-------------|
| GROUND | (255,255,255) | → | **Suitable** | Safe landing zones |
| VEGETATION | (75,180,60) | → | **Suitable** | Flat grass areas |
| BUILDING | (75,25,230) | → | **Obstacle** | Buildings to avoid |
| CAR | (200,130,0) | → | **Obstacle** | Vehicles to avoid |
| WATER | (48,130,245) | → | **Unsafe** | Water bodies |
| CLUTTER | (180,30,145) | → | **Unsafe** | Debris/unclear areas |
| IGNORE | (255,0,255) | → | **Background** | Ignored in training |

### Training Process
1. **Loads your pre-trained BiSeNetV2 model** (Cityscapes weights)
2. **Replaces the final classifier** for 4 UAV landing classes
3. **Fine-tunes with smart learning rates** (lower for backbone, higher for classifier)
4. **Uses aerial-specific augmentations** (rotations, flips)
5. **Exports to ONNX** for deployment

### Key Features
-  **Automatic data loading** from your DroneDeploy structure
-  **Smart RGB→class conversion** 
-  **Pre-trained weight loading** with automatic adaptation
-  **Memory-efficient training** with gradient clipping
-  **Comprehensive monitoring** with mIoU validation
-  **ONNX export** for deployment

## File Structure

```
training_tools/
├── practical_fine_tuning.py    # Main training script
├── quick_setup.py               # Easy setup and launcher  
├── visualize.py                 # Dataset and result visualization
├── dataset_preparation.py       # Advanced dataset tools
├── fine_tuning_pipeline.py      # Full 3-stage pipeline
└── README.md                    # This file
```

## Common Issues & Solutions

### 1. **Out of Memory**
```bash
# Reduce batch size
python quick_setup.py --batch_size 2

# Or use CPU
python quick_setup.py --device cpu
```

### 2. **No Images Found**
Check your dataset structure:
```
dataset-medium/
├── images/           # Contains *.tif files
└── labels/          # Contains *.png files  
```

### 3. **Can't Find Pre-trained Model**
Place your BiSeNetV2 `.pth` file in `../model_pths/` or specify manually:
```bash
python practical_fine_tuning.py --pretrained_model /path/to/your/model.pth
```

### 4. **Quick Test Run**
```bash
# Fast 5-epoch test with tiny batch size
python quick_setup.py --quick_test
```

## Results

After training, you'll find:
- `fine_tuned_models/best_model.pth` - Best performing model
- `fine_tuned_models/training_history.json` - Training metrics
- `fine_tuned_models/bisenetv2_uav_landing.onnx` - Deployment-ready model

## Next Steps

1. **Test the trained model** with your demo script
2. **Integrate ONNX model** into your real-time detection pipeline
3. **Fine-tune further** with real flight data if available

## Advanced Usage

### Using the Full 3-Stage Pipeline
```bash
# Stage 1: DroneDeploy general aerial training
# Stage 2: UDD-6 low-altitude specialization  
# Stage 3: Task-specific landing optimization
python fine_tuning_pipeline.py --dronedeploy_path ../datasets/drone_deploy_dataset_intermediate/dataset-medium
```

### Custom Dataset Preparation
```bash
# Prepare datasets with specific formats
python dataset_preparation.py --prepare dronedeploy --data_root ../datasets/
```

## Performance Tips

- **Use GPU** if available (automatically detected)
- **Start with small batch size** (4-8) and increase if memory allows
- **Monitor validation mIoU** - should improve steadily
- **Export to ONNX** for 2-3x faster inference than PyTorch
