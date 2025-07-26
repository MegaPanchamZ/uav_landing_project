# UAV Landing System - Repository Guide

This document provides a comprehensive guide to the UAV Landing System repository, explaining the codebase structure, implementation details, and development workflow.

## 📁 Repository Structure

```
uav_landing_project/
├── 🚀 Core Scripts
│   ├── train.py                    # Universal training with auto-hardware detection
│   ├── test.py                     # Comprehensive model evaluation
│   ├── download_datasets.py        # Automated dataset management
│   └── demo_neuro_symbolic.py      # Neuro-symbolic reasoning demo
│
├── 🧠 Neural Network Models
│   ├── models/
│   │   ├── mobilenetv3_edge_model.py      # Edge-optimized MobileNetV3
│   │   └── enhanced_architectures.py      # Advanced model variants
│   └── losses/
│       └── safety_aware_losses.py         # Multi-component loss functions
│
├── 📊 Dataset Implementations
│   └── datasets/
│       ├── semantic_drone_dataset.py      # SDD with 24→6 class mapping
│       ├── dronedeploy_1024_dataset.py    # DroneDeploy with large patches
│       ├── udd6_dataset.py                # Urban drone dataset
│       ├── cached_augmentation.py         # Memory-optimized augmentation
│       ├── enhanced_augmentation.py       # Advanced augmentation pipeline
│       └── ...                            # Additional dataset utilities
│
├── 🔗 Neuro-Symbolic Reasoning
│   └── scallop_integration/
│       └── landing_rules.scl              # Scallop logical rules
│
├── ⚙️ Configuration & Utilities
│   ├── configs/
│   │   ├── resolution_profiles.json       # Multi-resolution training configs
│   │   └── training/                      # Training configuration templates
│   ├── requirements.txt                   # Python dependencies
│   └── pyproject.toml                     # Project metadata
│
├── 📋 Documentation
│   ├── README.md                          # Main project documentation
│   ├── REPOSITORY_GUIDE.md               # This file
│   └── docs/                              # Detailed API and architecture docs
│       ├── API.md
│       ├── ARCHITECTURE.md
│       └── DATASETS.md
│
├── 🎬 Demonstrations
│   └── demos/
│       ├── demo_complete_system.py        # End-to-end system demo
│       ├── demo_neuro_symbolic_complete.py # Full neuro-symbolic pipeline
│       └── ...                            # Additional demo scripts
│
└── 🗂️ Outputs & Results
    ├── outputs/                           # Training checkpoints and logs
    ├── test_results/                      # Evaluation results and visualizations
    └── wandb/                             # W&B experiment logs
```

## 🚀 Core Scripts Deep Dive

### 1. `train.py` - Universal Training Script

**Purpose**: Single, generalized training script that works on any hardware configuration.

**Key Features**:
- **Hardware Detection**: Automatically detects GPU type, memory, CPU cores
- **Progressive Training**: Supports 3-stage curriculum learning
- **Auto-Optimization**: Optimizes batch size, workers, and memory usage
- **Multi-Dataset Support**: Works with SDD, DroneDeploy, and UDD6
- **Mixed Precision**: Automatic AMP for supported hardware

**Usage Examples**:
```bash
# Progressive training (recommended)
python train.py --stage 1 --epochs 30  # Semantic foundation
python train.py --stage 2 --epochs 20  # Landing specialization
python train.py --stage 3 --epochs 15  # Domain adaptation

# Hardware overrides
python train.py --batch_size 64 --num_workers 8  # Manual optimization
python train.py --device cpu                     # Force CPU training
```

**Implementation Highlights**:
- `HardwareDetector` class: Auto-detects optimal configurations
- `UniversalTrainer` class: Adapts to any hardware setup
- Progressive loading of stage-specific models
- Comprehensive W&B integration

### 2. `test.py` - Model Evaluation Script

**Purpose**: Comprehensive model testing with detailed metrics and visualizations.

**Key Features**:
- **Multi-Dataset Testing**: Test on all datasets simultaneously
- **Rich Metrics**: mIoU, per-class IoU, accuracy, confusion matrices
- **Visualizations**: Prediction overlays, confidence maps, class distributions
- **Performance Analysis**: Inference speed, memory usage tracking
- **Export Options**: JSON results, confusion matrix plots

**Usage Examples**:
```bash
# Comprehensive evaluation
python test.py --checkpoint outputs/stage3_best.pth --test_all_datasets

# Single dataset with visualizations
python test.py --checkpoint outputs/stage2_best.pth \
               --dataset dronedeploy \
               --save_predictions \
               --save_confusion_matrix

# Quick evaluation (subset)
python test.py --checkpoint outputs/stage1_best.pth \
               --max_samples 100 \
               --dataset sdd
```

**Output Structure**:
```
test_results/
├── results_dronedeploy_val.json      # Detailed metrics
├── confusion_matrix_dronedeploy_val.png
├── test_summary.json                 # Cross-dataset comparison
└── visualizations/
    └── dronedeploy/
        ├── batch_000.png              # Prediction visualizations
        └── ...
```

### 3. `download_datasets.py` - Dataset Management

**Purpose**: Automated download, verification, and organization of all datasets.

**Key Features**:
- **Multi-Source Support**: Kaggle, Google Drive, GitHub releases
- **Progress Tracking**: Progress bars for large downloads
- **Verification**: Checksum validation and integrity checks
- **Resume Downloads**: Interrupted download recovery
- **Space Management**: Disk space checks before download

**Usage Examples**:
```bash
# Download all datasets
python download_datasets.py --all

# Individual datasets
python download_datasets.py --dataset sdd          # Semantic Drone Dataset
python download_datasets.py --dataset dronedeploy # DroneDeploy Dataset
python download_datasets.py --dataset udd6        # Urban Drone Dataset

# Management commands
python download_datasets.py --list    # List available datasets
python download_datasets.py --verify  # Verify downloaded datasets
```

**Dataset Configurations**:
```python
datasets = {
    'sdd': {
        'source': 'kaggle',
        'kaggle_dataset': 'bulentsiyah/semantic-drone-dataset',
        'size_gb': 2.5,
        'extract_dir': 'semantic_drone_dataset'
    },
    'dronedeploy': {
        'source': 'google_drive',
        'file_id': '1Y3nK2_HlJeprk6q0B4hSKt-XhTLWGV1c',
        'size_gb': 8.0,
        'extract_dir': 'drone_deploy_dataset'
    },
    'udd6': {
        'source': 'github_release',
        'repo': 'MarcWong/UDD',
        'size_gb': 4.5,
        'extract_dir': 'udd6_dataset'
    }
}
```

### 4. `demo_neuro_symbolic.py` - Neuro-Symbolic Demo

**Purpose**: Showcase integration of neural predictions with Scallop symbolic reasoning.

**Key Features**:
- **Scallop Integration**: Combines neural outputs with logical rules
- **Safety Analysis**: Multi-layered safety assessment
- **Landing Site Detection**: Identifies and ranks potential landing zones
- **Interactive Visualization**: Rich matplotlib visualizations
- **Demo Mode**: Synthetic data demonstration

**Usage Examples**:
```bash
# Demo with synthetic data
python demo_neuro_symbolic.py --demo_mode

# Real image analysis
python demo_neuro_symbolic.py \
    --image aerial_photo.jpg \
    --weather clear \
    --uav_type small_drone \
    --model outputs/stage3_best.pth

# Batch processing
for img in *.jpg; do
    python demo_neuro_symbolic.py --image "$img" --weather cloudy
done
```

## 🧠 Neural Network Architecture

### MobileNetV3 Edge Model (`models/mobilenetv3_edge_model.py`)

**Design Philosophy**: Optimize for edge deployment while maintaining accuracy.

**Key Components**:
1. **Backbone**: MobileNetV3-Small with modifications
2. **Segmentation Head**: Lightweight decoder with skip connections
3. **Uncertainty Estimation**: Bayesian dropout for confidence maps
4. **Multi-Scale Fusion**: Feature pyramid network integration

**Model Variants**:
```python
# Standard model
model = create_edge_model(model_type='standard', num_classes=6)

# Enhanced model with uncertainty
model = create_edge_model(
    model_type='enhanced', 
    num_classes=6, 
    use_uncertainty=True,
    pretrained=True
)
```

**Performance Characteristics**:
- **Parameters**: ~2.9M (standard), ~3.8M (enhanced)
- **Memory**: ~2.1 GB VRAM during training
- **Inference**: 45+ FPS on RTX 4090, 512×512 resolution
- **Accuracy**: 78-89% depending on dataset and stage

### Loss Functions (`losses/safety_aware_losses.py`)

**Multi-Component Loss Strategy**:

1. **Focal Loss**: Handles class imbalance
   ```python
   focal_loss = α * (1 - pt)^γ * CE_loss
   ```

2. **Dice Loss**: Improves segmentation quality
   ```python
   dice_coeff = (2 * intersection + smooth) / (union + smooth)
   ```

3. **Boundary Loss**: Preserves edge details
   ```python
   boundary_loss = MSE(sobel_pred, sobel_target)
   ```

4. **Consistency Loss**: Reduces cross-dataset conflicts
   ```python
   consistency_loss = KL_divergence(similar_class_predictions)
   ```

**Dataset-Specific Weighting**:
```python
dataset_weights = {
    'semantic_drone': {'focal': 0.4, 'dice': 0.3, 'boundary': 0.2, 'consistency': 0.1},
    'dronedeploy':   {'focal': 0.5, 'dice': 0.3, 'boundary': 0.15, 'consistency': 0.05},
    'udd6':          {'focal': 0.45, 'dice': 0.25, 'boundary': 0.2, 'consistency': 0.1}
}
```

## 📊 Dataset Implementation Details

### Progressive Training Strategy

**Stage 1 - Semantic Foundation (SDD)**:
- **Purpose**: Rich semantic understanding from 24 fine-grained classes
- **Class Mapping**: 24 original → 6 unified landing classes
- **Batch Size**: Large (up to 256 on A100) for stable training
- **Focus**: High-quality feature representations

**Stage 2 - Landing Specialization (DroneDeploy)**:
- **Purpose**: Native 6-class landing decision patterns
- **Patch Strategy**: 1024×1024 patches with 50% overlap
- **Batch Size**: Medium (16-64) due to large patch size
- **Focus**: Landing-specific discriminative features

**Stage 3 - Domain Adaptation (UDD6)**:
- **Purpose**: High-altitude urban robustness
- **Strategy**: Fine-tuning with very low learning rate
- **Batch Size**: Large (up to 128) for stability
- **Focus**: Domain generalization and edge case handling

### Dataset-Specific Implementations

#### Semantic Drone Dataset (`semantic_drone_dataset.py`)
```python
class SemanticDroneDataset(Dataset):
    def __init__(self, data_root, class_mapping="advanced_6_class"):
        # 24→6 class mapping for landing relevance
        self.class_mapping = {
            'advanced_6_class': {
                # Safe surfaces
                'paved-area': 0, 'dirt': 0, 'grass': 1, 'gravel': 0,
                # Caution surfaces  
                'vegetation': 1, 'roof': 2,
                # Dangerous surfaces
                'water': 3, 'obstacle': 2, 'person': 2, 'dog': 2,
                'car': 4, 'bicycle': 4, 'truck': 4, 'bus': 4,
                # Others
                'bald-eagle': 5, 'bird': 5, 'other': 5
            }
        }
```

#### DroneDeploy Dataset (`dronedeploy_1024_dataset.py`)
```python
class DroneDeploy1024Dataset(Dataset):
    def __init__(self, data_root, patch_size=1024, stride_factor=0.5):
        # Large patch extraction for high-resolution analysis
        self.patch_size = patch_size
        self.stride = int(patch_size * stride_factor)
        
        # Native 6-class annotations optimized for landing
        self.classes = ['ground', 'vegetation', 'obstacle', 'water', 'vehicle', 'other']
```

#### UDD6 Dataset (`udd6_dataset.py`)
```python
class UDD6Dataset(Dataset):
    def __init__(self, data_root, split="train"):
        # High-altitude urban scenarios
        self.urban_mapping = {
            'building': 2,      # obstacle
            'road': 0,          # ground (safe for emergency)
            'vegetation': 1,    # caution
            'water': 3,         # danger
            'vehicle': 4,       # danger
            'other': 5          # unknown
        }
```

## 🔗 Neuro-Symbolic Integration

### Scallop Rule System (`scallop_integration/landing_rules.scl`)

**Safety Classification Rules**:
```prolog
% Basic surface safety
rel safe_surface = {"ground", "vegetation"}
rel hazardous_surface = {"obstacle", "water", "vehicle"}
rel uncertain_surface = {"other"}

% Landing site evaluation
rel suitable_landing_site(x, y, confidence) = 
    local_area_safe(x, y) and
    sufficient_space(x, y) and
    confidence := 0.8

% Mission-level safety assessment
rel mission_safety_level(level) = 
    area_safety_score(score) and
    weather_factor(weather, weather_mult) and
    uav_capability(uav, uav_mult) and
    final_score := score * weather_mult * uav_mult and
    final_score > 0.7 and
    level := "safe"
```

**Integration Workflow**:
1. Neural network produces segmentation map
2. Extract spatial and statistical features
3. Convert to Scallop facts
4. Apply logical reasoning rules
5. Generate safety assessment and landing sites

## ⚙️ Configuration System

### Hardware Auto-Detection (`train.py`)

**Detection Logic**:
```python
class HardwareDetector:
    def get_optimal_config(self):
        config = {'device': 'cpu', 'batch_size': 4, 'num_workers': 2}
        
        if self.gpu_info['available']:
            gpu = self.gpu_info['devices'][0]
            
            if 'a100' in gpu['name'].lower():
                config.update({'batch_size': 64, 'num_workers': 8})
            elif 'rtx' in gpu['name'].lower():
                config.update({'batch_size': 16, 'num_workers': 4})
            # ... more GPU-specific optimizations
        
        return config
```

### Resolution Profiles (`configs/resolution_profiles.json`)

**Multi-Resolution Training**:
```json
{
  "profiles": {
    "fast_training": {
      "input_size": [256, 256],
      "batch_size_multiplier": 4,
      "description": "Fast training for development"
    },
    "standard": {
      "input_size": [512, 512], 
      "batch_size_multiplier": 1,
      "description": "Standard resolution training"
    },
    "high_quality": {
      "input_size": [1024, 1024],
      "batch_size_multiplier": 0.25,
      "description": "High-resolution for final models"
    }
  }
}
```

## 🧪 Testing and Validation

### Comprehensive Evaluation Pipeline

**Metrics Computed**:
- **Overall**: Accuracy, mIoU, mean class accuracy
- **Per-Class**: IoU, precision, recall, F1-score
- **Spatial**: Boundary accuracy, connectivity metrics
- **Temporal**: Inference speed, memory usage

**Cross-Dataset Evaluation**:
```bash
# Train on one dataset, test on others
python train.py --stage 1 --epochs 30                    # Train on SDD
python test.py --checkpoint outputs/stage1_best.pth \
               --dataset dronedeploy                      # Test on DroneDeploy
```

**Ablation Studies**:
```bash
# Test different loss components
python train.py --focal_weight 0.5 --dice_weight 0.3     # Custom loss weights
python train.py --no_uncertainty                         # Without uncertainty estimation
```

## 🔄 Development Workflow

### Code Organization Principles

1. **Modular Design**: Each component is self-contained and testable
2. **Configuration-Driven**: Minimize hard-coded parameters
3. **Hardware Agnostic**: Automatic adaptation to available resources
4. **Documentation First**: Comprehensive docstrings and type hints
5. **Error Handling**: Graceful degradation and informative error messages

### Adding New Features

**1. New Dataset Integration**:
```python
# Create new dataset class
class MyCustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, transform=None):
        # Implementation
        pass
    
    def __getitem__(self, idx):
        return {'image': image, 'mask': mask}

# Register in download_datasets.py
datasets['my_dataset'] = {
    'name': 'My Custom Dataset',
    'source': 'custom_source',
    'size_gb': 5.0,
    'extract_dir': 'my_dataset'
}

# Add to train.py dataset selection
elif args.stage == 4:  # New stage
    datasets = create_my_custom_datasets(args.my_dataset_root)
```

**2. New Model Architecture**:
```python
# Add to models/enhanced_architectures.py
class MyNewModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # Implementation
    
    def forward(self, x):
        return predictions

# Register in create_edge_model
def create_edge_model(model_type='standard', **kwargs):
    if model_type == 'my_new_model':
        return MyNewModel(**kwargs)
```

**3. New Loss Function**:
```python
# Add to losses/safety_aware_losses.py
class MyCustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # Implementation
        return loss_dict

# Integrate in MultiDatasetLoss
def forward(self, pred, target, dataset_source):
    # ... existing losses
    custom_loss = self.my_custom_loss(pred, target)
    total_loss += weights['custom'] * custom_loss
```

### Testing Strategy

**Unit Tests**:
```bash
# Dataset loading
pytest tests/test_datasets.py

# Model architecture  
pytest tests/test_models.py

# Loss functions
pytest tests/test_losses.py
```

**Integration Tests**:
```bash
# End-to-end training
pytest tests/test_training_pipeline.py

# Inference pipeline
pytest tests/test_inference.py
```

**Performance Tests**:
```bash
# Memory usage
pytest tests/test_memory_usage.py

# Inference speed
pytest tests/test_inference_speed.py
```

## 📊 Monitoring and Debugging

### Weights & Biases Integration

**Comprehensive Logging**:
- Training/validation losses (all components)
- Hardware metrics (GPU memory, utilization, temperature)
- Learning rate schedules
- Per-class IoU progression
- Confusion matrices
- Sample predictions

**Custom Metrics**:
```python
# Log custom metrics
wandb.log({
    'hardware/gpu_memory_used': gpu_memory_gb,
    'train/focal_loss': focal_loss,
    'val/class_miou_ground': ground_iou,
    'model/parameter_count': param_count
})
```

### Error Handling and Debugging

**Common Issues and Solutions**:

1. **CUDA Out of Memory**:
   ```python
   # Automatic batch size reduction
   try:
       outputs = model(images)
   except RuntimeError as e:
       if "out of memory" in str(e):
           torch.cuda.empty_cache()
           # Reduce batch size and retry
   ```

2. **Dataset Loading Errors**:
   ```python
   # Graceful dataset fallback
   try:
       dataset = create_primary_dataset(data_root)
   except Exception as e:
       logger.warning(f"Primary dataset failed: {e}")
       dataset = create_fallback_dataset(data_root)
   ```

3. **Model Convergence Issues**:
   ```python
   # Learning rate scheduling with monitoring
   if val_loss_plateau_detected:
       scheduler.step()
       logger.info(f"Reducing LR to {optimizer.param_groups[0]['lr']}")
   ```

## 🚀 Performance Optimization

### Memory Optimization

**Gradient Accumulation**:
```python
# Simulate larger batch sizes
accumulation_steps = target_batch_size // actual_batch_size
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Memory Profiling**:
```python
# Track memory usage
torch.cuda.reset_peak_memory_stats()
# ... training code ...
peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
```

### Speed Optimization

**Data Loading Optimization**:
```python
# Optimal worker count
num_workers = min(cpu_count - 1, 8)  # Leave 1 CPU free, cap at 8

# Pin memory for GPU transfer
pin_memory = device.type == 'cuda'

# Persistent workers to avoid respawning
persistent_workers = num_workers > 0
```

**Model Optimization**:
```python
# Compile model for speed (PyTorch 2.0+)
model = torch.compile(model)

# Mixed precision training
with autocast():
    outputs = model(images)
```

## 📈 Future Extensions

### Planned Enhancements

1. **Real-Time Video Processing**:
   - Temporal consistency constraints
   - Frame-to-frame tracking
   - Adaptive quality scaling

2. **Multi-UAV Coordination**:
   - Distributed landing site allocation
   - Communication protocols
   - Collision avoidance

3. **Weather Integration**:
   - Real-time weather data
   - Weather-specific safety rules
   - Dynamic risk assessment

4. **Mobile Deployment**:
   - iOS/Android apps
   - Edge device optimization
   - Offline inference capability

### Research Directions

1. **Advanced Neuro-Symbolic Integration**:
   - Learnable logical rules
   - Differentiable programming
   - Continuous reasoning

2. **3D Landing Analysis**:
   - Depth estimation integration
   - Terrain analysis
   - 3D obstacle detection

3. **Adversarial Robustness**:
   - Attack-resistant models
   - Safety verification
   - Certified defenses

---

This repository guide provides the foundation for understanding, using, and extending the UAV Landing System. For specific API documentation, see the `docs/` directory. 