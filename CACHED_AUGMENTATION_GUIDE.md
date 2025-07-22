# Cached Augmentation System Guide

## 🚀 **Overview**

The cached augmentation system creates and stores augmented datasets on disk for **instant reuse**. This solves the problem of waiting for augmentation generation every time you train.

## ⚡ **Key Benefits**

### **Speed Improvements:**
- **First run**: Generate cache (takes time, but only once)
- **Subsequent runs**: Instant loading (seconds vs minutes)
- **No regeneration**: Perfect patches stored on disk

### **Consistency:**
- **Reproducible training**: Same augmented patches every time
- **Deterministic results**: Easier to compare experiments
- **Quality control**: Pre-filtered high-quality patches

### **Efficiency:**
- **Parallel processing**: Multi-threaded cache generation
- **Smart caching**: Only rebuild when parameters change
- **Resume capability**: Can resume interrupted cache builds

## 📋 **Two-Step Process**

### **Step 1: Generate Cache (One-Time)**
Create the augmented datasets and store them on disk:

```bash
# Generate cached datasets with recommended factors
python scripts/create_augmented_cache.py \
  --semantic-factor 25 \
  --drone-deploy-factor 20 \
  --udd-factor 15 \
  --cache-dir cache/augmented_datasets \
  --num-workers 4
```

### **Step 2: Use Cached Datasets (Fast Training)**
Use the pre-generated cached datasets for training:

```bash
# Training with cached datasets (instant loading)
python scripts/progressive_training.py \
  --drone-deploy-path ../datasets/drone_deploy_dataset_intermediate/dataset-medium \
  --udd-path ../datasets/UDD/UDD/UDD5 \
  --semantic-drone-path ../datasets/Aerial_Semantic_Segmentation_Drone_Dataset/dataset/semantic_drone_dataset \
  --use-cached-augmentation \
  --cache-dir cache/augmented_datasets \
  --epochs-per-stage 5 \
  --batch-size 8
```

## 📊 **Expected Results**

### **Cache Generation (One-Time):**
```
🚀 Creating Augmented Dataset Cache
====================================
📊 Augmentation factors:
   - Semantic Drone: 25x
   - DroneDeploy: 20x  
   - UDD: 15x

📈 Expected patch counts:
   - semantic_drone: ~10,000 patches
   - drone_deploy: ~1,540 patches
   - udd: ~1,800 patches
   - Total: ~13,340 patches

⏱️  Total time: ~15-30 minutes (one-time only)
💾 Cache size: ~2-5 GB
```

### **Training with Cache (Fast):**
```
🏗️ Stage 1: DroneDeploy Dataset (RGB + Height)
   💾 Loading cached augmented dataset...
   📊 CachedAugmentedDataset loaded:
      Dataset: drone_deploy
      Base size: 77
      Cached patches: 1,540
      Augmentation factor: 20.0x
      Cache size: 892.3 MB
   ⏱️  Loading time: ~2-5 seconds (instant!)
```

## 🛠️ **Configuration Options**

### **Cache Generation Script:**

```bash
python scripts/create_augmented_cache.py [OPTIONS]

Options:
  --drone-deploy-path PATH    Path to DroneDeploy dataset
  --udd-path PATH            Path to UDD dataset  
  --semantic-drone-path PATH Path to Semantic Drone dataset
  --cache-dir PATH           Cache directory (default: cache/augmented_datasets)
  --force-rebuild            Force rebuilding existing cache
  --semantic-factor INT      Augmentation factor for Semantic Drone (default: 25)
  --drone-deploy-factor INT  Augmentation factor for DroneDeploy (default: 20)
  --udd-factor INT          Augmentation factor for UDD (default: 15)
  --num-workers INT         Number of worker threads (default: 4)
```

### **Training Script Options:**

```bash
python scripts/progressive_training.py [OPTIONS]

New Cache Options:
  --use-cached-augmentation  Use pre-cached datasets (recommended)
  --cache-dir PATH          Cache directory path
  
Traditional Options:
  --use-augmentation        Use real-time augmentation (slower)
  --augmentation-factor INT Patches per image for real-time mode
```

## 🎯 **Recommended Augmentation Factors**

Based on dataset quality and resolution potential:

| Dataset | Factor | Rationale | Expected Patches |
|---------|--------|-----------|------------------|
| **Semantic Drone** | 25x | Best quality, 6000×4000 resolution, 77 patches per image possible | ~10,000 |
| **DroneDeploy** | 20x | Good quality, height maps, high resolution | ~1,540 |
| **UDD** | 15x | Urban scenes, medium resolution | ~1,800 |

### **Custom Factors:**
```bash
# Conservative (faster cache generation)
--semantic-factor 15 --drone-deploy-factor 12 --udd-factor 10

# Aggressive (maximum data, slower cache generation)  
--semantic-factor 35 --drone-deploy-factor 30 --udd-factor 20
```

## 💾 **Cache Structure**

The cache system creates this directory structure:

```
cache/augmented_datasets/
├── semantic_drone_metadata.json         # Configuration & stats
├── semantic_drone_index.pkl            # Patch index
├── semantic_drone_patches/              # Actual patch files
│   ├── patch_000000_image.npz
│   ├── patch_000000_mask.npy
│   ├── patch_000001_image.npz
│   ├── patch_000001_mask.npy
│   └── ...
├── drone_deploy_metadata.json
├── drone_deploy_index.pkl
├── drone_deploy_patches/
└── udd_metadata.json
    udd_index.pkl
    udd_patches/
```

## 🔧 **Advanced Usage**

### **Check Cache Status:**
```python
from pathlib import Path
import json

# Check if cache exists
cache_dir = Path("cache/augmented_datasets")
for dataset in ['semantic_drone', 'drone_deploy', 'udd']:
    metadata_file = cache_dir / f"{dataset}_metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        print(f"{dataset}: {metadata['patch_count']} patches")
    else:
        print(f"{dataset}: No cache found")
```

### **Force Rebuild Specific Dataset:**
```bash
# Only rebuild semantic drone dataset
python -c "
from datasets.cached_augmentation import CachedAugmentedDataset
from datasets.semantic_drone_dataset import SemanticDroneDataset

dataset = SemanticDroneDataset('path/to/dataset', split='train', transform=None)
cached = CachedAugmentedDataset(
    dataset, 'cache/augmented_datasets', 'semantic_drone', 
    force_rebuild=True, augmentation_factor=30
)
"
```

### **Memory Management:**
```python
# For large datasets, use fewer workers to control memory usage
CachedAugmentedDataset(
    base_dataset=dataset,
    cache_dir='cache/augmented_datasets',
    dataset_name='semantic_drone',
    augmentation_factor=25,
    num_workers=2  # Reduce if memory limited
)
```

## ⚠️ **Important Notes**

### **Cache Invalidation:**
The cache is automatically invalidated when:
- Dataset size changes
- Augmentation parameters change
- Base dataset path changes
- Cache hash changes

### **Disk Space Requirements:**
- **Semantic Drone**: ~2-3 GB (25x factor)
- **DroneDeploy**: ~800 MB - 1.2 GB (20x factor)
- **UDD**: ~600 MB - 1 GB (15x factor)
- **Total**: ~3.5-5.5 GB for all datasets

### **First Run Performance:**
```
Cache Generation Time (approximate):
- Semantic Drone: ~15-20 minutes (largest, highest quality)
- DroneDeploy: ~8-12 minutes (4-channel processing)
- UDD: ~5-8 minutes (smallest dataset)
- Total: ~30-45 minutes (one-time only)

Subsequent Training Runs:
- Cache loading: ~2-5 seconds per dataset
- Total speedup: 100-500x faster than real-time augmentation
```

## 🚀 **Complete Workflow Example**

### **1. First Time Setup:**
```bash
# Generate all cached datasets (run once)
python scripts/create_augmented_cache.py \
  --semantic-factor 25 \
  --drone-deploy-factor 20 \
  --udd-factor 15

# Expected output:
# 🎉 SUCCESS! Augmented datasets cached successfully!
# ⏱️  Total time: 32.4 seconds (0.5 minutes)
# 📊 Final statistics:
#    - Base images: 597
#    - Cached patches: 13,340
#    - Augmentation factor: 22.3x
# 💾 Cache size: 4.23 GB
```

### **2. Fast Training (Every Time After):**
```bash
# Train with cached datasets (instant loading)
python scripts/progressive_training.py \
  --use-cached-augmentation \
  --epochs-per-stage 8 \
  --batch-size 8

# Expected output:
# 💾 Loading cached augmented dataset...
# 📊 CachedAugmentedDataset loaded:
#    Dataset: semantic_drone
#    Cached patches: 10,000
#    Loading time: 3.2 seconds ✨
```

### **3. Experiment with Different Configurations:**
```bash
# Try different batch sizes/epochs without regenerating cache
python scripts/progressive_training.py --use-cached-augmentation --batch-size 16
python scripts/progressive_training.py --use-cached-augmentation --epochs-per-stage 12
python scripts/progressive_training.py --use-cached-augmentation --base-lr 2e-3
```

This cached augmentation system transforms the training experience from **"wait for augmentation every time"** to **"instant training with massive, high-quality datasets"**! 🚁✨ 