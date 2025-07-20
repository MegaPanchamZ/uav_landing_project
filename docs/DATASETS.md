# Dataset Guide

Comprehensive guide to datasets used in the UAV Landing Detection system with Neurosymbolic Memory.

## Overview

The training pipeline uses **two specialized datasets** in a staged approach to create the production BiSeNetV2 model:

1. **DroneDeploy Dataset**: Intermediate aerial view adaptation
2. **UDD6 Dataset**: Final landing-specific fine-tuning

The trained model serves as the perception component in our neurosymbolic system, with memory enhancement handling challenging scenarios.

## Stage 1: DroneDeploy Dataset

### Dataset Information
- **Name**: DroneDeploy Semantic Segmentation
- **Source**: DroneDeploy challenge dataset
- **Purpose**: Aerial view domain adaptation
- **Location**: `../datasets/drone_deploy_dataset_intermediate/dataset-medium/`
- **Model Output**: BiSeNetV2 ONNX model for production use

### Statistics
- **Total Images**: 55
- **Train Split**: 44 images
- **Validation Split**: 11 images
- **Image Format**: JPG/PNG
- **Label Format**: RGB color-coded masks
- **Resolution**: Variable (resized to 512×512 for current system)

### Class Definitions

| Class ID | Name | RGB Color | Hex | Description |
|----------|------|-----------|-----|-------------|
| 0 | Background | (0, 0, 0) | #000000 | Unlabeled areas |
| 1 | Building | (128, 0, 0) | #800000 | Residential/commercial buildings |
| 2 | Road | (128, 128, 0) | #808000 | Paved roads and pathways |
| 3 | Trees | (0, 128, 0) | #008000 | Trees and large vegetation |
| 4 | Car | (0, 0, 128) | #000080 | Vehicles and cars |
| 5 | Pool | (128, 0, 128) | #800080 | Swimming pools |
| 6 | Other | (0, 128, 128) | #008080 | Other structures |

### Directory Structure
```
drone_deploy_dataset_intermediate/
├── dataset-medium/
│   ├── index.csv                    # Image metadata
│   ├── images/                      # RGB images
│   │   ├── 107f24d6e9_F1BE1D4184INSPIRE.jpg
│   │   ├── 11cdce7802_B6A62F8BE0INSPIRE.jpg
│   │   └── ...
│   ├── labels/                      # RGB segmentation masks
│   │   ├── 107f24d6e9_F1BE1D4184INSPIRE_label.png
│   │   ├── 11cdce7802_B6A62F8BE0INSPIRE_label.png
│   │   └── ...
│   └── elevations/                  # Elevation data (unused)
│       └── *.tif
```

### Usage in Training
```python
class DroneDeployDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        # Load images and RGB labels
        # Convert RGB to class indices
        # Apply augmentations
```

## Stage 2: UDD6 Dataset

### Dataset Information
- **Name**: Urban Drone Dataset (UDD6)
- **Source**: Urban drone imagery
- **Purpose**: Landing-specific class fine-tuning
- **Location**: `../datasets/UDD/UDD/UDD6/`
- **Output**: Production BiSeNetV2 model optimized for landing detection

### Statistics
- **Train Images**: 106
- **Validation Images**: 35
- **Test Images**: Available but not used
- **Image Format**: JPG
- **Label Format**: PNG grayscale masks
- **Resolution**: Variable (resized to 512×512 for current system)

### Original UDD6 Classes

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | Other | Background and unlabeled |
| 1 | Facade | Building facades |
| 2 | Road | Roads and pathways |
| 3 | Vegetation | Trees, grass, vegetation |
| 4 | Vehicle | Cars, trucks, vehicles |
| 5 | Roof | Building rooftops |

### Landing Class Mapping

For UAV landing detection, UDD6 classes are mapped to landing suitability:

```python
UDD_TO_LANDING_MAPPING = {
    0: 0,  # Other → Background
    1: 3,  # Facade → Danger (buildings)
    2: 1,  # Road → Safe Landing (paved surfaces)
    3: 2,  # Vegetation → Caution (soft but uneven)
    4: 3,  # Vehicle → Danger (obstacles)
    5: 2,  # Roof → Caution (flat but may be fragile)
}
```

### Final Landing Classes

| Class ID | Name | Color | Suitability | Examples |
|----------|------|-------|-------------|----------|
| 0 | Background | Black | N/A | Unlabeled areas |
| 1 | Safe Landing | Green | High | Roads, parking lots, clearings |
| 2 | Caution | Yellow | Medium | Grass, rooftops, dirt areas |
| 3 | Danger | Red | None | Buildings, vehicles, obstacles |

### Directory Structure
```
UDD/UDD/UDD6/
├── src/
│   ├── train/
│   │   ├── images/
│   │   │   ├── 000001.jpg
│   │   │   ├── 000002.jpg
│   │   │   └── ...
│   │   └── gt/
│   │       ├── 000001.png
│   │       ├── 000002.png
│   │       └── ...
│   └── val/
│       ├── images/
│       └── gt/
```

## Data Processing Pipeline

### Model Integration
The trained BiSeNetV2 model is exported to ONNX format (`models/bisenetv2_uav_landing.onnx`) and integrated into the UAVLandingDetector class. The neurosymbolic memory system enhances this base perception capability.

### Current System Usage
- **Input Resolution**: 512×512 (production system)
- **Output Classes**: 4 classes (Background, Safe, Caution, Danger)
- **Integration**: ONNX Runtime with memory enhancement
- **Real-time Performance**: 6+ FPS with memory processing

### Preprocessing Steps

1. **Image Loading**: RGB images loaded with OpenCV/PIL
2. **Resizing**: All images resized to 512×512 for current production system
3. **Normalization**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
4. **Label Conversion**: RGB labels → class indices, UDD6 mapping applied

### Data Augmentation

```python
train_transforms = A.Compose([
    A.RandomRotate90(p=0.3),           # Rotation invariance
    A.HorizontalFlip(p=0.5),           # Horizontal symmetry
    A.VerticalFlip(p=0.2),             # Aerial view symmetry
    A.RandomBrightnessContrast(p=0.3), # Lighting variations
    A.ColorJitter(p=0.2),              # Color variations
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Validation Transforms

```python
val_transforms = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## 📊 Dataset Analysis

### Class Distribution Analysis

Run the dataset analysis script:

```bash
python scripts/analyze_dataset.py
```

**DroneDeploy Distribution:**
```
Class 0 (Background): 35.2%
Class 1 (Building): 18.7%
Class 2 (Road): 15.3%
Class 3 (Trees): 12.8%
Class 4 (Car): 8.9%
Class 5 (Pool): 4.1%
Class 6 (Other): 5.0%
```

**UDD6 → Landing Distribution:**
```
Class 0 (Background): 28.5%
Class 1 (Safe Landing): 22.3%
Class 2 (Caution): 31.7%
Class 3 (Danger): 17.5%
```

### Quality Assessment

- **Image Quality**: High resolution aerial imagery
- **Label Quality**: Manual annotation, some noise expected
- **Diversity**: Various urban/suburban environments
- **Challenges**: Weather variations, shadows, occlusions

## 🛠️ Custom Dataset Creation

### Preparing Your Own Dataset

1. **Image Collection**
   ```bash
   # Aerial imagery from drone/satellite
   # Consistent altitude (50-100m recommended)
   # Good lighting conditions
   # Various environments
   ```

2. **Annotation Guidelines**
   - **Safe Landing**: Flat, clear surfaces (roads, fields, parking lots)
   - **Caution**: Uneven but possible (grass, gravel, rooftops)
   - **Danger**: Obstacles and hazards (buildings, trees, vehicles, water)
   - **Background**: Sky, distant objects, unlabeled areas

3. **Annotation Tools**
   - CVAT (Computer Vision Annotation Tool)
   - Labelbox
   - VGG Image Annotator (VIA)
   - Supervisely

4. **Format Conversion**
   ```python
   def convert_annotations(annotation_dir, output_dir):
       """Convert annotations to training format."""
       # Load annotations (JSON/XML)
       # Convert to RGB masks or grayscale indices
       # Save in consistent format
   ```

### Dataset Validation

```python
def validate_dataset(image_dir, label_dir):
    """Validate dataset consistency."""
    issues = []
    
    for img_path in Path(image_dir).glob("*.jpg"):
        label_path = Path(label_dir) / f"{img_path.stem}.png"
        
        # Check if label exists
        if not label_path.exists():
            issues.append(f"Missing label: {label_path}")
            continue
            
        # Check dimensions match
        img = cv2.imread(str(img_path))
        label = cv2.imread(str(label_path), 0)
        
        if img.shape[:2] != label.shape:
            issues.append(f"Size mismatch: {img_path}")
        
        # Check class values
        unique_classes = np.unique(label)
        if max(unique_classes) > 3:  # 4 classes: 0,1,2,3
            issues.append(f"Invalid classes in {label_path}: {unique_classes}")
    
    return issues
```

## 📈 Dataset Expansion

### Data Collection Strategies

1. **Temporal Diversity**: Different times of day, seasons
2. **Geographic Diversity**: Urban, suburban, rural environments  
3. **Weather Conditions**: Clear, cloudy, various lighting
4. **Altitude Variation**: 30-150m flight heights
5. **Camera Angles**: Nadir (straight down) primarily

### Synthetic Data Generation

```python
def generate_synthetic_data():
    """Generate synthetic training data."""
    # Use game engines (Unity, Unreal)
    # Simulate various environments
    # Add realistic textures and lighting
    # Export with semantic segmentation
```

### Active Learning

```python
def select_hard_examples(model, unlabeled_data, n_samples=100):
    """Select challenging examples for annotation."""
    # Run inference on unlabeled data
    # Calculate uncertainty/entropy
    # Select most uncertain examples
    # Send for manual annotation
```

## 🔍 Troubleshooting

### Common Issues

1. **Missing Labels**
   ```bash
   find images/ -name "*.jpg" | while read img; do
       label="${img/images/labels}"
       label="${label/.jpg/.png}"
       [ ! -f "$label" ] && echo "Missing: $label"
   done
   ```

2. **Class Imbalance**
   ```python
   # Use weighted loss functions
   class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
   criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
   ```

3. **Memory Issues**
   ```python
   # Reduce batch size or image resolution
   BATCH_SIZE = 4
   INPUT_SIZE = (224, 224)  # Instead of (256, 256)
   ```

### Performance Tips

- **Data Loading**: Use `num_workers=4` for parallel loading
- **Caching**: Cache preprocessed data for faster training
- **Validation**: Regular validation prevents overfitting
- **Monitoring**: Track class-specific metrics

---

## 📞 Support

For dataset questions:
- Check `scripts/analyze_dataset.py` for analysis tools
- Review `docs/TRAINING.md` for training details
- Contact dataset providers for licensing questions

**Happy Data Preparation!** 📊🚁
