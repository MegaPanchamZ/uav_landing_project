# 🚁 Refined Edge UAV Landing Strategy (KDP-Net Inspired)

##  **Research-Backed Approach**

Based on the **KDP-Net research paper** and proper **DroneDeploy dataset utilization**, this refined strategy addresses the real constraints while following proven methodologies.

### **Key Insights from KDP-Net Paper**
- **Proven datasets**: UDD (high-altitude 60-100m) + SDD (low-altitude 5-30m) + DroneDeploy (10cm resolution)
- **Image preprocessing**: Large images (6000×4000) cropped to 1024×1024 patches
- **Dataset splits**: 6:2:2 ratio (train:val:test)
- **BiSeNetV2 baseline**: Achieved good results but KDP-Net improved mIoU by 4-6%
- **6 proven classes**: Building, clutter, vegetation, water, ground, car

## 📊 **Optimal 6-Class Landing System**

### **DroneDeploy-Aligned Classes**
```python
REFINED_LANDING_CLASSES = {
    0: "ground",       # Safe flat landing areas (roads, dirt, pavement)
    1: "vegetation",   # Acceptable emergency landing (grass, low vegetation)  
    2: "building",     # Hard obstacles to avoid
    3: "water",        # Critical hazard - no landing
    4: "car",          # Dynamic obstacles
    5: "clutter"       # Mixed debris/objects
}

# Landing decision mapping:
LANDING_PRIORITY = {
    0: "SAFE_PRIMARY",    # ground → ideal landing
    1: "SAFE_SECONDARY",  # vegetation → acceptable landing
    2: "AVOID",           # building → obstacle
    3: "CRITICAL_AVOID",  # water → hazard
    4: "AVOID",           # car → dynamic obstacle  
    5: "CAUTION"          # clutter → case-by-case
}
```

### **Why This Works Better**
- **Proven dataset**: DroneDeploy has 51 areas with 10cm resolution
- **Real UAV imagery**: Actually collected from drones, not street-level
- **Manageable classes**: 6 classes with clear landing implications
- **Research validation**: Used in published UAV landing research

## 🏗️ **Refined Architecture Strategy**

### **Option 1: Fine-tuned BiSeNetV2 (Baseline)**
```python
class RefinedBiSeNetV2(nn.Module):
    """
    BiSeNetV2 fine-tuned for DroneDeploy 6-class landing detection.
    Following KDP-Net paper methodology.
    """
    def __init__(self):
        super().__init__()
        # Load Cityscapes pretrained BiSeNetV2
        self.backbone = BiSeNetV2(pretrained='cityscapes')
        
        # Replace final classifier for 6 landing classes
        self.backbone.head = nn.Conv2d(128, 6, kernel_size=1)
        
        # Edge enhancement module (from KDP-Net)
        self.edge_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),  # Edge prediction
            nn.Sigmoid()
        )
```

### **Option 2: Lightweight EdgeNet (Speed Priority)**
```python
class KDPInspiredEdgeNet(nn.Module):
    """
    Lightweight model inspired by KDP-Net but optimized for edge deployment.
    """
    def __init__(self):
        super().__init__()
        # MobileNetV3 backbone for speed
        self.backbone = mobilenet_v3_small(pretrained=True)
        
        # KDP-inspired multi-scale module
        self.kdp_module = KDPModule(576, [1, 2, 3])  # Different dilation rates
        
        # Bilateral segmentation network (simplified)
        self.bsn = SimplifiedBSN(6)  # 6 classes
        
        # Edge enhancement
        self.edge_module = EdgeModule()
```

## 📚 **Proper Dataset Preprocessing (Following KDP-Net)**

### **DroneDeploy Processing Pipeline**
```python
class DroneDeploy1024Dataset(Dataset):
    """
    DroneDeploy dataset with KDP-Net preprocessing methodology.
    - Large images cropped to 1024×1024 patches
    - 6:2:2 train/val/test split
    - 6 classes aligned with landing decisions
    """
    
    def __init__(self, data_root, split='train', patch_size=1024):
        self.patch_size = patch_size
        self.split = split
        
        # Load DroneDeploy images (very high resolution)
        self.large_images = self._load_large_images(data_root)
        
        # Generate 1024×1024 patches following KDP-Net methodology
        self.patches = self._generate_patches()
        
        # Apply 6:2:2 split as in paper
        self.patches = self._apply_split(split)
    
    def _generate_patches(self):
        """Generate 1024×1024 patches from large images."""
        patches = []
        
        for img_path, label_path in self.large_images:
            # Load full resolution image
            image = cv2.imread(img_path)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            
            h, w = image.shape[:2]
            
            # Sliding window to generate patches
            for y in range(0, h - self.patch_size + 1, self.patch_size // 2):
                for x in range(0, w - self.patch_size + 1, self.patch_size // 2):
                    # Extract patch
                    img_patch = image[y:y+self.patch_size, x:x+self.patch_size]
                    label_patch = label[y:y+self.patch_size, x:x+self.patch_size]
                    
                    # Skip patches with mostly empty/unknown pixels
                    if self._is_valid_patch(label_patch):
                        patches.append({
                            'image': img_patch,
                            'label': self._map_to_landing_classes(label_patch),
                            'source': img_path,
                            'coords': (x, y)
                        })
        
        return patches
    
    def _map_to_landing_classes(self, dronedeply_label):
        """Map DroneDeploy classes to landing classes."""
        # DroneDeploy grayscale to landing class mapping
        mapping = {
            81: 2,   # Building → building (avoid)
            91: 0,   # Road → ground (safe)
            99: 4,   # Car → car (avoid)
            105: 5,  # Background → clutter
            132: 1,  # Trees → vegetation (acceptable)
            155: 3,  # Pool/Water → water (critical avoid)
        }
        
        landing_label = np.zeros_like(dronedeply_label)
        for dd_class, landing_class in mapping.items():
            landing_label[dronedeply_label == dd_class] = landing_class
        
        return landing_label
```

### **Expected Dataset Scale**
- **DroneDeploy images**: 51 areas × ~10-20 large images = ~500 source images
- **1024×1024 patches**: Each large image → 20-50 patches = **10,000-25,000 training patches**
- **Effective training data**: Much larger than original ~400 images!

## ⚡ **Hybrid Loss Function (KDP-Net Inspired)**

```python
class KDPInspiredLoss(nn.Module):
    """
    Hybrid loss function inspired by KDP-Net paper.
    Combines main segmentation loss with edge enhancement.
    """
    
    def __init__(self, num_classes=6, edge_weight=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.edge_weight = edge_weight
        
        # Main losses
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.dice_loss = DiceLoss()
        
        # Edge enhancement
        self.edge_loss = nn.BCELoss()
        
    def forward(self, predictions, targets):
        """
        predictions: {
            'main': [B, 6, H, W],      # Main segmentation
            'edge': [B, 1, H, W]       # Edge prediction
        }
        """
        main_pred = predictions['main']
        edge_pred = predictions['edge']
        
        # Generate edge ground truth using Canny (following KDP-Net)
        edge_gt = self._generate_edge_gt(targets)
        
        # Main segmentation losses
        focal = self.focal_loss(main_pred, targets)
        dice = self.dice_loss(main_pred, targets)
        
        # Edge enhancement loss
        edge = self.edge_loss(edge_pred.squeeze(1), edge_gt.float())
        
        # Weighted combination (following KDP-Net approach)
        total_loss = 0.5 * focal + 0.2 * dice + self.edge_weight * edge
        
        return {
            'total': total_loss,
            'focal': focal,
            'dice': dice,
            'edge': edge
        }
    
    def _generate_edge_gt(self, targets):
        """Generate edge ground truth using Canny edge detection."""
        batch_size = targets.size(0)
        edge_gt = torch.zeros(batch_size, targets.size(1), targets.size(2))
        
        for i in range(batch_size):
            # Convert to numpy for Canny
            target_np = targets[i].cpu().numpy().astype(np.uint8)
            
            # Apply Canny edge detection
            edges = cv2.Canny(target_np, 50, 150)
            edge_gt[i] = torch.from_numpy(edges / 255.0)
        
        return edge_gt.to(targets.device)
```

## 🚀 **Training Strategy (Research-Validated)**

### **Stage 1: DroneDeploy Baseline Training**
```bash
# Following KDP-Net methodology
python train_refined_edge.py \
    --dataset dronedeploy \
    --model bisenetv2_refined \
    --patch_size 1024 \
    --batch_size 8 \
    --num_epochs 100 \
    --split_ratio 6:2:2 \
    --use_edge_loss \
    --lr 0.001
```

### **Stage 2: Multi-Dataset Domain Adaptation**
```bash
# Add UDD6 for high-altitude scenarios
python train_refined_edge.py \
    --dataset dronedeploy+udd6 \
    --model bisenetv2_refined \
    --pretrained outputs/stage1/best_model.pth \
    --batch_size 8 \
    --num_epochs 50 \
    --lr 0.0001  # Lower LR for fine-tuning
```

### **Stage 3: Edge Optimization**
```bash
# Convert to edge-optimized model
python optimize_for_edge.py \
    --input_model outputs/stage2/best_model.pth \
    --target_inference_ms 30 \
    --quantization int8 \
    --output edge_landing_optimized.onnx
```

## 📊 **Expected Performance (Research-Based)**

### **Model Comparison (Following KDP-Net Results)**
| Model | mIoU | Speed (ms) | Size (MB) | Notes |
|-------|------|------------|-----------|-------|
| **BiSeNetV2 Fine-tuned** | 75-80% | 25-35ms | 15MB | Research baseline |
| **KDP-Inspired EdgeNet** | 70-75% | 15-25ms | 8MB | Speed optimized |
| **Ultra-Fast EdgeNet** | 65-70% | 8-15ms | 4MB | Extreme edge |

### **Safety-Critical Metrics**
- **Water detection recall**: >98% (critical for safety)
- **Building detection precision**: >95% (avoid collisions)
- **Ground classification accuracy**: >92% (safe landing)

##  **Implementation Priorities**

### **Phase 1: Proven Baseline (2 weeks)**
1. **DroneDeploy preprocessing**: Implement 1024×1024 patch generation
2. **BiSeNetV2 fine-tuning**: Use Cityscapes pretrained → DroneDeploy 6-class
3. **Hybrid loss function**: Implement edge-enhanced training
4. **Validation**: Achieve research-comparable mIoU (~75%)

### **Phase 2: Edge Optimization (1 week)**
1. **Model compression**: Quantization, pruning, knowledge distillation
2. **ONNX conversion**: Deploy-ready format
3. **Speed benchmarking**: Target <30ms inference
4. **Safety validation**: Test critical scenarios

### **Phase 3: Real-Time Integration (1 week)**
1. **Landing zone analysis**: Fast spatial reasoning
2. **Uncertainty quantification**: Confidence-based decisions
3. **Fail-safe mechanisms**: Conservative bias for safety
4. **Flight controller integration**: ROS/MAVLink compatibility

## 💡 **Key Advantages of This Approach**

### **Research-Validated**
- Follows proven KDP-Net methodology
- Uses validated datasets (DroneDeploy, UDD, SDD)
- Leverages research-proven 6-class system

### **Practical for Edge Deployment**
- Realistic model sizes (4-15MB)
- Achievable inference speeds (15-35ms)
- Conservative safety bias

### **Scalable Training Data**
- 10K+ patches from DroneDeploy cropping
- Multi-dataset domain adaptation
- Extreme augmentation for robustness

### **Industry-Ready**
- ONNX deployment format
- Uncertainty quantification
- Comprehensive safety mechanisms

This refined approach leverages **actual research results** while maintaining focus on **real-time edge deployment**. The combination of proven datasets, research-validated architectures, and practical edge constraints should yield a deployable UAV landing system. 🚁

Ready to implement the research-backed approach? 