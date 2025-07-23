# 🚁 Multi-Dataset UAV Landing Strategy

## 📊 **Dataset Inventory & Strategic Usage**

### **Dataset 1: Semantic Drone Dataset (SDD)**
- **Source**: 400 training images (6000×4000px, 24MP)
- **Altitude**: 5-30m (low-altitude scenarios)
- **Classes**: 20 detailed semantic classes
- **View**: Nadir (bird's eye view)
- **Domain**: Urban residential areas
- **Role**: **Primary semantic learning** + **Fine-grained feature extraction**

### **Dataset 2: DroneDeploy Machine Learning Segmentation**
- **Source**: 51 areas across US (super crisp 10cm resolution)
- **Classes**: 6 classes (building, clutter, vegetation, water, ground, car)
- **View**: High-resolution orthomosaics
- **Domain**: Mixed rural/urban areas
- **Role**: **Landing-specific training** + **High-resolution details**

### **Dataset 3: Urban Drone Dataset (UDD6)**
- **Source**: 200 training images (3840×2160+)
- **Altitude**: 60-100m (high-altitude scenarios)
- **Classes**: 6 classes (road, roof, vehicle, other, facade, vegetation)
- **Domain**: Dense urban environments
- **Role**: **High-altitude domain adaptation** + **Urban robustness**

## 🎯 **Strategic Multi-Dataset Usage Plan**

### **The Challenge: Avoiding Learning Conflicts**

**Potential Issues**:
- **Class confusion**: Same objects labeled differently across datasets
- **Domain shift**: Different altitudes, resolutions, environments
- **Label inconsistency**: Semantic granularity mismatch
- **Catastrophic forgetting**: Later datasets overwriting earlier learning

**Solution: Hierarchical Progressive Training**

```
Stage 1: Semantic Foundation    (SDD - 20 classes → 6 landing classes)
           ↓
Stage 2: Landing Specialization (DroneDeploy - native 6 classes)
           ↓  
Stage 3: Domain Adaptation      (UDD6 - altitude/urban robustness)
           ↓
Stage 4: Joint Refinement       (All datasets with unified loss)
```

## 🏗️ **Unified 6-Class Landing System**

### **Harmonized Class Mapping**

```python
UNIFIED_LANDING_CLASSES = {
    0: "ground",       # Safe flat surfaces (roads, dirt, pavement)
    1: "vegetation",   # Grass, trees, bushes (emergency acceptable)
    2: "obstacle",     # Buildings, walls, large objects (avoid)
    3: "water",        # Water bodies, pools (critical hazard)
    4: "vehicle",      # Cars, bikes, moving objects (dynamic avoid)
    5: "other"         # Clutter, unknown, complex mixed areas
}

# Cross-dataset mapping strategy
DATASET_MAPPINGS = {
    'semantic_drone': {
        # SDD 20 classes → 6 landing classes
        'tree': 1, 'grass': 1, 'other_vegetation': 1,          # vegetation
        'dirt': 0, 'gravel': 0, 'paved_area': 0,              # ground
        'rocks': 2, 'roof': 2, 'wall': 2, 'fence': 2,         # obstacle
        'window': 2, 'door': 2, 'obstacle': 2,                # obstacle
        'water': 3, 'pool': 3,                                # water
        'person': 4, 'dog': 4, 'car': 4, 'bicycle': 4,       # vehicle/dynamic
        'fence-pole': 5                                       # other
    },
    
    'dronedeploy': {
        # Native 6 classes → 6 landing classes (direct mapping)
        'ground': 0, 'vegetation': 1, 'building': 2,
        'water': 3, 'car': 4, 'clutter': 5
    },
    
    'udd6': {
        # UDD6 6 classes → 6 landing classes
        'road': 0,           # ground
        'vegetation': 1,     # vegetation
        'roof': 2,           # obstacle
        'facade': 2,         # obstacle
        'vehicle': 4,        # vehicle
        'other': 5           # other
    }
}
```

## ⚡ **Multi-Dataset Loss Functions**

### **Problem: Different Dataset Characteristics**
- **SDD**: Rich semantics but needs mapping
- **DroneDeploy**: Native landing classes but limited scale
- **UDD6**: Different altitude/perspective

### **Solution: Adaptive Multi-Component Loss**

```python
class MultiDatasetLoss(nn.Module):
    """
    Adaptive loss function that handles multiple datasets with different characteristics.
    """
    
    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
        
        # Component losses
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()
        self.consistency_loss = ConsistencyLoss()
        
        # Dataset-specific weights
        self.dataset_weights = {
            'semantic_drone': {'focal': 0.4, 'dice': 0.3, 'boundary': 0.2, 'consistency': 0.1},
            'dronedeploy': {'focal': 0.5, 'dice': 0.3, 'boundary': 0.15, 'consistency': 0.05},
            'udd6': {'focal': 0.45, 'dice': 0.25, 'boundary': 0.2, 'consistency': 0.1}
        }
    
    def forward(self, predictions, targets, dataset_source, confidence_maps=None):
        """
        Compute adaptive loss based on dataset source.
        
        Args:
            predictions: Model predictions [B, 6, H, W]
            targets: Ground truth labels [B, H, W]
            dataset_source: Source dataset identifier
            confidence_maps: Uncertainty/confidence maps [B, H, W]
        """
        
        # Get dataset-specific weights
        weights = self.dataset_weights.get(dataset_source, self.dataset_weights['dronedeploy'])
        
        # Core segmentation losses
        focal = self.focal_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        boundary = self.boundary_loss(predictions, targets)
        
        # Consistency loss (reduces cross-dataset conflicts)
        consistency = self.consistency_loss(predictions, targets, dataset_source)
        
        # Weighted combination
        total_loss = (
            weights['focal'] * focal +
            weights['dice'] * dice +
            weights['boundary'] * boundary +
            weights['consistency'] * consistency
        )
        
        # Confidence weighting (if available)
        if confidence_maps is not None:
            confidence_weight = torch.mean(confidence_maps)
            total_loss = total_loss * confidence_weight
        
        return {
            'total': total_loss,
            'focal': focal,
            'dice': dice,
            'boundary': boundary,
            'consistency': consistency
        }

class ConsistencyLoss(nn.Module):
    """
    Ensures consistent predictions across similar semantic concepts from different datasets.
    """
    
    def __init__(self):
        super().__init__()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
        # Define semantic similarity groups
        self.similarity_groups = {
            'safe_surfaces': [0, 1],    # ground, vegetation
            'obstacles': [2, 4],        # obstacle, vehicle
            'hazards': [3],             # water
            'uncertain': [5]            # other
        }
    
    def forward(self, predictions, targets, dataset_source):
        """Compute consistency loss to reduce cross-dataset conflicts."""
        
        # Convert predictions to probabilities
        pred_probs = F.softmax(predictions, dim=1)
        
        consistency_loss = 0.0
        
        # Encourage similar predictions for semantically similar classes
        for group_name, class_ids in self.similarity_groups.items():
            if len(class_ids) > 1:
                # Compute similarity within group
                group_probs = pred_probs[:, class_ids]
                
                # Encourage uniform distribution within semantic groups
                uniform_target = torch.ones_like(group_probs) / len(class_ids)
                
                # Only apply where these classes are present
                group_mask = torch.isin(targets, torch.tensor(class_ids, device=targets.device))
                
                if group_mask.any():
                    masked_probs = group_probs[group_mask.unsqueeze(1).expand(-1, len(class_ids), -1, -1)]
                    masked_target = uniform_target[group_mask.unsqueeze(1).expand(-1, len(class_ids), -1, -1)]
                    
                    group_loss = F.mse_loss(masked_probs, masked_target)
                    consistency_loss += group_loss
        
        return consistency_loss
```

## 🚀 **Progressive Training Strategy**

### **Stage 1: Semantic Foundation (SDD)**
```python
# Focus on rich semantic understanding
def train_stage1_semantic_foundation():
    dataset = SemanticDroneDataset(
        class_mapping='20_to_6_landing',
        patch_size=1024,
        augmentation='aggressive'
    )
    
    loss_fn = MultiDatasetLoss()
    
    # Emphasize semantic learning
    loss_weights = {
        'focal': 0.4,     # Class separation
        'dice': 0.3,      # Segmentation quality  
        'boundary': 0.2,  # Edge preservation
        'consistency': 0.1 # Cross-class consistency
    }
    
    train_epochs = 50
    learning_rate = 1e-3
```

### **Stage 2: Landing Specialization (DroneDeploy)**
```python
# Focus on landing-specific decisions
def train_stage2_landing_specialization():
    dataset = DroneDeploy1024Dataset(
        native_6_classes=True,
        patch_size=1024,
        edge_enhancement=True
    )
    
    # Load Stage 1 model
    model = load_checkpoint('stage1_best_model.pth')
    
    # Lower learning rate for fine-tuning
    loss_weights = {
        'focal': 0.5,     # Emphasize landing decisions
        'dice': 0.3,      # Maintain segmentation quality
        'boundary': 0.15, # Edge refinement
        'consistency': 0.05 # Minimal conflicts
    }
    
    train_epochs = 30
    learning_rate = 1e-4  # Lower LR for stability
```

### **Stage 3: Domain Adaptation (UDD6)**
```python
# Focus on altitude/urban robustness
def train_stage3_domain_adaptation():
    dataset = UDD6Dataset(
        class_mapping='6_to_6_landing',
        patch_size=1024,
        altitude_simulation=True
    )
    
    # Load Stage 2 model
    model = load_checkpoint('stage2_best_model.pth')
    
    # Domain adaptation focused
    loss_weights = {
        'focal': 0.45,
        'dice': 0.25,
        'boundary': 0.2,
        'consistency': 0.1  # Important for cross-domain
    }
    
    train_epochs = 20
    learning_rate = 5e-5  # Very low LR
```

### **Stage 4: Joint Refinement**
```python
# Multi-dataset joint training
def train_stage4_joint_refinement():
    # Combine all datasets with careful sampling
    combined_dataset = MultiDatasetLoader([
        ('semantic_drone', 0.4),    # 40% SDD
        ('dronedeploy', 0.4),       # 40% DroneDeploy  
        ('udd6', 0.2)               # 20% UDD6
    ])
    
    # Balanced multi-dataset loss
    loss_fn = MultiDatasetLoss(adaptive_weights=True)
    
    train_epochs = 15
    learning_rate = 1e-5  # Minimal changes
```

## 📊 **Expected Benefits & Risk Mitigation**

### **Benefits of Multi-Dataset Training**
1. **Rich semantic understanding** (SDD's 20 classes)
2. **Landing-optimized decisions** (DroneDeploy's native classes)
3. **Altitude robustness** (UDD6's high-altitude perspective)
4. **Domain generalization** (varied environments/conditions)
5. **Larger effective dataset** (400 + 10K+ + 200 patches)

### **Risk Mitigation Strategies**
1. **Progressive training**: Prevents catastrophic forgetting
2. **Consistency loss**: Reduces cross-dataset conflicts
3. **Adaptive loss weights**: Accounts for dataset characteristics
4. **Careful class mapping**: Maintains semantic coherence
5. **Lower learning rates**: Preserves previous learning

### **Alternative: Single Dataset Focus**
If multi-dataset proves problematic:

```python
# Option A: DroneDeploy Only (Safest)
primary_dataset = DroneDeploy1024Dataset()  # 10K+ patches, native 6 classes

# Option B: SDD Primary + DroneDeploy Fine-tuning  
stage1 = SemanticDroneDataset()  # Rich semantics
stage2 = DroneDeploy1024Dataset()  # Landing specialization
```

## 🎯 **Recommended Approach**

### **Conservative Strategy (Recommended)**
1. **Start with DroneDeploy only** - native 6 classes, no mapping issues
2. **If performance insufficient** - add SDD with careful mapping
3. **If generalization needed** - add UDD6 for domain adaptation

### **Aggressive Strategy (If data hungry)**
1. **Full progressive training** as outlined above
2. **Careful monitoring** for learning conflicts
3. **Ablation studies** to validate each stage

The key is **careful monitoring** at each stage and **reverting** if we see learning degradation. The progressive approach with consistency losses should minimize conflicts while maximizing the benefits of each dataset.