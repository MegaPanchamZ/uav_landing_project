#!/usr/bin/env python3
"""
Staged Fine-Tuning Pipeline for UAV Landing Detection

Stage 1: BiSeNetV2 (Cityscapes) -> DroneDeploy (Intermediate Fine-Tuning)
Stage 2: Stage1 Model -> UDD-6 (Task-Specific Fine-Tuning)

Following the approach from path.md and intermediate_training.md
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from typing import Dict, Tuple, List
import albumentations as A
from albumentations.pytorch import ToTensorV2

# DroneDeploy RGB -> Class mapping (from our analysis)
DRONE_DEPLOY_MAPPING = {
    # BGR format (as found in our analysis) -> Class ID  
    (60, 180, 75): 2,     # VEGETATION -> 2
    (145, 30, 180): 6,    # CLUTTER -> 6
    (255, 0, 255): 0,     # IGNORE -> 0
    (255, 255, 255): 3,   # GROUND -> 3
    (230, 25, 75): 1,     # BUILDING -> 1
    (0, 130, 200): 5,     # CAR -> 5
    (245, 130, 48): 4,    # WATER -> 4
}

# UDD-6 RGB -> Class mapping (official UDD6 classes)
UDD6_MAPPING = {
    # RGB format -> Original UDD6 class ID
    (0, 0, 0): 0,         # Other -> 0
    (102, 102, 156): 1,   # Facade -> 1
    (128, 64, 128): 2,    # Road -> 2
    (107, 142, 35): 3,    # Vegetation -> 3
    (0, 0, 142): 4,       # Vehicle -> 4
    (70, 70, 70): 5,      # Roof -> 5
}

# UDD6 -> UAV Landing class mapping
UDD_TO_LANDING = {
    0: 0,  # Other -> ignore
    1: 2,  # Facade -> high_obstacle
    2: 1,  # Road -> safe_landing
    3: 1,  # Vegetation -> safe_landing (assuming short grass)
    4: 2,  # Vehicle -> high_obstacle
    5: 2,  # Roof -> high_obstacle
}

# Stage 1: DroneDeploy classes (7 classes)
STAGE1_CLASSES = {
    0: "ignore",
    1: "building", 
    2: "vegetation",
    3: "ground",
    4: "water", 
    5: "car",
    6: "clutter"
}

# Stage 2: UAV Landing classes (4 classes)  
STAGE2_CLASSES = {
    0: "ignore",
    1: "safe_landing",     # Ground, short vegetation
    2: "high_obstacle",    # Buildings, cars
    3: "unsafe_terrain"    # Water, clutter
}

# Stage 1 -> Stage 2 class mapping
STAGE_EVOLUTION = {
    0: 0,  # ignore -> ignore
    1: 2,  # building -> high_obstacle
    2: 1,  # vegetation -> safe_landing (assuming short grass)
    3: 1,  # ground -> safe_landing
    4: 3,  # water -> unsafe_terrain
    5: 2,  # car -> high_obstacle  
    6: 3   # clutter -> unsafe_terrain
}

class DroneDeployDataset(Dataset):
    """Dataset for Stage 1: DroneDeploy intermediate fine-tuning."""
    
    def __init__(self, dataset_path: str, split: str = "train", transform=None):
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / "images"
        self.labels_dir = self.dataset_path / "labels"
        self.transform = transform
        
        # Get all image files
        self.image_files = list(self.images_dir.glob("*.tif"))
        
        # Simple train/val split (80/20)
        num_files = len(self.image_files)
        if split == "train":
            self.image_files = self.image_files[:int(0.8 * num_files)]
        else:
            self.image_files = self.image_files[int(0.8 * num_files):]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load corresponding label
        img_name = img_path.stem.replace("-ortho", "")
        label_path = self.labels_dir / f"{img_name}-label.png"
        
        if not label_path.exists():
            # Create dummy label if not found
            label_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            label_img = cv2.imread(str(label_path))
            label_mask = self._convert_rgb_to_class_ids(label_img)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=label_mask)
            image = augmented['image']
            label_mask = augmented['mask']
        
        return {
            'image': image,
            'mask': label_mask.long(),
            'image_path': str(img_path)
        }
    
    def _convert_rgb_to_class_ids(self, label_img):
        """Convert RGB label image to class IDs."""
        h, w = label_img.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Convert BGR image to class IDs
        for bgr_color, class_id in DRONE_DEPLOY_MAPPING.items():
            mask = np.all(label_img == bgr_color, axis=2)
            class_mask[mask] = class_id
        
        return class_mask

class UDDDataset(Dataset):
    """Dataset for Stage 2: UDD task-specific fine-tuning."""
    
    def __init__(self, dataset_path: str, split: str = "train", transform=None):
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / split / "src"  # UDD structure
        self.labels_dir = self.dataset_path / split / "gt"
        self.transform = transform
        
        # Get image files
        self.image_files = list(self.images_dir.glob("*.JPG"))
        if not self.image_files:
            self.image_files = list(self.images_dir.glob("*.jpg"))
        if not self.image_files:
            self.image_files = list(self.images_dir.glob("*.png"))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load corresponding label
        label_path = self.labels_dir / img_path.name
        if not label_path.exists():
            # Try different extension
            label_path = self.labels_dir / (img_path.stem + ".png")
        
        if label_path.exists():
            label_img = cv2.imread(str(label_path))
            label_mask = self._convert_udd_to_landing_classes(label_img)
        else:
            label_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=label_mask)
            image = augmented['image']  
            label_mask = augmented['mask']
        
        return {
            'image': image,
            'mask': label_mask.long(),
            'image_path': str(img_path)
        }
    
    def _convert_udd_to_landing_classes(self, label_img):
        """Convert UDD6 RGB labels to UAV landing classes."""
        h, w = label_img.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Convert BGR to RGB and map to landing classes
        label_rgb = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
        
        for rgb_color, udd_class in UDD6_MAPPING.items():
            # Find pixels matching this UDD class
            mask = np.all(label_rgb == rgb_color, axis=2)
            # Map to landing class
            landing_class = UDD_TO_LANDING[udd_class]
            class_mask[mask] = landing_class
        
        return class_mask

def create_transforms():
    """Create data transforms for training and validation."""
    
    train_transform = A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2), 
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    return train_transform, val_transform

class SimpleBiSeNetV2(nn.Module):
    """Simplified BiSeNetV2 for our staged training."""
    
    def __init__(self, num_classes=19, pretrained_path=None):
        super().__init__()
        
        # Simplified architecture based on BiSeNetV2 principles
        # Context Path (encoder)
        self.context_path = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Spatial Path (high resolution features)
        self.spatial_path = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Feature Fusion Module
        self.fusion = nn.Sequential(
            nn.Conv2d(512 + 128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Classification head
        self.classifier = nn.Conv2d(256, num_classes, 1)
        
        # Load pretrained weights if provided
        if pretrained_path and Path(pretrained_path).exists():
            self._load_pretrained_weights(pretrained_path)
    
    def _load_pretrained_weights(self, pretrained_path):
        """Load pretrained weights with size adaptation."""
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            # Filter out classifier layer if num_classes doesn't match
            model_dict = self.state_dict()
            filtered_dict = {}
            
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
                else:
                    print(f"Skipping layer {k} due to size mismatch")
            
            self.load_state_dict(filtered_dict, strict=False)
            print(f"✅ Loaded pretrained weights from {pretrained_path}")
            
        except Exception as e:
            print(f"⚠️  Could not load pretrained weights: {e}")
    
    def forward(self, x):
        # Get spatial features
        spatial_out = self.spatial_path(x)
        
        # Get context features
        context_out = self.context_path(x)
        
        # Upsample context features to match spatial features
        context_out = F.interpolate(context_out, size=spatial_out.shape[2:], 
                                  mode='bilinear', align_corners=False)
        
        # Fuse features
        fused = torch.cat([context_out, spatial_out], dim=1)
        fused = self.fusion(fused)
        
        # Final classification
        out = self.classifier(fused)
        
        # Upsample to input size
        out = F.interpolate(out, size=(512, 512), mode='bilinear', align_corners=False)
        
        return out

def main():
    """Demo of the staged training approach."""
    
    print("🚁 Staged Fine-Tuning Pipeline Demo")
    print("=" * 40)
    
    # Stage 1: DroneDeploy dataset check
    drone_deploy_path = "../datasets/drone_deploy_dataset_intermediate/dataset-medium"
    if Path(drone_deploy_path).exists():
        print(f"✅ Stage 1 dataset found: {drone_deploy_path}")
        
        # Create Stage 1 dataset
        train_transform, val_transform = create_transforms()
        train_dataset = DroneDeployDataset(drone_deploy_path, "train", train_transform)
        val_dataset = DroneDeployDataset(drone_deploy_path, "val", val_transform)
        
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Val samples: {len(val_dataset)}")
        print(f"   Classes: {list(STAGE1_CLASSES.values())}")
        
        # Test loading a sample
        try:
            sample = train_dataset[0]
            print(f"   Sample image shape: {sample['image'].shape}")
            print(f"   Sample mask shape: {sample['mask'].shape}")
            print(f"   Unique mask values: {torch.unique(sample['mask'])}")
        except Exception as e:
            print(f"   ⚠️  Sample loading error: {e}")
    else:
        print(f"❌ Stage 1 dataset not found: {drone_deploy_path}")
    
    # Stage 2: UDD dataset check
    udd_path = "../datasets/UDD/UDD/UDD6"
    if Path(udd_path).exists():
        print(f"✅ Stage 2 dataset found: {udd_path}")
        
        udd_train = UDDDataset(udd_path, "train", train_transform)
        print(f"   Train samples: {len(udd_train)}")
        print(f"   Classes: {list(STAGE2_CLASSES.values())}")
        
        # Test loading
        try:
            sample = udd_train[0]
            print(f"   Sample image shape: {sample['image'].shape}")
            print(f"   Sample mask shape: {sample['mask'].shape}")  
        except Exception as e:
            print(f"   ⚠️  Sample loading error: {e}")
    else:
        print(f"❌ Stage 2 dataset not found: {udd_path}")
    
    # Model architecture test
    print(f"\n🧠 Model Architecture Test")
    model = SimpleBiSeNetV2(num_classes=len(STAGE1_CLASSES))
    
    # Test forward pass
    test_input = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        output = model(test_input)
    
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\n🚀 Next Steps:")
    print("1. Run Stage 1 training (DroneDeploy)")
    print("2. Save best Stage 1 model")
    print("3. Load Stage 1 model for Stage 2 training (UDD)")
    print("4. Export final model to ONNX")

if __name__ == "__main__":
    main()
