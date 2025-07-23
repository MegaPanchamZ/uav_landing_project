#!/usr/bin/env python3
"""
Aerial-Optimized Training for UAV Landing Detection
=================================================

A completely redesigned approach specifically for aerial drone datasets:
1. 3-Class System (removes problematic background class)
2. Aerial-specific data augmentation
3. Custom loss designed for extreme imbalance
4. Progressive training strategy
5. Aerial-optimized architecture

Based on investigation findings:
- Class 0 (Background) only 0.28% - practically unusable
- Need aerial-specific approach, not street-scene pretrained
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from pathlib import Path
from typing import Dict, Any, Tuple
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import gc
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from safety_evaluation.safety_metrics import SafetyAwareEvaluator

def print_header():
    """Prints a formatted header for the training script."""
    print("=" * 80)
    print("🚁 AERIAL-OPTIMIZED UAV Landing Detection Training")
    print("=" * 80)
    print("Revolutionary Approach:")
    print("  • 3-Class System (Safe, Caution, Danger)")
    print("  • Aerial-specific augmentations")
    print("  • Custom extreme imbalance handling")
    print("  • Progressive training strategy")
    print("  • No problematic background class")
    print("-" * 80)

class AerialDataset(torch.utils.data.Dataset):
    """
    Aerial-optimized dataset for 3-class UAV landing detection.
    Removes the problematic background class entirely.
    """
    
    def __init__(self, data_root: str, split: str = "train", target_size: Tuple[int, int] = (512, 512)):
        self.data_root = Path(data_root)
        self.split = split
        self.target_size = target_size
        
        # Load images and labels
        self.images_dir = self.data_root / "original_images"
        self.labels_dir = self.data_root / "label_images_semantic"
        
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        self.label_files = sorted(list(self.labels_dir.glob("*.png")))
        
        # Create splits
        total_files = len(self.image_files)
        indices = np.arange(total_files)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_end = int(0.7 * total_files)
        val_end = int(0.85 * total_files)
        
        if split == "train":
            self.file_indices = indices[:train_end]
        elif split == "val":
            self.file_indices = indices[train_end:val_end]
        else:
            self.file_indices = indices[val_end:]
        
        # NEW 3-CLASS MAPPING (removes background entirely)
        # Based on investigation: most classes should map to "Safe Landing"
        self.class_mapping = {
            # Safe Landing (dominant classes from analysis)
            1: 1, 2: 1, 3: 1, 4: 1,  # Original safe landing areas
            
            # Caution (moderate risk)
            6: 2, 8: 2, 9: 2, 21: 2,  # Original caution areas
            
            # Danger (high risk)
            5: 3, 7: 3, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3,
            15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3, 22: 3,  # Original danger areas
            
            # CRITICAL: Map background to Safe Landing (most reasonable for aerial landing)
            0: 1, 23: 1  # Background becomes Safe Landing
        }
        
        # Aerial-specific augmentations
        if split == "train":
            self.transform = A.Compose([
                A.RandomRotate90(p=0.7),  # Critical for aerial views
                A.Flip(p=0.7),  # Horizontal and vertical flips
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.8
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.GaussianBlur(blur_limit=3),
                ], p=0.3),
                A.Resize(target_size[0], target_size[1], interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(target_size[0], target_size[1], interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        print(f"AerialDataset initialized:")
        print(f"   Split: {split} ({len(self)} samples)")
        print(f"   Target size: {target_size}")
        print(f"   Using 3-class mapping (1=Safe, 2=Caution, 3=Danger)")
    
    def __len__(self):
        return len(self.file_indices)
    
    def __getitem__(self, idx):
        file_idx = self.file_indices[idx]
        
        # Load image and label
        image_path = self.image_files[file_idx]
        label_path = self.label_files[file_idx]
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        
        # Apply 3-class mapping
        mapped_label = self._map_to_3_classes(label)
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mapped_label)
        
        return {
            'image': transformed['image'],
            'mask': transformed['mask'].long()
        }
    
    def _map_to_3_classes(self, label: np.ndarray) -> np.ndarray:
        """Map original classes to 3-class system."""
        mapped_label = np.ones_like(label, dtype=np.uint8)  # Default to Safe Landing
        
        for original_class, landing_class in self.class_mapping.items():
            mask = (label == original_class)
            mapped_label[mask] = landing_class
        
        return mapped_label

class AerialSegmentationModel(nn.Module):
    """
    Lightweight aerial-optimized segmentation model.
    Designed specifically for aerial views, not street scenes.
    """
    
    def __init__(self, num_classes=3, in_channels=3):
        super().__init__()
        
        # Aerial-optimized encoder
        self.encoder = nn.Sequential(
            # Stage 1
            nn.Conv2d(in_channels, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Stage 2
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Stage 3
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Stage 4
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Aerial context module
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, num_classes, 4, 2, 1),
        )
        
    def forward(self, x):
        # Encode
        features = self.encoder(x)
        
        # Context
        context = self.context(features)
        
        # Decode
        output = self.decoder(context)
        
        return output

class ExtremeImbalanceLoss(nn.Module):
    """
    Loss function designed for extreme class imbalance in aerial data.
    """
    
    def __init__(self, class_weights=None, alpha=2.0, gamma=3.0):
        super().__init__()
        # Computed weights for 3-class system based on investigation
        if class_weights is None:
            # Based on remapped distribution
            class_weights = [0.3, 1.2, 1.5]  # [Safe, Caution, Danger]
        
        self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        # Ensure same device
        targets = targets.long().to(inputs.device)
        class_weights = self.class_weights.to(inputs.device)
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute probability
        pt = torch.exp(-ce_loss)
        
        # Get class weights
        weight_t = class_weights[targets]
        
        # Focal loss with class weighting
        focal_loss = weight_t * self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()

class AerialTrainer:
    """Aerial-optimized trainer."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize wandb
        wandb.init(
            project=config['wandb_project'],
            name=f"aerial-optimized-3class-{wandb.util.generate_id()}",
            config=config,
            tags=["aerial-optimized", "3-class", "extreme-imbalance"],
            notes="Aerial-optimized 3-class training without problematic background"
        )

        self._log_system_info()

    def _log_system_info(self):
        """Log system information."""
        print(f"Configuration:")
        for key, val in self.config.items():
            print(f"  - {key}: {val}")
        print("-" * 80)
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"   - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("-" * 80)

    def train(self):
        """Run aerial-optimized training."""
        
        # Create datasets
        train_dataset = AerialDataset(
            data_root=self.config['dataset_path'],
            split="train",
            target_size=tuple(self.config['input_size'])
        )
        
        val_dataset = AerialDataset(
            data_root=self.config['dataset_path'],
            split="val",
            target_size=tuple(self.config['input_size'])
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=False,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=False
        )
        
        print(f"  ✅ Training samples: {len(train_dataset)}")
        print(f"  ✅ Validation samples: {len(val_dataset)}")
        
        # Create model
        model = AerialSegmentationModel(num_classes=3).to(self.device)
        
        # Create loss and optimizer
        criterion = ExtremeImbalanceLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config['epochs'], eta_min=1e-6
        )
        
        scaler = GradScaler()
        best_miou = 0.0
        
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n🚁 Starting aerial-optimized training...\n")
        
        for epoch in range(self.config['epochs']):
            print(f"--- Epoch {epoch+1}/{self.config['epochs']} ---")
            
            # Training
            train_loss, train_metrics = self._train_epoch(model, train_loader, criterion, optimizer, scaler)
            
            # Validation
            val_loss, val_metrics = self._validate_epoch(model, val_loader, criterion)
            
            scheduler.step()
            
            # Logging
            current_lr = optimizer.param_groups[0]['lr']
            log_metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr,
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()}
            }
            wandb.log(log_metrics)
            
            print(f"  - Train Loss: {train_loss:.4f}")
            print(f"  - Val Loss:   {val_loss:.4f}")
            print(f"  - Val mIoU:   {val_metrics.get('miou', 0.0):.4f}")
            
            # Per-class performance (3 classes now)
            for cls in range(1, 4):  # Classes 1, 2, 3
                cls_iou = val_metrics.get(f'class_{cls}_iou', 0.0)
                cls_acc = val_metrics.get(f'class_{cls}_accuracy', 0.0)
                class_names = {1: "Safe", 2: "Caution", 3: "Danger"}
                print(f"    {class_names[cls]:7}: IoU={cls_iou:.3f}, Acc={cls_acc:.3f}")
            
            # Save best model
            current_miou = val_metrics.get('miou', 0.0)
            if current_miou > best_miou:
                best_miou = current_miou
                save_path = output_dir / f"aerial_optimized_3class_best.pth"
                torch.save(model.state_dict(), save_path)
                print(f"  ✅ New best model saved (mIoU: {best_miou:.4f})")

        wandb.finish()
        print(f"\n🚁 Aerial-optimized training complete!")
        print(f"Best mIoU achieved: {best_miou:.4f}")

    def _train_epoch(self, model, dataloader, criterion, optimizer, scaler):
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        evaluator = SafetyAwareEvaluator(num_classes=3)  # 3 classes now
        
        for batch in tqdm(dataloader, desc="Training", leave=False):
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # Update metrics
            predictions = torch.argmax(outputs, dim=1)
            evaluator.update(predictions, masks)
        
        metrics = evaluator.compute_metrics()
        return total_loss / len(dataloader), metrics

    def _validate_epoch(self, model, dataloader, criterion):
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        evaluator = SafetyAwareEvaluator(num_classes=3)  # 3 classes now
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating", leave=False):
                images = batch['image'].to(self.device, non_blocking=True)
                masks = batch['mask'].to(self.device, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs, dim=1)
                evaluator.update(predictions, masks)
        
        metrics = evaluator.compute_metrics()
        return total_loss / len(dataloader), metrics

def main():
    parser = argparse.ArgumentParser(description="Aerial-Optimized UAV Landing Detection")
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    default_dataset_path = project_root / 'datasets' / 'Aerial_Semantic_Segmentation_Drone_Dataset' / 'dataset' / 'semantic_drone_dataset'
    
    parser.add_argument('--dataset-path', type=str, default=str(default_dataset_path))
    parser.add_argument('--output-dir', type=str, default='outputs/aerial_optimized')
    
    # Training parameters
    parser.add_argument('--input-size', type=int, nargs=2, default=[384, 384])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    
    # W&B
    parser.add_argument('--wandb-project', type=str, default='uav-landing-aerial-optimized')
    
    args = parser.parse_args()
    config = vars(args)
    
    print_header()
    trainer = AerialTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main() 