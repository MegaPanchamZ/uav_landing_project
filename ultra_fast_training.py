#!/usr/bin/env python3
"""
Ultra-Fast Staged Fine-Tuning Script

Optimized for speed and low memory usage:
- Smaller input size (256x256 instead of 512x512)
- Mixed precision training
- Efficient data loading
- Gradient accumulation for small batches
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
import time
import json
from tqdm import tqdm
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from staged_training import (
    DroneDeployDataset, UDDDataset, STAGE1_CLASSES, STAGE2_CLASSES,
    DRONE_DEPLOY_MAPPING, UDD6_MAPPING, UDD_TO_LANDING
)

class UltraFastBiSeNet(nn.Module):
    """Ultra-lightweight BiSeNet for fast training."""
    
    def __init__(self, num_classes=7):
        super().__init__()
        
        # Much smaller network for speed
        self.backbone = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Downsample 1 (128x128)
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Downsample 2 (64x64)
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Feature processing
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Simple upsampling path
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Conv2d(32, num_classes, 1)
    
    def forward(self, x):
        # Encode
        features = self.backbone(x)
        
        # Decode
        x = self.decoder(features)
        
        # Upsample to match input size
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        
        # Classify
        x = self.classifier(x)
        
        return x

def create_fast_transforms():
    """Create fast transforms with smaller input size."""
    
    train_transform = A.Compose([
        A.Resize(256, 256),  # Much smaller for speed
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    return train_transform, val_transform

class UltraFastTrainer:
    """Ultra-fast trainer with optimizations."""
    
    def __init__(self, device='auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        # Enable mixed precision for speed
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        if self.scaler:
            print("⚡ Mixed precision training enabled")
        
    def train_stage1(self, dataset_path, epochs=8, batch_size=8, lr=2e-4, accumulation_steps=1):
        """Ultra-fast Stage 1 training."""
        
        print(f"\n🌍 ULTRA-FAST STAGE 1: DroneDeploy Fine-Tuning")
        print("=" * 55)
        
        # Fast transforms
        train_transform, val_transform = create_fast_transforms()
        
        # Create datasets with efficient loading
        train_dataset = DroneDeployDataset(dataset_path, "train", train_transform)
        val_dataset = DroneDeployDataset(dataset_path, "val", val_transform)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        # Fast data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=4,  # More workers
            pin_memory=True,
            persistent_workers=True  # Keep workers alive
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False, 
            num_workers=2,
            pin_memory=True
        )
        
        # Ultra-fast model
        model = UltraFastBiSeNet(num_classes=len(STAGE1_CLASSES))
        model.to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Optimizations
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr*2, epochs=epochs, steps_per_epoch=len(train_loader)
        )
        
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        
        # Ultra-fast training loop
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            model.train()
            train_loss = 0
            optimizer.zero_grad()
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for i, batch in enumerate(train_pbar):
                images = batch['image'].to(self.device, non_blocking=True)
                masks = batch['mask'].to(self.device, non_blocking=True)
                
                # Mixed precision forward pass
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, masks) / accumulation_steps
                    
                    self.scaler.scale(loss).backward()
                    
                    if (i + 1) % accumulation_steps == 0:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, masks) / accumulation_steps
                    loss.backward()
                    
                    if (i + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                
                train_loss += loss.item() * accumulation_steps
                train_pbar.set_postfix({
                    'loss': loss.item() * accumulation_steps,
                    'lr': optimizer.param_groups[0]['lr']
                })
            
            # Fast validation
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation", leave=False):
                    images = batch['image'].to(self.device, non_blocking=True)
                    masks = batch['mask'].to(self.device, non_blocking=True)
                    
                    if self.scaler:
                        with torch.cuda.amp.autocast():
                            outputs = model(images)
                            loss = criterion(outputs, masks)
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                    
                    val_loss += loss.item()
            
            # Update metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1:2d}: Train: {avg_train_loss:.4f}, "
                  f"Val: {avg_val_loss:.4f}, Time: {epoch_time:.1f}s")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 'ultra_stage1_best.pth')
        
        print(f"✅ Stage 1 complete! Best val loss: {best_val_loss:.4f}")
        return model, history
    
    def train_stage2(self, dataset_path, stage1_model_path, epochs=10, batch_size=8, lr=1e-4):
        """Ultra-fast Stage 2 training."""
        
        print(f"\n🚁 ULTRA-FAST STAGE 2: UDD6 Task-Specific Fine-Tuning")
        print("=" * 60)
        
        # Fast transforms
        train_transform, val_transform = create_fast_transforms()
        
        # UDD datasets
        train_dataset = UDDDataset(dataset_path, "train", train_transform)
        val_dataset = UDDDataset(dataset_path, "val", val_transform)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        if len(train_dataset) == 0:
            print("❌ No training samples!")
            return None, None
        
        # Fast data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=2, pin_memory=True
        )
        
        # Load Stage 1 model and adapt
        model = UltraFastBiSeNet(num_classes=len(STAGE1_CLASSES))
        model.load_state_dict(torch.load(stage1_model_path, map_location='cpu'))
        
        # Replace classifier for Stage 2
        model.classifier = nn.Conv2d(32, len(STAGE2_CLASSES), 1)
        model.to(self.device)
        
        # Lower learning rate for fine-tuning
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr*1.5, epochs=epochs, steps_per_epoch=len(train_loader)
        )
        
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'val_iou': []}
        
        # Training loop
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            model.train()
            train_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images = batch['image'].to(self.device, non_blocking=True)
                masks = batch['mask'].to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            iou_scores = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation", leave=False):
                    images = batch['image'].to(self.device, non_blocking=True)
                    masks = batch['mask'].to(self.device, non_blocking=True)
                    
                    if self.scaler:
                        with torch.cuda.amp.autocast():
                            outputs = model(images)
                            loss = criterion(outputs, masks)
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                    
                    val_loss += loss.item()
                    
                    # Quick IoU calculation
                    preds = outputs.argmax(dim=1)
                    for class_id in [1, 2]:  # Main landing classes
                        pred_mask = (preds == class_id)
                        true_mask = (masks == class_id)
                        intersection = (pred_mask & true_mask).sum().float()
                        union = (pred_mask | true_mask).sum().float()
                        if union > 0:
                            iou_scores.append(intersection / union)
            
            # Update metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            mean_iou = torch.mean(torch.stack(iou_scores)) if iou_scores else 0
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_iou'].append(mean_iou.item() if isinstance(mean_iou, torch.Tensor) else mean_iou)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1:2d}: Train: {avg_train_loss:.4f}, "
                  f"Val: {avg_val_loss:.4f}, IoU: {mean_iou:.3f}, Time: {epoch_time:.1f}s")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 'ultra_stage2_best.pth')
        
        print(f"✅ Stage 2 complete! Best val loss: {best_val_loss:.4f}")
        return model, history

def main():
    """Ultra-fast main pipeline."""
    
    print("⚡ ULTRA-FAST Staged Fine-Tuning Pipeline")
    print("=" * 50)
    
    trainer = UltraFastTrainer()
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        # Adjust batch size based on GPU memory
        if gpu_memory < 10:
            batch_size = 6
            print("🔋 Using smaller batch size for 8GB GPU")
        else:
            batch_size = 8
    else:
        batch_size = 4
    
    # Stage 1: DroneDeploy
    drone_deploy_path = "../datasets/drone_deploy_dataset_intermediate/dataset-medium"
    if Path(drone_deploy_path).exists():
        print("🚀 Starting Ultra-Fast Stage 1...")
        stage1_model, stage1_history = trainer.train_stage1(
            drone_deploy_path, 
            epochs=6,  # Fewer epochs for speed
            batch_size=batch_size, 
            lr=3e-4,   # Higher learning rate
            accumulation_steps=1
        )
        print("✅ Stage 1 completed!")
    else:
        print(f"❌ DroneDeploy dataset not found")
        return
    
    # Stage 2: UDD6
    udd_path = "../datasets/UDD/UDD/UDD6"
    if Path(udd_path).exists() and Path("ultra_stage1_best.pth").exists():
        print("🚀 Starting Ultra-Fast Stage 2...")
        stage2_model, stage2_history = trainer.train_stage2(
            udd_path, "ultra_stage1_best.pth", 
            epochs=8,  # Fewer epochs
            batch_size=batch_size, 
            lr=1e-4
        )
        
        if stage2_model is not None:
            print("🎉 Ultra-fast staged training completed!")
        else:
            print("❌ Stage 2 failed")
    else:
        print(f"❌ UDD dataset or Stage 1 model missing")

if __name__ == "__main__":
    main()
