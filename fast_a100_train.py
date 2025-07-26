#!/usr/bin/env python3
"""
FAST A100 Training - No Complexity
==================================
Simple, fast training optimized for A100 with 32 cores
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
import sys

# Add paths
sys.path.append('/workspace/uav_landing')
sys.path.append('/workspace/uav_landing/datasets')
sys.path.append('/workspace/uav_landing/models')

from datasets.semantic_drone_dataset import SemanticDroneDataset, create_semantic_drone_transforms
from models.mobilenetv3_edge_model import EnhancedEdgeLandingNet
import os
from tqdm import tqdm

def setup_fast_training():
    """Setup for maximum A100 speed."""
    # Use ALL 32 cores efficiently
    os.environ['OMP_NUM_THREADS'] = '16'
    os.environ['MKL_NUM_THREADS'] = '16'
    
    # cuDNN optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(16)
    
    print("🚀 Fast A100 setup complete!")

def fast_train():
    """Fast A100 training - no complications."""
    
    setup_fast_training()
    
    device = torch.device('cuda')
    
    # Model
    model = EnhancedEdgeLandingNet(num_classes=6).to(device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Fast transforms
    transforms = create_semantic_drone_transforms(
        input_size=(512, 512),
        is_training=True,
        advanced_augmentation=False  # Fast!
    )
    
    # Dataset - NO memory preloading!
    train_dataset = SemanticDroneDataset(
        data_root='./datasets/semantic_drone_dataset',
        split="train",
        transform=transforms,
        class_mapping="advanced_6_class",
        use_random_crops=False,  # Disable for speed
        preload_to_memory=False  # NO memory preloading!
    )
    
    print(f"Dataset: {len(train_dataset)} samples")
    
    # DataLoader - Use ALL 32 cores!
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16,  # Optimal for A100
        shuffle=True,
        num_workers=24,  # 24 of 32 cores
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    print(f"DataLoader: batch_size=16, workers=24")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = GradScaler('cuda')
    
    print("\n🔥 Starting FAST A100 training...")
    
    # Training loop
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True).long()  # Fix: Convert to long!
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        # Print progress every few batches
        if batch_idx % 5 == 0:
            print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
        
        if batch_idx >= 20:  # Quick test - more batches
            break
    
    avg_loss = total_loss / (batch_idx + 1)
    print(f"\n✅ Fast training complete!")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Batches processed: {batch_idx + 1}")
    print(f"🚀 A100 training is FAST!")

if __name__ == "__main__":
    fast_train() 