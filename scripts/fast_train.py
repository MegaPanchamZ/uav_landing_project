#!/usr/bin/env python3

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import numpy as np
from tqdm import tqdm
import gc

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "datasets"))

from enhanced_uav_detector import EdgeLandingNet
from dronedeploy_1024_dataset import DroneDeploy1024Dataset

def setup_cuda():
    """Optimize CUDA settings for WSL."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Conservative memory settings for WSL
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)
        torch.cuda.empty_cache()

def create_wsl_dataloader(data_root: str, split: str, batch_size: int = 6):
    """Create WSL-optimized dataloader."""
    
    dataset = DroneDeploy1024Dataset(
        data_root=data_root,
        split=split,
        patch_size=512,
        stride_factor=0.75,  # Fewer patches = faster
        min_valid_pixels=0.2,  # Higher threshold = fewer patches
        augmentation=(split == 'train'),
        cache_patches=True
    )
    
    # WSL-safe DataLoader settings
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=2,  # Reduced for WSL stability
        pin_memory=False,  # Disabled for WSL
        persistent_workers=False,  # Disabled for WSL
        prefetch_factor=2,  # Reduced prefetch
        drop_last=True
    )
    
    return dataloader, dataset

def fast_train():
    """WSL-optimized training."""
    
    print("🚀 WSL-Optimized UAV Landing Training")
    print(f"   Platform: WSL")
    print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU: {gpu_name}")
        print(f"   GPU Memory: {gpu_memory:.1f}GB")
    
    # Setup
    setup_cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    data_root = "../../datasets/drone_deploy_dataset_intermediate/dataset-medium"
    
    print("\n📚 Loading datasets...")
    train_loader, train_dataset = create_wsl_dataloader(data_root, 'train', batch_size=6)
    val_loader, val_dataset = create_wsl_dataloader(data_root, 'val', batch_size=8)
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Clear memory before model creation
    gc.collect()
    torch.cuda.empty_cache()
    
    # Model
    model = EdgeLandingNet(num_classes=6, input_size=512).to(device)
    print(f"🧠 Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Optimizer with conservative learning rate
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,  # More conservative
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Simple scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=0.8
    )
    
    # Loss with class weights
    try:
        class_weights = train_dataset.get_class_weights().to(device)
        print(f"   Class weights: {class_weights}")
    except:
        class_weights = torch.ones(6).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Mixed precision
    scaler = GradScaler()
    
    # Training loop
    model.train()
    
    print(f"\n🚀 Starting WSL training (3 epochs)...")
    
    for epoch in range(3):
        start_time = time.time()
        total_loss = 0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/3",
            leave=False
        )
        
        batch_count = 0
        for batch_idx, batch in enumerate(progress_bar):
            try:
                images = batch['image'].to(device, non_blocking=False)
                masks = batch['mask'].to(device, non_blocking=False)
                
                optimizer.zero_grad(set_to_none=True)
                
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                batch_count += 1
                
                # Update progress
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/batch_count:.4f}',
                    'GPU': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
                })
                
                # Memory cleanup every 50 batches
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                
            except Exception as e:
                print(f"   ⚠️  Batch {batch_idx} failed: {e}")
                continue
        
        scheduler.step()
        epoch_time = time.time() - start_time
        avg_loss = total_loss / max(batch_count, 1)
        
        print(f"   Epoch {epoch+1}: {avg_loss:.4f} loss, {epoch_time:.1f}s")
        
        # Memory cleanup after epoch
        torch.cuda.empty_cache()
        gc.collect()
        
        # Quick validation
        if (epoch + 1) % 2 == 0:
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation", leave=False):
                    try:
                        images = batch['image'].to(device, non_blocking=False)
                        masks = batch['mask'].to(device, non_blocking=False)
                        
                        with autocast():
                            outputs = model(images)
                            loss = criterion(outputs, masks)
                        
                        val_loss += loss.item()
                        val_batches += 1
                        
                    except Exception as e:
                        continue
            
            avg_val_loss = val_loss / max(val_batches, 1)
            print(f"   Validation: {avg_val_loss:.4f} loss")
            
            model.train()
    
    # Save model
    output_dir = Path("../outputs/wsl_training")
    output_dir.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': 3,
        'loss': avg_loss,
    }, output_dir / "wsl_model.pth")
    
    print(f"\n✅ WSL training complete!")
    print(f"   Model saved: {output_dir}/wsl_model.pth")

if __name__ == "__main__":
    fast_train() 