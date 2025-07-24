#!/usr/bin/env python3

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import gc

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "datasets"))
sys.path.append(str(project_root / "models"))

from edge_landing_net import EdgeLandingNet
from dronedeploy_1024_dataset import DroneDeploy1024Dataset

def simple_train():
    """Ultra-simple training for WSL."""
    
    print("🚁 Simple UAV Training (WSL-Safe)")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.set_per_process_memory_fraction(0.7)
    
    # Data
    data_root = "../../datasets/drone_deploy_dataset_intermediate/dataset-medium"
    
    print("\n📚 Loading dataset...")
    train_dataset = DroneDeploy1024Dataset(
        data_root=data_root,
        split='train',
        patch_size=512,
        stride_factor=0.8,  # Fewer patches
        min_valid_pixels=0.25,  # Higher threshold
        augmentation=True,
        cache_patches=True
    )
    
    # Simple DataLoader (no multiprocessing)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # Small and safe
        shuffle=True,
        num_workers=0,  # No multiprocessing
        pin_memory=False
    )
    
    print(f"   Train patches: {len(train_dataset)}")
    print(f"   Train batches: {len(train_loader)}")
    
    # Model
    model = EdgeLandingNet(num_classes=6, input_size=512).to(device)
    print(f"🧠 Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Simple optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )
    
    # Simple loss
    criterion = nn.CrossEntropyLoss()
    
    # Training
    model.train()
    
    print(f"\n🚀 Starting training (2 epochs)...")
    
    for epoch in range(2):
        start_time = time.time()
        total_loss = 0
        processed_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/2")):
            try:
                # Move to device
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                processed_batches += 1
                
                # Memory cleanup
                if batch_idx % 25 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                
            except Exception as e:
                print(f"   ⚠️  Batch {batch_idx} failed: {e}")
                continue
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / max(processed_batches, 1)
        
        print(f"   Epoch {epoch+1}: {avg_loss:.4f} loss, {epoch_time:.1f}s, {processed_batches} batches")
        
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save model
    output_dir = Path("../outputs/simple_training")
    output_dir.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': 2,
        'loss': avg_loss,
    }, output_dir / "simple_model.pth")
    
    print(f"\n✅ Training complete!")
    print(f"   Model saved: {output_dir}/simple_model.pth")
    print(f"   GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

if __name__ == "__main__":
    simple_train() 