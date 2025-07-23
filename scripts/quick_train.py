#!/usr/bin/env python3
"""
Quick Training Script for RTX 4060 Ti
====================================

Minimal, working training script that just works.
No complex caching, fallbacks, or multi-dataset handling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import sys
import os
from pathlib import Path
from tqdm import tqdm
import time

# Add parent directory to path
sys.path.append('..')

def main():
    print("🚁 Quick UAV Landing Training")
    print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    device = torch.device('cuda')
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Try to create dataset
    try:
        from datasets.dronedeploy_1024_dataset import DroneDeploy1024Dataset
        
        print("\n📚 Creating DroneDeploy dataset...")
        
        # Create simple dataset without caching
        train_dataset = DroneDeploy1024Dataset(
            data_root='../../datasets/drone_deploy_dataset_intermediate/dataset-medium',
            split='train',
            patch_size=512,
            stride_factor=0.8,  # Less overlap = fewer patches = faster
            min_valid_pixels=0.2,
            augmentation=True,
            cache_patches=False  # No caching for simplicity
        )
        
        val_dataset = DroneDeploy1024Dataset(
            data_root='../../datasets/drone_deploy_dataset_intermediate/dataset-medium',
            split='val',
            patch_size=512,
            stride_factor=0.8,
            min_valid_pixels=0.2,
            augmentation=False,
            cache_patches=False
        )
        
        print(f"   Train patches: {len(train_dataset)}")
        print(f"   Val patches: {len(val_dataset)}")
        
        if len(train_dataset) == 0:
            print("❌ No training data found!")
            return
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,  # Small batch for RTX 4060 Ti
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        ) if len(val_dataset) > 0 else None
        
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader) if val_loader else 0}")
        
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create simple model
    try:
        from models.edge_landing_net import create_edge_model
        
        model = create_edge_model(
            model_type='standard',
            num_classes=6,
            input_size=512,
            use_uncertainty=False
        ).to(device)
        
        print(f"\n🧠 Model created:")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = GradScaler()
    
    # Simple loss function
    class_weights = train_dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    print(f"\n🚀 Starting training...")
    print(f"   Class weights: {class_weights}")
    
    # Training loop
    model.train()
    epoch_losses = []
    
    try:
        for epoch in range(3):  # Just 3 epochs for testing
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/3")
            print(f"{'='*50}")
            
            epoch_loss = 0.0
            num_batches = 0
            
            # Training
            pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
            for batch_idx, batch in enumerate(pbar):
                try:
                    images = batch['image'].to(device, non_blocking=True)
                    targets = batch['mask'].to(device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass with mixed precision
                    with autocast():
                        outputs = model(images)
                        if isinstance(outputs, dict):
                            predictions = outputs['main']
                        else:
                            predictions = outputs
                        
                        loss = criterion(predictions, targets)
                    
                    # Backward pass
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{epoch_loss/num_batches:.4f}',
                        'GPU': f'{torch.cuda.memory_allocated(0)/1e9:.1f}GB'
                    })
                    
                    # Clear memory
                    del images, targets, outputs, predictions, loss
                    
                    # Memory cleanup every 10 batches
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"❌ Batch {batch_idx} failed: {e}")
                    continue
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
            epoch_losses.append(avg_loss)
            
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"   Average Loss: {avg_loss:.4f}")
            print(f"   GPU Memory: {torch.cuda.memory_allocated(0)/1e9:.1f}GB")
            
            # Simple validation if available
            if val_loader and len(val_loader) > 0:
                model.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc='Validation'):
                        try:
                            images = batch['image'].to(device, non_blocking=True)
                            targets = batch['mask'].to(device, non_blocking=True)
                            
                            with autocast():
                                outputs = model(images)
                                if isinstance(outputs, dict):
                                    predictions = outputs['main']
                                else:
                                    predictions = outputs
                                
                                loss = criterion(predictions, targets)
                            
                            val_loss += loss.item()
                            val_batches += 1
                            
                            del images, targets, outputs, predictions, loss
                            
                        except Exception as e:
                            print(f"❌ Val batch failed: {e}")
                            continue
                
                avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
                print(f"   Validation Loss: {avg_val_loss:.4f}")
                
                model.train()
            
            # Save checkpoint
            if epoch == 2:  # Save final model
                checkpoint_path = Path('../outputs/rtx4060ti_training')
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'epoch_losses': epoch_losses
                }, checkpoint_path / 'quick_model.pth')
                
                print(f"💾 Model saved: {checkpoint_path}/quick_model.pth")
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n Training Complete!")
    print(f"   Final Loss: {epoch_losses[-1] if epoch_losses else 'N/A'}")
    print(f"   GPU Memory Peak: {torch.cuda.max_memory_allocated(0)/1e9:.1f}GB")

if __name__ == "__main__":
    main() 