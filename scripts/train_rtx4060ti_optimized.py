#!/usr/bin/env python3
"""
RTX 4060 Ti Optimized UAV Landing Training
==========================================

Memory-optimized training script for RTX 4060 Ti (8GB VRAM).
Features:
- Mixed precision training (AMP)
- Gradient accumulation for effective larger batch sizes
- Memory-efficient data loading
- Optimal batch sizes and patch sizes
- Real-time GPU memory monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
import argparse
import json
import time
import warnings
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import gc
import psutil

# Import our components
import sys
sys.path.append('..')
from models.edge_landing_net import create_edge_model
from datasets.dronedeploy_1024_dataset import create_dronedeploy_datasets
from datasets.edge_landing_dataset import create_edge_datasets

warnings.filterwarnings('ignore')


class MemoryMonitor:
    """Monitor GPU and CPU memory usage during training."""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.device_name = torch.cuda.get_device_name(0)
            self.total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    def get_gpu_memory(self):
        if not self.gpu_available:
            return 0, 0
        
        allocated = torch.cuda.memory_allocated(0) / 1e9
        cached = torch.cuda.memory_reserved(0) / 1e9
        return allocated, cached
    
    def get_cpu_memory(self):
        return psutil.virtual_memory().percent
    
    def print_memory_stats(self, prefix=""):
        if self.gpu_available:
            allocated, cached = self.get_gpu_memory()
            print(f"{prefix}GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached (Total: {self.total_memory:.1f}GB)")
        
        cpu_percent = self.get_cpu_memory()
        print(f"{prefix}CPU Memory: {cpu_percent:.1f}% used")
    
    def clear_cache(self):
        if self.gpu_available:
            torch.cuda.empty_cache()
        gc.collect()


class RTX4060TiTrainer:
    """
    Memory-optimized trainer for RTX 4060 Ti.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: str = 'cuda'
    ):
        self.device = device
        self.config = config
        self.memory_monitor = MemoryMonitor()
        
        # Setup model
        self.model = model.to(device)
        
        # Mixed precision training
        self.use_amp = config['training']['mixed_precision']
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient accumulation
        self.grad_accum_steps = config['training']['gradient_accumulation_steps']
        
        # Memory optimization
        if config['optimization']['memory_efficient']:
            self._enable_memory_optimizations()
        
        # Training state
        self.epoch = 0
        self.best_miou = 0.0
        
        print(f"🚁 RTX4060TiTrainer initialized:")
        print(f"   Device: {device} ({self.memory_monitor.device_name})")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Mixed precision: {self.use_amp}")
        print(f"   Gradient accumulation: {self.grad_accum_steps}")
        self.memory_monitor.print_memory_stats("   ")
    
    def _enable_memory_optimizations(self):
        """Enable various memory optimizations."""
        
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = self.config['hardware']['benchmark_cudnn']
            print("    CUDA optimizations enabled")
        except:
            print("   ⚠️  CUDA optimizations not available")
        
        # Set memory fraction
        if 'gpu_memory_fraction' in self.config['hardware']:
            fraction = self.config['hardware']['gpu_memory_fraction']
            torch.cuda.set_per_process_memory_fraction(fraction)
            print(f"    GPU memory fraction set to {fraction}")
    
    def create_loss_function(self, class_weights: torch.Tensor) -> nn.Module:
        """Create memory-efficient loss function."""
        
        class MemoryEfficientLoss(nn.Module):
            def __init__(self, class_weights, alpha=0.25, gamma=2.0):
                super().__init__()
                self.class_weights = class_weights
                self.alpha = alpha
                self.gamma = gamma
            
            def forward(self, pred, target):
                # Simple focal loss (memory efficient)
                ce_loss = F.cross_entropy(pred, target, weight=self.class_weights, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                
                return {
                    'total': focal_loss.mean(),
                    'focal': focal_loss.mean()
                }
        
        return MemoryEfficientLoss(class_weights)
    
    def create_data_loaders(self, data_root: str) -> Tuple[DataLoader, DataLoader]:
        """Create memory-optimized data loaders."""
        
        print(f"\n📚 Loading datasets...")
        
        # Try DroneDeploy first
        try:
            datasets = create_dronedeploy_datasets(
                data_root=data_root,
                patch_size=self.config['data']['patch_size'],
                stride_factor=self.config['data']['stride_factor'],
                min_valid_pixels=self.config['data']['min_valid_pixels'],
                augmentation=self.config['data']['augmentation'],
                edge_enhancement=self.config['data']['edge_enhancement']
            )
            
            if datasets['train'] is None or len(datasets['train']) == 0:
                raise ValueError("No DroneDeploy data found")
            
            print(f"    DroneDeploy dataset loaded")
            print(f"   Train patches: {len(datasets['train'])}")
            print(f"   Val patches: {len(datasets['val'])}")
            
        except Exception as e:
            print(f"   ❌ DroneDeploy failed: {e}")
            print(f"   🔄 Falling back to EdgeLandingDataset...")
            
            datasets = create_edge_datasets(
                data_root=data_root,
                input_size=self.config['model']['input_size'],
                extreme_augmentation=True
            )
        
        # Create data loaders with memory optimization
        train_loader = DataLoader(
            
            datasets['train'],
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=self.config['training']['pin_memory'],
            drop_last=True,
            persistent_workers=True,  # Memory optimization
            prefetch_factor=2         # Memory optimization
        )
        
        val_loader = DataLoader(
            datasets['val'],
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            pin_memory=self.config['training']['pin_memory'],
            persistent_workers=True,
            prefetch_factor=2
        )
        
        return train_loader, val_loader, datasets['train']
    
    def train_epoch(self, train_loader, loss_fn, optimizer, class_weights):
        """Memory-efficient training epoch."""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Clear cache before training
        self.memory_monitor.clear_cache()
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device, non_blocking=True)
            targets = batch['mask'].to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    predictions = outputs['main']
                else:
                    predictions = outputs
                
                # Compute loss
                loss_dict = loss_fn(predictions, targets)
                loss = loss_dict['total'] / self.grad_accum_steps  # Scale for gradient accumulation
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1
            
            # Memory management
            if batch_idx % 20 == 0:
                allocated, cached = self.memory_monitor.get_gpu_memory()
                pbar.set_postfix({
                    'loss': f'{loss.item() * self.grad_accum_steps:.4f}',
                    'GPU': f'{allocated:.1f}GB'
                })
            
            # Clear unused variables
            del images, targets, outputs, predictions, loss
            
            # Aggressive memory cleanup every 50 batches
            if batch_idx % 50 == 0:
                self.memory_monitor.clear_cache()
        
        return {'loss': total_loss / num_batches}
    
    def validate_epoch(self, val_loader, loss_fn, class_weights):
        """Memory-efficient validation epoch."""
        
        self.model.eval()
        total_loss = 0.0
        
        # Simple IoU computation
        intersection = torch.zeros(6, device=self.device)
        union = torch.zeros(6, device=self.device)
        
        self.memory_monitor.clear_cache()
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Validation')
            
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device, non_blocking=True)
                targets = batch['mask'].to(self.device, non_blocking=True)
                
                # Forward pass with mixed precision
                with autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    if isinstance(outputs, dict):
                        predictions = outputs['main']
                    else:
                        predictions = outputs
                    
                    # Compute loss
                    loss_dict = loss_fn(predictions, targets)
                    total_loss += loss_dict['total'].item()
                
                # Compute IoU
                pred_classes = predictions.argmax(dim=1)
                
                for class_id in range(6):
                    pred_mask = (pred_classes == class_id)
                    target_mask = (targets == class_id)
                    
                    intersection[class_id] += (pred_mask & target_mask).sum().float()
                    union[class_id] += (pred_mask | target_mask).sum().float()
                
                # Memory cleanup
                del images, targets, outputs, predictions
                
                if batch_idx % 20 == 0:
                    allocated, _ = self.memory_monitor.get_gpu_memory()
                    pbar.set_postfix({'GPU': f'{allocated:.1f}GB'})
        
        # Compute mIoU
        iou_per_class = intersection / (union + 1e-8)
        miou = iou_per_class.mean().item()
        
        return {
            'loss': total_loss / len(val_loader),
            'miou': miou,
            'iou_per_class': iou_per_class.cpu().numpy()
        }
    
    def train(self, data_root: str, checkpoint_dir: str = 'outputs/rtx4060ti_training'):
        """Main training loop."""
        
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🚀 Starting RTX 4060 Ti optimized training...")
        self.memory_monitor.print_memory_stats("Initial ")
        
        # Create data loaders
        train_loader, val_loader, train_dataset = self.create_data_loaders(data_root)
        
        # Get class weights
        class_weights = train_dataset.get_class_weights().to(self.device)
        print(f"   Class weights: {class_weights}")
        
        # Setup training components
        loss_fn = self.create_loss_function(class_weights)
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['training']['num_epochs']
        )
        
        print(f"\n📊 Training Configuration:")
        print(f"   Epochs: {self.config['training']['num_epochs']}")
        print(f"   Batch size: {self.config['training']['batch_size']}")
        print(f"   Gradient accumulation: {self.grad_accum_steps}")
        print(f"   Effective batch size: {self.config['training']['batch_size'] * self.grad_accum_steps}")
        print(f"   Learning rate: {self.config['training']['learning_rate']}")
        
        # Training loop
        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            self.epoch = epoch
            
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{self.config['training']['num_epochs']}")
            print(f"{'='*70}")
            
            # Memory stats before epoch
            self.memory_monitor.print_memory_stats("Pre-epoch ")
            
            # Train
            train_metrics = self.train_epoch(train_loader, loss_fn, optimizer, class_weights)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader, loss_fn, class_weights)
            
            # Update scheduler
            scheduler.step()
            
            # Save checkpoint
            is_best = val_metrics['miou'] > self.best_miou
            if is_best:
                self.best_miou = val_metrics['miou']
                self._save_checkpoint(epoch, val_metrics, checkpoint_dir / 'best_model.pth')
            
            # Regular checkpoint
            if epoch % self.config['checkpointing']['save_every'] == 0:
                self._save_checkpoint(epoch, val_metrics, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"   Train Loss: {train_metrics['loss']:.4f}")
            print(f"   Val Loss: {val_metrics['loss']:.4f}")
            print(f"   Val mIoU: {val_metrics['miou']:.4f}")
            print(f"   Best mIoU: {self.best_miou:.4f}")
            print(f"   Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
            
            # Memory stats after epoch
            self.memory_monitor.print_memory_stats("Post-epoch ")
            
            # Cleanup between epochs
            self.memory_monitor.clear_cache()
        
        print(f"\n Training Complete!")
        print(f"   Best mIoU: {self.best_miou:.4f}")
        print(f"   Best model: {checkpoint_dir}/best_model.pth")
        
        return {
            'best_miou': self.best_miou,
            'final_loss': val_metrics['loss']
        }
    
    def _save_checkpoint(self, epoch, metrics, filepath):
        """Save training checkpoint."""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics,
            'best_miou': self.best_miou,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        print(f"💾 Checkpoint saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='RTX 4060 Ti Optimized UAV Landing Training')
    parser.add_argument('--config', default='../configs/rtx4060ti_config.json', help='Config file')
    parser.add_argument('--data_root', default='../datasets/drone_deploy_dataset_intermediate/dataset-medium', help='Data root')
    parser.add_argument('--checkpoint_dir', default='outputs/rtx4060ti_training', help='Checkpoint directory')
    parser.add_argument('--device', default='cuda', help='Device')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    print(f"🚁 RTX 4060 Ti Optimized UAV Landing Training")
    print(f"   Config: {args.config}")
    print(f"   Data root: {args.data_root}")
    print(f"   Device: {args.device}")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    
    # Create model
    model = create_edge_model(
        model_type=config['model']['type'],
        num_classes=config['model']['num_classes'],
        input_size=config['model']['input_size'],
        use_uncertainty=config['model']['use_uncertainty']
    )
    
    # Create trainer
    trainer = RTX4060TiTrainer(
        model=model,
        config=config,
        device=args.device
    )
    
    try:
        # Start training
        results = trainer.train(
            data_root=args.data_root,
            checkpoint_dir=args.checkpoint_dir
        )
        
        print(f"\n Training completed successfully!")
        print(f"   Final mIoU: {results['best_miou']:.4f}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 