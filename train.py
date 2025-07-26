#!/usr/bin/env python3
"""
UAV Landing System - Generalized Training Script
===============================================

Unified training script that automatically adapts to:
- Different hardware configurations (CPU, single GPU, multi-GPU)
- Various dataset configurations
- Memory constraints
- Progressive training strategy

Supports:
- Progressive 3-stage training (SDD → DroneDeploy → UDD6)
- Automatic hardware detection and optimization
- Mixed precision training
- W&B integration
- Comprehensive logging and checkpointing

OPTIMIZATIONS:
- Persistent workers to eliminate process spawn overhead
- Improved num_workers logic based on CPU cores
- Prefetching for smoother data pipeline
- DataLoader profiling tool for empirical optimization
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
import json
import time
import warnings
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import platform
import psutil
import subprocess
from collections import defaultdict
from datetime import datetime

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  W&B not available. Install with: pip install wandb")

warnings.filterwarnings('ignore')

# Import our components
from models.mobilenetv3_edge_model import create_edge_model
from datasets.semantic_drone_dataset import SemanticDroneDataset, create_semantic_drone_transforms
from datasets.dronedeploy_1024_dataset import DroneDeploy1024Dataset, create_dronedeploy_datasets
from datasets.udd6_dataset import UDD6Dataset, create_udd6_transforms
from losses.safety_aware_losses import CombinedSafetyLoss


class HardwareDetector:
    """Automatically detect and optimize for available hardware."""
    
    def __init__(self):
        self.system_info = self._detect_system()
        self.gpu_info = self._detect_gpu()
        self.cpu_info = self._detect_cpu()
        self.memory_info = self._detect_memory()
        
    def _detect_system(self) -> Dict:
        """Detect system information."""
        return {
            'platform': platform.system(),
            'machine': platform.machine(),
            'python_version': platform.python_version(),
            'processor': platform.processor()
        }
    
    def _detect_gpu(self) -> Dict:
        """Detect GPU information and capabilities."""
        gpu_info = {
            'available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'devices': []
        }
        
        if gpu_info['available']:
            for i in range(gpu_info['device_count']):
                props = torch.cuda.get_device_properties(i)
                device_info = {
                    'id': i,
                    'name': props.name,
                    'memory': props.total_memory / 1024**3,  # GB
                    'compute_capability': f"{props.major}.{props.minor}",
                    'multi_processor_count': props.multi_processor_count
                }
                gpu_info['devices'].append(device_info)
        
        return gpu_info
    
    def _detect_cpu(self) -> Dict:
        """Detect CPU information."""
        return {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None
        }
    
    def _detect_memory(self) -> Dict:
        """Detect system memory."""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / 1024**3,
            'available_gb': memory.available / 1024**3,
            'usage_percent': memory.percent
        }
    
    def get_optimal_config(self) -> Dict:
        """Get optimal training configuration based on hardware."""
        config = {
            'device': 'cpu',
            'batch_size': 4,
            'num_workers': 2,
            'pin_memory': False,
            'mixed_precision': False,
            'gradient_accumulation': 1,
            ## OPTIMIZATION ##: Add persistent_workers and prefetch_factor to config
            'persistent_workers': False,
            'prefetch_factor': 2
        }
        
        # GPU optimizations
        if self.gpu_info['available']:
            config['device'] = 'cuda'
            config['pin_memory'] = True
            config['mixed_precision'] = True
            
            # Get primary GPU
            gpu = self.gpu_info['devices'][0]
            gpu_memory = gpu['memory']
            gpu_name = gpu['name'].lower()
            
            ## OPTIMIZATION ##: More aggressive num_workers based on CPU cores
            # A good rule of thumb is to start with the number of logical cores.
            # We cap it at 16 to avoid excessive resource usage on high-end servers.
            cpu_cores = self.cpu_info['logical_cores']
            base_num_workers = min(cpu_cores, 16)
            
            # Optimize based on GPU type and memory
            if 'a100' in gpu_name:
                # A100 optimizations
                if gpu_memory >= 40:  # A100-40GB or A100-80GB
                    config['batch_size'] = 64
                    config['num_workers'] = base_num_workers
                else:
                    config['batch_size'] = 32
                    config['num_workers'] = max(6, base_num_workers // 2)
            elif 'v100' in gpu_name:
                # V100 optimizations
                config['batch_size'] = 32
                config['num_workers'] = max(4, base_num_workers // 2)
            elif 'rtx' in gpu_name or 'titan' in gpu_name:
                # RTX series optimizations
                if gpu_memory >= 24:  # RTX 3090/4090, Titan
                    config['batch_size'] = 16
                    config['num_workers'] = max(4, base_num_workers // 2)
                else:
                    config['batch_size'] = 8
                    config['num_workers'] = max(4, base_num_workers // 3)
            elif 'gtx' in gpu_name:
                # GTX series (older)
                config['batch_size'] = 4
                config['num_workers'] = 4
                config['mixed_precision'] = float(gpu.get('compute_capability', '0.0')) >= 7.0
            else:
                # Generic GPU
                if gpu_memory >= 8:
                    config['batch_size'] = 8
                    config['num_workers'] = max(4, base_num_workers // 3)
                else:
                    config['batch_size'] = 4
                    config['num_workers'] = 4
            
            ## OPTIMIZATION ##: Enable persistent workers if we have workers
            if config['num_workers'] > 0:
                config['persistent_workers'] = True
        
        # CPU optimizations
        cpu_cores = min(self.cpu_info['logical_cores'], 16)  # Cap at 16
        if not self.gpu_info['available']:
            # CPU-only training
            config['num_workers'] = max(1, cpu_cores // 2)
        
        # Memory-based adjustments
        available_memory = self.memory_info['available_gb']
        if available_memory < 8:
            # Reduce workers if system RAM is very low to avoid thrashing
            config['num_workers'] = max(1, config['num_workers'] // 2)
            config['batch_size'] = max(1, config['batch_size'] // 2)
        elif available_memory > 64:
            # Lots of RAM - can use more workers
            if config['num_workers'] < cpu_cores:
                config['num_workers'] = min(config['num_workers'] * 2, cpu_cores)
        
        # Ensure num_workers is at least 0
        config['num_workers'] = max(0, config['num_workers'])
        
        return config
    
    def print_system_info(self):
        """Print detailed system information."""
        print("🖥️  System Information:")
        print(f"   Platform: {self.system_info['platform']} {self.system_info['machine']}")
        print(f"   Python: {self.system_info['python_version']}")
        
        print(f"\n💾 Memory: {self.memory_info['total_gb']:.1f} GB total, {self.memory_info['available_gb']:.1f} GB available")
        print(f"🔧 CPU: {self.cpu_info['physical_cores']} cores ({self.cpu_info['logical_cores']} threads)")
        
        if self.gpu_info['available']:
            print(f"\n🚀 GPU Information:")
            for gpu in self.gpu_info['devices']:
                print(f"   GPU {gpu['id']}: {gpu['name']}")
                print(f"   Memory: {gpu['memory']:.1f} GB")
                print(f"   Compute: {gpu['compute_capability']}")
        else:
            print("\n⚠️  No GPU detected - using CPU training")


class MultiDatasetLoss(nn.Module):
    """Adaptive loss function for multi-dataset progressive training."""
    
    def __init__(self, num_classes=6, alpha=0.25, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        
        # Dataset-specific weights
        self.dataset_weights = {
            'semantic_drone': {'focal': 0.4, 'dice': 0.3, 'boundary': 0.2, 'consistency': 0.1},
            'dronedeploy': {'focal': 0.5, 'dice': 0.3, 'boundary': 0.15, 'consistency': 0.05},
            'udd6': {'focal': 0.45, 'dice': 0.25, 'boundary': 0.2, 'consistency': 0.1}
        }
    
    def _focal_loss(self, pred, target, class_weights=None):
        """Focal loss for handling class imbalance."""
        ce_loss = F.cross_entropy(pred, target, weight=class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
    def _dice_loss(self, pred, target):
        """Dice loss for segmentation quality."""
        smooth = 1e-5
        pred_soft = F.softmax(pred, dim=1)
        
        dice_loss = 0
        for class_id in range(self.num_classes):
            pred_class = pred_soft[:, class_id]
            target_class = (target == class_id).float()
            
            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()
            
            dice_coeff = (2 * intersection + smooth) / (union + smooth)
            dice_loss += 1 - dice_coeff
        
        return dice_loss / self.num_classes
    
    def _boundary_loss(self, pred, target):
        """Boundary loss for edge preservation."""
        pred_edges = self._compute_edges(pred.argmax(1).float())
        target_edges = self._compute_edges(target.float())
        return F.mse_loss(pred_edges, target_edges)
    
    def _compute_edges(self, tensor):
        """Compute edges using Sobel operator."""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=tensor.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=tensor.device).unsqueeze(0).unsqueeze(0)
        
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(1)
        
        edges_x = F.conv2d(tensor, sobel_x, padding=1)
        edges_y = F.conv2d(tensor, sobel_y, padding=1)
        
        return torch.sqrt(edges_x ** 2 + edges_y ** 2)
    
    def forward(self, pred, target, dataset_source='dronedeploy', class_weights=None):
        """Compute adaptive loss based on dataset source."""
        weights = self.dataset_weights.get(dataset_source, self.dataset_weights['dronedeploy'])
        
        # Compute component losses
        focal = self._focal_loss(pred, target, class_weights)
        dice = self._dice_loss(pred, target)
        boundary = self._boundary_loss(pred, target)
        
        total_loss = (
            weights['focal'] * focal +
            weights['dice'] * dice +
            weights['boundary'] * boundary
        )
        
        return {
            'total': total_loss,
            'focal': focal,
            'dice': dice,
            'boundary': boundary
        }


class UniversalTrainer:
    """Universal trainer that adapts to any hardware configuration."""
    
    def __init__(
        self,
        model: nn.Module,
        hardware_config: Dict,
        checkpoint_dir: str = 'outputs',
        use_wandb: bool = False
    ):
        self.device = torch.device(hardware_config['device'])
        self.model = model.to(self.device)
        self.config = hardware_config
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.best_metrics = {}
        
        # Mixed precision scaler
        self.scaler = GradScaler() if hardware_config['mixed_precision'] else None
        
        print(f"🚁 Universal Trainer initialized:")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {self.config['batch_size']}")
        print(f"   Workers: {self.config['num_workers']}")
        ## OPTIMIZATION ##: Display new config options
        print(f"   Persistent Workers: {self.config.get('persistent_workers', False)}")
        print(f"   Prefetch Factor: {self.config.get('prefetch_factor', 'N/A')}")
        print(f"   Mixed precision: {self.config['mixed_precision']}")
        print(f"   W&B: {self.use_wandb}")
    
    def train_stage(
        self,
        stage: int,
        train_dataset,
        val_dataset,
        num_epochs: int,
        learning_rate: float = 1e-3,
        stage_name: str = "Training"
    ) -> Dict[str, float]:
        """Generic training stage that adapts to any dataset."""
        
        print(f"\n🎯 {stage_name} (Stage {stage})")
        print(f"   Epochs: {num_epochs}, LR: {learning_rate}")
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Val samples: {len(val_dataset)}")
        
        ## OPTIMIZATION ##: Common DataLoader arguments with optimizations
        loader_kwargs = {
            'num_workers': self.config['num_workers'],
            'pin_memory': self.config['pin_memory']
        }
        
        # Add persistent_workers and prefetch_factor if available
        if self.config.get('persistent_workers', False) and self.config['num_workers'] > 0:
            loader_kwargs['persistent_workers'] = True
            loader_kwargs['prefetch_factor'] = self.config.get('prefetch_factor', 2)
        
        # Create optimized data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            drop_last=True,
            **loader_kwargs
        )
        
        # Validation loader (no need for persistent workers or prefetching for val)
        val_loader_kwargs = {
            'num_workers': self.config['num_workers'],
            'pin_memory': self.config['pin_memory']
        }
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            **val_loader_kwargs
        )
        
        # Get class weights if available
        class_weights = None
        if hasattr(train_dataset, 'get_class_weights'):
            try:
                class_weights = train_dataset.get_class_weights().to(self.device)
            except:
                pass
        elif hasattr(train_dataset, 'get_sample_weights'):
            try:
                class_weights = train_dataset.get_sample_weights().to(self.device)
            except:
                pass
        
        # Setup training components
        loss_fn = MultiDatasetLoss(num_classes=6)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # 🔧 FIXED: Better learning rate scheduling for stability
        if num_epochs <= 30:
            # For short training, use plateau scheduler instead of aggressive cosine
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
            )
        else:
            # For longer training, use gentle cosine annealing
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate*0.1)
        
        # 🔧 FIXED: Add gradient clipping for stability
        max_grad_norm = 1.0
        
        # 🔧 FIXED: Add early stopping to prevent overfitting
        early_stopping_patience = max(5, num_epochs // 4)  # Dynamic patience
        epochs_without_improvement = 0
        
        # Initialize W&B for this stage
        if self.use_wandb:
            wandb.init(
                project="uav-landing-universal",
                name=f"stage{stage}_{stage_name.lower().replace(' ', '_')}",
                config={
                    'stage': stage,
                    'stage_name': stage_name,
                    'num_epochs': num_epochs,
                    'batch_size': self.config['batch_size'],
                    'learning_rate': learning_rate,
                    'hardware': self.config
                },
                reinit=True
            )
        
        # Training loop
        best_miou = 0.0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"{stage_name} - Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_metrics = self._train_epoch(
                train_loader, loss_fn, optimizer, class_weights, max_grad_norm
            )
            
            # Validate
            val_metrics = self._validate_epoch(
                val_loader, loss_fn, class_weights
            )
            
            # Update scheduler
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['miou'])  # Use validation mIoU for plateau scheduler
            else:
                scheduler.step()  # Standard step for cosine annealing
            
            # Save checkpoint
            is_best = val_metrics['miou'] > best_miou
            if is_best:
                best_miou = val_metrics['miou']
                epochs_without_improvement = 0  # Reset counter
                self._save_checkpoint(epoch, stage, val_metrics, f'stage{stage}_best.pth')
            else:
                epochs_without_improvement += 1
            
            # 🔧 FIXED: Early stopping check
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered after {epochs_without_improvement} epochs without improvement")
                print(f"   Best mIoU achieved: {best_miou:.4f}")
                break
            
            # Log to W&B
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'stage': stage,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'val_miou': val_metrics['miou'],
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'best_miou': best_miou
                })
            
            # Print summary
            print(f"📊 Epoch {epoch} Summary:")
            print(f"   Train Loss: {train_metrics['loss']:.4f}")
            print(f"   Val Loss: {val_metrics['loss']:.4f}")
            print(f"   Val mIoU: {val_metrics['miou']:.4f}")
            print(f"   Best mIoU: {best_miou:.4f}")
            print(f"   LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # 🔧 FIXED: Stability warnings
            if val_metrics['loss'] > 2.0:
                print("⚠️  WARNING: High validation loss detected - possible training instability")
            if train_metrics['loss'] < 0.01 and val_metrics['loss'] > 1.0:
                print("⚠️  WARNING: Possible overfitting detected (train loss very low, val loss high)")
        
        stage_metrics = {
            'final_miou': best_miou,
            'final_loss': val_metrics['loss'],
            'epochs': num_epochs
        }
        
        self.best_metrics[f'stage{stage}'] = stage_metrics
        
        print(f"\n✅ {stage_name} Complete!")
        print(f"   Best mIoU: {best_miou:.4f}")
        
        return stage_metrics
    
    def _train_epoch(self, dataloader, loss_fn, optimizer, class_weights, max_grad_norm):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc='Training')
        
        for batch in pbar:
            images = batch['image'].to(self.device, non_blocking=True)
            targets = batch['mask'].to(self.device, non_blocking=True).long()
            
            optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            with autocast(enabled=self.config['mixed_precision']):
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    predictions = outputs['main']
                else:
                    predictions = outputs
                
                loss_dict = loss_fn(predictions, targets, 'dronedeploy', class_weights)
                loss = loss_dict['total']
            
            # Backward pass with optional mixed precision
            if self.config['mixed_precision'] and self.scaler:
                self.scaler.scale(loss).backward()
                # 🔧 FIXED: Add gradient clipping
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                # 🔧 FIXED: Add gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return {'loss': total_loss / num_batches}
    
    def _validate_epoch(self, dataloader, loss_fn, class_weights):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        
        intersection = torch.zeros(6, device=self.device)
        union = torch.zeros(6, device=self.device)
        total_correct = 0
        total_pixels = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc='Validation')
            
            for batch in pbar:
                images = batch['image'].to(self.device, non_blocking=True)
                targets = batch['mask'].to(self.device, non_blocking=True).long()
                
                with autocast(enabled=self.config['mixed_precision']):
                    outputs = self.model(images)
                    if isinstance(outputs, dict):
                        predictions = outputs['main']
                    else:
                        predictions = outputs
                    
                    loss_dict = loss_fn(predictions, targets, 'dronedeploy', class_weights)
                    total_loss += loss_dict['total'].item()
                
                # Compute metrics
                pred_classes = predictions.argmax(dim=1)
                correct = (pred_classes == targets).sum()
                total_correct += correct.item()
                total_pixels += targets.numel()
                
                # Per-class IoU
                for class_id in range(6):
                    pred_mask = (pred_classes == class_id)
                    target_mask = (targets == class_id)
                    
                    intersection[class_id] += (pred_mask & target_mask).sum().float()
                    union[class_id] += (pred_mask | target_mask).sum().float()
        
        # Compute final metrics
        avg_loss = total_loss / len(dataloader)
        overall_accuracy = total_correct / total_pixels
        iou_per_class = intersection / (union + 1e-8)
        miou = iou_per_class.mean().item()
        
        return {
            'loss': avg_loss,
            'miou': miou,
            'accuracy': overall_accuracy,
            'iou_per_class': iou_per_class.cpu().numpy()
        }
    
    def _save_checkpoint(self, epoch, stage, metrics, filename):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'stage': stage,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics,
            'best_metrics': self.best_metrics,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")


## OPTIMIZATION ##: A new function to profile the dataloader performance
def profile_dataloader(dataloader: DataLoader, stage_name: str, num_steps: int = 100):
    """
    Measures the speed of a DataLoader to diagnose bottlenecks.
    """
    print(f"\n⏱️  Profiling DataLoader for '{stage_name}'...")
    print(f"   Will fetch {num_steps} batches.")
    
    start_time = time.time()
    
    # Iterate through the dataloader to measure fetch time
    for i, batch in enumerate(tqdm(dataloader, total=num_steps, desc="Profiling")):
        if i >= num_steps - 1:
            break
    
    end_time = time.time()
    
    total_time = end_time - start_time
    steps_per_sec = num_steps / total_time
    
    print("\n--- DataLoader Profile Report ---")
    print(f"  Stage: {stage_name}")
    print(f"  Total time for {num_steps} batches: {total_time:.2f} seconds")
    print(f"  Speed: {steps_per_sec:.2f} batches/second")
    print("---------------------------------")
    print("💡 Tip: To improve speed, try adjusting --num_workers.")
    print("   - If speed increases with more workers, you are CPU/IO bound.")
    print("   - If speed plateaus or decreases, you have reached the optimal number of workers.")
    print("   - Run this profiler with different --num_workers values to find the sweet spot.")
    
    # Exit after profiling so we don't proceed to train
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='UAV Landing System - Universal Training')
    
    # Dataset paths
    parser.add_argument('--sdd_data_root', type=str, default='./datasets/semantic_drone_dataset',
                        help='Path to Semantic Drone Dataset')
    parser.add_argument('--dronedeploy_data_root', type=str, default='./datasets/drone_deploy_dataset',
                        help='Path to DroneDeploy dataset')
    parser.add_argument('--udd6_data_root', type=str, default='./datasets/udd6_dataset',
                        help='Path to UDD6 dataset')
    
    # Training configuration
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2, 3],
                        help='Training stage (1=SDD, 2=DroneDeploy, 3=UDD6)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train')
    parser.add_argument('--checkpoint_dir', type=str, default='outputs',
                        help='Checkpoint directory')
    parser.add_argument('--model_type', type=str, default='enhanced', choices=['standard', 'enhanced'],
                        help='Model variant')
    
    # Hardware overrides (optional)
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override auto-detected batch size')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Override auto-detected number of workers')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device (cpu/cuda)')
    
    # Optional features
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    ## OPTIMIZATION ##: Add a profiling flag
    parser.add_argument('--profile-dataloader', action='store_true',
                        help='Run a quick benchmark on the DataLoader and exit.')
    
    args = parser.parse_args()
    
    print("🚁 UAV Landing System - Universal Training")
    print("==========================================")
    
    # Detect hardware and get optimal configuration
    detector = HardwareDetector()
    detector.print_system_info()
    
    config = detector.get_optimal_config()
    
    # Override with user parameters if provided
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.num_workers is not None:  # Allow setting num_workers to 0
        config['num_workers'] = args.num_workers
        # If user manually sets workers, respect that for persistent_workers
        config['persistent_workers'] = config['device'] == 'cuda' and config['num_workers'] > 0
    if args.device:
        config['device'] = args.device
    
    print(f"\n⚙️  Optimized Training Configuration:")
    print(f"   Device: {config['device']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Workers: {config['num_workers']}")
    print(f"   Persistent Workers: {config.get('persistent_workers', False)}")
    print(f"   Prefetch Factor: {config.get('prefetch_factor', 2)}")
    print(f"   Mixed precision: {config['mixed_precision']}")
    
    # Create model
    model = create_edge_model(
        model_type=args.model_type,
        num_classes=6,
        use_uncertainty=True,
        pretrained=True
    )
    
    ## OPTIMIZATION ##: Logic to handle the new profiling feature
    if args.profile_dataloader:
        # Create a dummy dataset and loader to profile
        if args.stage == 1:
            dataset = SemanticDroneDataset(
                data_root=args.sdd_data_root,
                split="train",
                transform=create_semantic_drone_transforms(
                    input_size=(512, 512),
                    is_training=True
                ),
                class_mapping="advanced_6_class"
            )
            stage_name = "Stage 1: Semantic Foundation"
        elif args.stage == 2:
            datasets = create_dronedeploy_datasets(
                data_root=args.dronedeploy_data_root,
                patch_size=512,
                augmentation=True
            )
            dataset = datasets['train']
            stage_name = "Stage 2: Landing Specialization"
        else:  # stage 3
            dataset = UDD6Dataset(
                data_root=args.udd6_data_root,
                split="train",
                transform=create_udd6_transforms(is_training=True)
            )
            stage_name = "Stage 3: Domain Adaptation"
        
        loader_kwargs = {
            'num_workers': config['num_workers'],
            'pin_memory': config['pin_memory']
        }
        if config.get('persistent_workers', False) and config['num_workers'] > 0:
            loader_kwargs['persistent_workers'] = True
            loader_kwargs['prefetch_factor'] = config.get('prefetch_factor', 2)
        
        loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            **loader_kwargs
        )
        profile_dataloader(loader, stage_name)
        # The program will exit inside the profile function
    
    # Create trainer
    trainer = UniversalTrainer(
        model=model,
        hardware_config=config,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.use_wandb
    )
    
    try:
        # Run the specified stage
        if args.stage == 1:
            print(f"\n🚀 Running Stage 1: Semantic Foundation Training")
            
            transforms = create_semantic_drone_transforms(
                input_size=(512, 512),
                is_training=True
            )
            
            train_dataset = SemanticDroneDataset(
                data_root=args.sdd_data_root,
                split="train",
                transform=transforms,
                class_mapping="advanced_6_class"
            )
            
            val_dataset = SemanticDroneDataset(
                data_root=args.sdd_data_root,
                split="val",
                transform=create_semantic_drone_transforms(
                    input_size=(512, 512),
                    is_training=False
                ),
                class_mapping="advanced_6_class"
            )
            
            # 🔧 FIXED: Better learning rate for stage 1
            stage1_lr = 5e-4 if args.epochs <= 30 else 1e-3
            
            results = trainer.train_stage(
                stage=1,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                num_epochs=args.epochs,
                learning_rate=stage1_lr,
                stage_name="Semantic Foundation"
            )
            
        elif args.stage == 2:
            print(f"\n🚀 Running Stage 2: Landing Specialization")
            
            datasets = create_dronedeploy_datasets(
                data_root=args.dronedeploy_data_root,
                patch_size=512,
                augmentation=True
            )
            
            # 🔧 FIXED: Better learning rate for stage 2
            stage2_lr = 1e-4 if args.epochs <= 30 else 5e-4
            
            results = trainer.train_stage(
                stage=2,
                train_dataset=datasets['train'],
                val_dataset=datasets['val'],
                num_epochs=args.epochs,
                learning_rate=stage2_lr,
                stage_name="Landing Specialization"
            )
            
        elif args.stage == 3:
            print(f"\n🚀 Running Stage 3: Domain Adaptation")
            
            transforms = create_udd6_transforms(is_training=True)
            
            train_dataset = UDD6Dataset(
                data_root=args.udd6_data_root,
                split="train",
                transform=transforms
            )
            
            val_dataset = UDD6Dataset(
                data_root=args.udd6_data_root,
                split="val",
                transform=create_udd6_transforms(is_training=False)
            )
            
            # 🔧 FIXED: Better learning rate for stage 3 (fine-tuning)
            stage3_lr = 2e-5 if args.epochs <= 30 else 5e-5
            
            results = trainer.train_stage(
                stage=3,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                num_epochs=args.epochs,
                learning_rate=stage3_lr,
                stage_name="Domain Adaptation"
            )
        
        print(f"\n🎉 Training completed successfully!")
        print(f"   Final mIoU: {results['final_miou']:.4f}")
        print(f"   Best model: {args.checkpoint_dir}/stage{args.stage}_best.pth")
        
        # 🔧 FIXED: Add training recommendations
        if results['final_miou'] < 0.3:
            print(f"\n💡 Training Tips for Better Performance:")
            print(f"   • Try longer training: --epochs 50")
            print(f"   • Use data augmentation if not enabled")
            print(f"   • Check dataset quality and balance")
            print(f"   • Consider transfer learning from a pretrained stage")
        
        if args.stage == 1:
            print(f"\n🚀 Next Step: Run Stage 2 Landing Specialization")
            print(f"   python train.py --stage 2 --epochs 30")
        elif args.stage == 2:
            print(f"\n🚀 Next Step: Run Stage 3 Domain Adaptation") 
            print(f"   python train.py --stage 3 --epochs 20")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 