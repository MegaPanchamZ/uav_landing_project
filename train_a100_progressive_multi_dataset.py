#!/usr/bin/env python3
"""
A100 Progressive Multi-Dataset UAV Landing Training
==================================================

Complete implementation of the 4-stage progressive training strategy:
- Stage 1: Semantic Foundation (SDD - Semantic Drone Dataset)
- Stage 2: Landing Specialization (DroneDeploy)
- Stage 3: Domain Adaptation (UDD6)
- Stage 4: Joint Refinement (All datasets)

Optimized for A100 GPU with large batch sizes and advanced training techniques.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
import json
import time
import warnings
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import wandb
from collections import defaultdict, Counter
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import our components
from models.mobilenetv3_edge_model import create_edge_model
from datasets.semantic_drone_dataset import SemanticDroneDataset, create_semantic_drone_transforms
from datasets.dronedeploy_1024_dataset import DroneDeploy1024Dataset, create_dronedeploy_datasets
from datasets.udd6_dataset import UDD6Dataset, create_udd6_transforms
from losses.safety_aware_losses import CombinedSafetyLoss

warnings.filterwarnings('ignore')


class MultiDatasetLoss(nn.Module):
    """
    Adaptive loss function for multi-dataset progressive training.
    Implements the strategy from LATEST_MULTI_DATASET_STRATEGY.md
    """
    
    def __init__(self, num_classes=6, alpha=0.25, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        
        # Component losses
        self.focal_loss = self._focal_loss
        self.dice_loss = self._dice_loss
        self.boundary_loss = self._boundary_loss
        self.consistency_loss = self._consistency_loss
        
        # Dataset-specific weights from strategy document
        self.dataset_weights = {
            'semantic_drone': {'focal': 0.4, 'dice': 0.3, 'boundary': 0.2, 'consistency': 0.1},
            'dronedeploy': {'focal': 0.5, 'dice': 0.3, 'boundary': 0.15, 'consistency': 0.05},
            'udd6': {'focal': 0.45, 'dice': 0.25, 'boundary': 0.2, 'consistency': 0.1}
        }
        
        # Semantic similarity groups for consistency loss
        self.similarity_groups = {
            'safe_surfaces': [0, 1],    # ground, vegetation
            'obstacles': [2, 4],        # obstacle, vehicle
            'hazards': [3],             # water
            'uncertain': [5]            # other
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
    
    def _consistency_loss(self, pred, target, dataset_source):
        """Consistency loss to reduce cross-dataset conflicts."""
        if dataset_source == 'dronedeploy':  # Native classes, no consistency needed
            return torch.tensor(0.0, device=pred.device)
        
        pred_probs = F.softmax(pred, dim=1)
        consistency_loss = 0.0
        
        # Encourage consistent predictions for semantically similar classes
        for group_name, class_ids in self.similarity_groups.items():
            if len(class_ids) > 1:
                group_probs = pred_probs[:, class_ids]
                
                # Encourage smooth transitions between similar classes
                for i in range(len(class_ids) - 1):
                    for j in range(i + 1, len(class_ids)):
                        class_i_mask = (target == class_ids[i])
                        class_j_mask = (target == class_ids[j])
                        
                        if class_i_mask.any() and class_j_mask.any():
                            prob_diff = torch.abs(
                                group_probs[:, i][class_i_mask].mean() - 
                                group_probs[:, j][class_j_mask].mean()
                            )
                            consistency_loss += prob_diff
        
        return consistency_loss
    
    def forward(self, pred, target, dataset_source='dronedeploy', class_weights=None):
        """
        Compute adaptive loss based on dataset source.
        
        Args:
            pred: Model predictions [B, num_classes, H, W]
            target: Ground truth labels [B, H, W]
            dataset_source: Source dataset identifier
            class_weights: Class weights for focal loss
        """
        
        # Get dataset-specific weights
        weights = self.dataset_weights.get(dataset_source, self.dataset_weights['dronedeploy'])
        
        # Compute component losses
        focal = self.focal_loss(pred, target, class_weights)
        dice = self.dice_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        consistency = self.consistency_loss(pred, target, dataset_source)
        
        # Weighted combination
        total_loss = (
            weights['focal'] * focal +
            weights['dice'] * dice +
            weights['boundary'] * boundary +
            weights['consistency'] * consistency
        )
        
        return {
            'total': total_loss,
            'focal': focal,
            'dice': dice,
            'boundary': boundary,
            'consistency': consistency
        }


class A100ProgressiveTrainer:
    """
    Progressive trainer optimized for A100 with comprehensive monitoring.
    
    Features:
    - Detailed W&B tracking with metrics visualization
    - Regular model checkpointing and cross-dataset saving
    - Validation mIoU monitoring with best model tracking
    - Training progress visualization
    - Memory and GPU utilization tracking
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        use_wandb: bool = True,
        checkpoint_dir: str = 'outputs/a100_progressive',
        use_amp: bool = True
    ):
        # CPU BOTTLENECK SOLUTION 2: Apply system-level optimizations
        self._optimize_cpu_settings()
        
        self.model = model.to(device)
        self.device = device
        self.use_wandb = use_wandb
        self.use_amp = use_amp
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state tracking
        self.current_stage = 0
        self.global_step = 0
        self.best_metrics = {
            'stage1': {'miou': 0.0, 'epoch': 0},
            'stage2': {'miou': 0.0, 'epoch': 0},
            'stage3': {'miou': 0.0, 'epoch': 0}
        }
        
        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None
        
        # Initialize comprehensive W&B tracking
        if self.use_wandb:
            self._setup_wandb_tracking()
        
        print(f"🚁 A100ProgressiveTrainer initialized:")
        print(f"   Device: {device}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Mixed precision: {use_amp}")
        print(f"   Checkpoint dir: {checkpoint_dir}")
        print(f"   W&B tracking: {use_wandb}")
    
    def _setup_wandb_tracking(self):
        """Setup comprehensive W&B tracking with custom metrics."""
        # Define custom metrics for W&B
        wandb.define_metric("epoch")
        wandb.define_metric("global_step")
        wandb.define_metric("stage")
        
        # Training metrics
        wandb.define_metric("train/loss", step_metric="global_step")
        wandb.define_metric("train/focal_loss", step_metric="global_step")
        wandb.define_metric("train/dice_loss", step_metric="global_step")
        wandb.define_metric("train/boundary_loss", step_metric="global_step")
        wandb.define_metric("train/consistency_loss", step_metric="global_step")
        wandb.define_metric("train/learning_rate", step_metric="global_step")
        wandb.define_metric("train/batch_time", step_metric="global_step")
        
        # Validation metrics
        wandb.define_metric("val/loss", step_metric="epoch")
        wandb.define_metric("val/miou", step_metric="epoch")
        wandb.define_metric("val/accuracy", step_metric="epoch")
        wandb.define_metric("val/class_miou_*", step_metric="epoch")
        
        # Hardware metrics
        wandb.define_metric("hardware/gpu_memory_used", step_metric="global_step")
        wandb.define_metric("hardware/gpu_utilization", step_metric="global_step")
        wandb.define_metric("hardware/temperature", step_metric="global_step")
        
        # Stage-specific metrics
        wandb.define_metric("stage*/best_miou", step_metric="epoch")
        wandb.define_metric("stage*/epochs_completed", step_metric="epoch")
        
        print("   📊 W&B tracking configured with custom metrics")
    
    def _save_checkpoint(
        self, 
        epoch: int, 
        stage: int,
        metrics: Dict[str, float], 
        filename: str = None,
        is_best: bool = False,
        save_optimizer: bool = True
    ):
        """Enhanced checkpoint saving with comprehensive metadata."""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stage{stage}_epoch{epoch}_{timestamp}.pth"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Comprehensive checkpoint data
        checkpoint = {
            # Model and training state
            'epoch': epoch,
            'stage': stage,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            
            # Metrics and performance
            'metrics': metrics,
            'best_metrics': self.best_metrics,
            'is_best': is_best,
            
            # Training configuration
            'model_config': {
                'num_classes': 6,
                'model_type': 'enhanced',
                'parameters': sum(p.numel() for p in self.model.parameters())
            },
            
            # Environment info
            'training_info': {
                'device': str(self.device),
                'amp_enabled': self.use_amp,
                'timestamp': datetime.now().isoformat(),
                'wandb_run_id': wandb.run.id if self.use_wandb and wandb.run else None
            }
        }
        
        # Optionally save optimizer state (large file)
        if save_optimizer and hasattr(self, 'optimizer'):
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Log to W&B
        if self.use_wandb:
            artifact = wandb.Artifact(
                name=f"model_stage{stage}_epoch{epoch}",
                type="model",
                description=f"Stage {stage} model at epoch {epoch}, mIoU: {metrics.get('miou', 0.0):.4f}"
            )
            artifact.add_file(str(checkpoint_path))
            wandb.log_artifact(artifact)
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        if is_best:
            print(f"   🏆 New best model for stage {stage}!")
        
        return checkpoint_path
    
    def _log_hardware_metrics(self):
        """Log GPU and hardware metrics to W&B."""
        if not self.use_wandb:
            return
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU metrics
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            wandb.log({
                "hardware/gpu_memory_used": mem_info.used / 1024**3,  # GB
                "hardware/gpu_memory_total": mem_info.total / 1024**3,  # GB
                "hardware/gpu_utilization": utilization.gpu,
                "hardware/memory_utilization": utilization.memory,
                "hardware/temperature": temperature,
                "global_step": self.global_step
            })
        except Exception as e:
            # Fallback to torch CUDA info
            if torch.cuda.is_available():
                wandb.log({
                    "hardware/gpu_memory_used": torch.cuda.memory_allocated(0) / 1024**3,
                    "hardware/gpu_memory_reserved": torch.cuda.memory_reserved(0) / 1024**3,
                    "global_step": self.global_step
                })
    
    def _compute_detailed_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute detailed validation metrics including per-class IoU."""
        
        pred_classes = predictions.argmax(dim=1)
        
        # Overall accuracy
        correct = (pred_classes == targets).sum().item()
        total = targets.numel()
        accuracy = correct / total
        
        # Per-class IoU
        ious = []
        class_names = ['ground', 'vegetation', 'obstacle', 'water', 'vehicle', 'other']
        class_metrics = {}
        
        for class_id in range(6):
            pred_mask = (pred_classes == class_id)
            target_mask = (targets == class_id)
            
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            
            if union > 0:
                iou = intersection / union
                ious.append(iou.item())
                class_metrics[f'class_miou_{class_names[class_id]}'] = iou.item()
            else:
                ious.append(0.0)
                class_metrics[f'class_miou_{class_names[class_id]}'] = 0.0
        
        # Mean IoU
        miou = sum(ious) / len(ious)
        
        return {
            'accuracy': accuracy,
            'miou': miou,
            **class_metrics
        }

    def train_stage1_semantic_foundation(
        self,
        data_root: str,
        num_epochs: int = 50,
        batch_size: int = 256,  # Massive batch for A100 SXM + 251GB RAM
        lr: float = 1e-3
    ) -> Dict[str, float]:
        """
        Stage 1: Semantic Foundation Training on Semantic Drone Dataset.
        
        Focus: Rich semantic understanding from 24→6 class mapping.
        Strategy: Leverage SDD's high-quality annotations for foundation learning.
        CPU Bottleneck Optimized for A100 SXM with 32 vCPU and 251GB RAM.
        """
        
        print(f"\n🎯 Stage 1: Semantic Foundation (SDD)")
        print(f"   Strategy: Rich semantic understanding from 24 classes")
        print(f"   Epochs: {num_epochs}, Batch size: {batch_size}, LR: {lr}")
        print(f"   Hardware: A100 SXM + 32 vCPU + 251GB RAM optimization")
        print(f"   🔧 CPU Bottleneck Fixes: Reduced workers, optimized transforms, smart caching")
        
        # Create transforms
        transforms = create_semantic_drone_transforms(
            input_size=(512, 512),
            is_training=True,
            advanced_augmentation=False  # Fast augmentation for A100
        )
        
        # SOLUTION 3: MEMORY-BASED DATASET (eliminates CPU bottleneck entirely!)
        print("🚀 Loading datasets to memory (251GB RAM optimization)")
        
        train_dataset = SemanticDroneDataset(
            data_root=data_root,
            split="train",
            transform=transforms,
            class_mapping="advanced_6_class",
            use_random_crops=True,
            crops_per_image=8,  # More crops for massive RAM
            cache_images=True,   # Keep disk cache as backup
            preload_to_memory=True  # 🔥 LOAD ENTIRE DATASET TO RAM!
        )
        
        val_dataset = SemanticDroneDataset(
            data_root=data_root,
            split="val", 
            transform=create_semantic_drone_transforms(
                input_size=(512, 512),
                is_training=False
            ),
            class_mapping="advanced_6_class",
            use_random_crops=False,
            cache_images=True,
            preload_to_memory=True  # 🔥 VALIDATION ALSO IN RAM!
        )
        
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Val samples: {len(val_dataset)}")
        
        # SOLUTION 4: MINIMAL WORKERS (data already in RAM!)
        # With memory preloading, we need VERY few workers
        print("🔥 Optimizing DataLoaders for memory-based data (minimal CPU overhead)")
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True, drop_last=True,  # Only 2 workers needed!
            persistent_workers=False, prefetch_factor=1,  # Minimal prefetching
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=1, pin_memory=True,  # Single worker for validation
            persistent_workers=False, prefetch_factor=1,
        )
        
        # Get class weights
        class_weights = train_dataset.get_sample_weights().to(self.device)
        print(f"   Class weights: {class_weights}")
        
        # Setup training components
        loss_fn = MultiDatasetLoss(num_classes=6)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project="uav-a100-progressive",
                name=f"stage1_semantic_foundation",
                config={
                    'stage': 1,
                    'dataset': 'semantic_drone',
                    'num_epochs': num_epochs,
                    'batch_size': batch_size,
                    'learning_rate': lr,
                    'model': 'MobileNetV3Edge'
                }
            )
            wandb.watch(self.model, log_freq=100)
        
        # Training loop
        best_miou = 0.0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*70}")
            print(f"Stage 1 - Epoch {epoch}/{num_epochs}")
            print(f"{'='*70}")
            
            # Train
            train_metrics = self._train_epoch(
                train_loader, loss_fn, optimizer, 'semantic_drone', class_weights
            )
            
            # Validate
            val_metrics = self._validate_epoch(
                val_loader, loss_fn, 'semantic_drone', class_weights
            )
            
            # Update scheduler
            scheduler.step()
            
            # Save checkpoint
            is_best = val_metrics['miou'] > best_miou
            if is_best:
                best_miou = val_metrics['miou']
                self._save_checkpoint(epoch, 1, val_metrics, f'stage1_best.pth', is_best)
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'stage': 1,
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
        
        stage1_metrics = {
            'final_miou': best_miou,
            'final_loss': val_metrics['loss'],
            'epochs': num_epochs
        }
        
        self.best_metrics['stage1'] = stage1_metrics
        
        print(f"\n✅ Stage 1 Complete!")
        print(f"   Best mIoU: {best_miou:.4f}")
        
        return stage1_metrics
    
    def train_stage2_landing_specialization(
        self,
        data_root: str,
        num_epochs: int = 30,
        batch_size: int = 64,  # Larger batch for A100 SXM + 251GB RAM with 1024px patches
        lr: float = 1e-4
    ) -> Dict[str, float]:
        """
        Stage 2: Landing Specialization (DroneDeploy)
        Focus on landing-specific decisions with native 6 classes.
        """
        
        print(f"\n🎯 Stage 2: Landing Specialization (DroneDeploy)")
        print(f"   Strategy: Native 6-class landing decisions")
        print(f"   Epochs: {num_epochs}, Batch size: {batch_size}, LR: {lr}")
        
        # Load Stage 1 model
        stage1_checkpoint = self.checkpoint_dir / 'stage1_best.pth'
        if stage1_checkpoint.exists():
            checkpoint = torch.load(stage1_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   Loaded Stage 1 model: mIoU={checkpoint['metrics']['miou']:.4f}")
        else:
            print(f"   ⚠️  No Stage 1 checkpoint found, using current model")
        
        # Create DroneDeploy datasets
        try:
            datasets = create_dronedeploy_datasets(
                data_root=data_root,
                patch_size=1024,  # Large patches for quality
                stride_factor=0.5,
                edge_enhancement=True,
                augmentation=True
            )
            
            if datasets['train'] is None or len(datasets['train']) == 0:
                raise ValueError("No DroneDeploy training data found")
            
            print(f"   Train patches: {len(datasets['train'])}")
            print(f"   Val patches: {len(datasets['val'])}")
            
        except Exception as e:
            print(f"❌ Failed to load DroneDeploy dataset: {e}")
            return {'final_miou': 0.0, 'final_loss': float('inf'), 'epochs': 0}
        
        # Create data loaders
        train_loader = DataLoader(
            datasets['train'], batch_size=batch_size, shuffle=True,
            num_workers=8, pin_memory=True, drop_last=True,
            persistent_workers=True
        )
        val_loader = DataLoader(
            datasets['val'], batch_size=batch_size, shuffle=False,
            num_workers=8, pin_memory=True,
            persistent_workers=True
        )
        
        # Get class weights
        class_weights = datasets['train'].get_class_weights().to(self.device)
        print(f"   Class weights: {class_weights}")
        
        # Setup training (lower LR for fine-tuning)
        loss_fn = MultiDatasetLoss(num_classes=6)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Update wandb
        if self.use_wandb:
            wandb.init(
                project="uav-a100-progressive",
                name=f"stage2_landing_specialization",
                config={
                    'stage': 2,
                    'dataset': 'dronedeploy',
                    'num_epochs': num_epochs,
                    'batch_size': batch_size,
                    'learning_rate': lr
                },
                reinit=True
            )
        
        # Training loop
        best_miou = 0.0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*70}")
            print(f"Stage 2 - Epoch {epoch}/{num_epochs}")
            print(f"{'='*70}")
            
            # Train
            train_metrics = self._train_epoch(
                train_loader, loss_fn, optimizer, 'dronedeploy', class_weights
            )
            
            # Validate
            val_metrics = self._validate_epoch(
                val_loader, loss_fn, 'dronedeploy', class_weights
            )
            
            # Update scheduler
            scheduler.step()
            
            # Save checkpoint
            is_best = val_metrics['miou'] > best_miou
            if is_best:
                best_miou = val_metrics['miou']
                self._save_checkpoint(epoch, 2, val_metrics, f'stage2_best.pth', is_best)
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'stage': 2,
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
        
        stage2_metrics = {
            'final_miou': best_miou,
            'final_loss': val_metrics['loss'],
            'epochs': num_epochs
        }
        
        self.best_metrics['stage2'] = stage2_metrics
        
        print(f"\n✅ Stage 2 Complete!")
        print(f"   Best mIoU: {best_miou:.4f}")
        
        return stage2_metrics
    
    def train_stage3_domain_adaptation(
        self,
        data_root: str,
        num_epochs: int = 20,
        batch_size: int = 128,  # Maximum batch for A100 SXM stage 3
        lr: float = 5e-5
    ) -> Dict[str, float]:
        """
        Stage 3: Domain Adaptation (UDD6)
        Focus on altitude/urban robustness with high-altitude perspective.
        """
        
        print(f"\n🎯 Stage 3: Domain Adaptation (UDD6)")
        print(f"   Strategy: High-altitude urban robustness")
        print(f"   Epochs: {num_epochs}, Batch size: {batch_size}, LR: {lr}")
        
        # Load Stage 2 model
        stage2_checkpoint = self.checkpoint_dir / 'stage2_best.pth'
        if stage2_checkpoint.exists():
            checkpoint = torch.load(stage2_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   Loaded Stage 2 model: mIoU={checkpoint['metrics']['miou']:.4f}")
        else:
            print(f"   ⚠️  No Stage 2 checkpoint found, using current model")
        
        # Create UDD6 dataset
        try:
            transforms = create_udd6_transforms(is_training=True)
            
            train_dataset = UDD6Dataset(
                data_root=data_root,
                split="train",
                transform=transforms,
                use_random_crops=True,
                crops_per_image=4
            )
            
            val_dataset = UDD6Dataset(
                data_root=data_root,
                split="val",
                transform=create_udd6_transforms(is_training=False),
                use_random_crops=False
            )
            
            print(f"   Train samples: {len(train_dataset)}")
            print(f"   Val samples: {len(val_dataset)}")
            
        except Exception as e:
            print(f"❌ Failed to load UDD6 dataset: {e}")
            return {'final_miou': 0.0, 'final_loss': float('inf'), 'epochs': 0}
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=8, pin_memory=True, drop_last=True,
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=8, pin_memory=True,
            persistent_workers=True
        )
        
        # Get class weights
        class_weights = train_dataset.get_class_weights().to(self.device)
        print(f"   Class weights: {class_weights}")
        
        # Setup training (very low LR for domain adaptation)
        loss_fn = MultiDatasetLoss(num_classes=6)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Update wandb
        if self.use_wandb:
            wandb.init(
                project="uav-a100-progressive",
                name=f"stage3_domain_adaptation",
                config={
                    'stage': 3,
                    'dataset': 'udd6',
                    'num_epochs': num_epochs,
                    'batch_size': batch_size,
                    'learning_rate': lr
                },
                reinit=True
            )
        
        # Training loop
        best_miou = 0.0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*70}")
            print(f"Stage 3 - Epoch {epoch}/{num_epochs}")
            print(f"{'='*70}")
            
            # Train
            train_metrics = self._train_epoch(
                train_loader, loss_fn, optimizer, 'udd6', class_weights
            )
            
            # Validate
            val_metrics = self._validate_epoch(
                val_loader, loss_fn, 'udd6', class_weights
            )
            
            # Update scheduler
            scheduler.step()
            
            # Save checkpoint
            is_best = val_metrics['miou'] > best_miou
            if is_best:
                best_miou = val_metrics['miou']
                self._save_checkpoint(epoch, 3, val_metrics, f'stage3_best.pth', is_best)
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'stage': 3,
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
        
        stage3_metrics = {
            'final_miou': best_miou,
            'final_loss': val_metrics['loss'],
            'epochs': num_epochs
        }
        
        self.best_metrics['stage3'] = stage3_metrics
        
        print(f"\n✅ Stage 3 Complete!")
        print(f"   Best mIoU: {best_miou:.4f}")
        
        return stage3_metrics
    
    def _train_epoch(self, dataloader, loss_fn, optimizer, dataset_source, class_weights):
        """Train for one epoch with comprehensive W&B logging."""
        
        self.model.train()
        total_loss = 0.0
        loss_components = defaultdict(float)
        num_batches = 0
        batch_times = []
        
        pbar = tqdm(dataloader, desc=f'Training')
        
        for batch_idx, batch in enumerate(pbar):
            batch_start_time = time.time()
            
            images = batch['image'].to(self.device, non_blocking=True)
            targets = batch['mask'].to(self.device, non_blocking=True).long()
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    predictions = outputs['main']
                else:
                    predictions = outputs
                
                # Compute loss
                loss_dict = loss_fn(predictions, targets, dataset_source, class_weights)
                loss = loss_dict['total']
            
            # Mixed precision backward pass
            if self.use_amp and self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # Track batch time
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # Update metrics
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key != 'total':
                    loss_components[key] += value.item() if hasattr(value, 'item') else value
            num_batches += 1
            self.global_step += 1
            
            # Log to W&B every 10 batches
            if self.use_wandb and batch_idx % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                log_dict = {
                    'train/loss': loss.item(),
                    'train/learning_rate': current_lr,
                    'train/batch_time': batch_time,
                    'global_step': self.global_step
                }
                
                # Add component losses
                for key, value in loss_dict.items():
                    if key != 'total':
                        log_dict[f'train/{key}_loss'] = value.item() if hasattr(value, 'item') else value
                
                wandb.log(log_dict)
                
                # Log hardware metrics every 50 batches
                if batch_idx % 50 == 0:
                    self._log_hardware_metrics()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                'batch_time': f'{batch_time:.2f}s'
            })
        
        # Compute epoch averages
        avg_batch_time = sum(batch_times) / len(batch_times)
        epoch_loss = total_loss / num_batches
        
        # Log epoch summary to W&B
        if self.use_wandb:
            epoch_log = {
                'train/epoch_loss': epoch_loss,
                'train/avg_batch_time': avg_batch_time,
                'train/throughput_samples_per_sec': len(dataloader.dataset) / sum(batch_times),
                'global_step': self.global_step
            }
            
            # Add average component losses
            for key, value in loss_components.items():
                epoch_log[f'train/epoch_{key}_loss'] = value / num_batches
            
            wandb.log(epoch_log)
        
        return {
            'loss': epoch_loss,
            'avg_batch_time': avg_batch_time,
            **{key: value / num_batches for key, value in loss_components.items()}
        }
    
    def _validate_epoch(self, dataloader, loss_fn, dataset_source, class_weights):
        """Validate for one epoch with detailed W&B metrics logging."""
        
        self.model.eval()
        total_loss = 0.0
        loss_components = defaultdict(float)
        
        # Detailed metrics computation
        intersection = torch.zeros(6, device=self.device)
        union = torch.zeros(6, device=self.device)
        class_correct = torch.zeros(6, device=self.device)
        class_total = torch.zeros(6, device=self.device)
        total_correct = 0
        total_pixels = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f'Validation')
            
            for batch in pbar:
                images = batch['image'].to(self.device, non_blocking=True)
                targets = batch['mask'].to(self.device, non_blocking=True).long()
                
                # Mixed precision forward pass
                with autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    if isinstance(outputs, dict):
                        predictions = outputs['main']
                    else:
                        predictions = outputs
                    
                    # Compute loss
                    loss_dict = loss_fn(predictions, targets, dataset_source, class_weights)
                    total_loss += loss_dict['total'].item()
                    
                    # Track component losses
                    for key, value in loss_dict.items():
                        if key != 'total':
                            loss_components[key] += value.item() if hasattr(value, 'item') else value
                
                # Compute detailed metrics
                pred_classes = predictions.argmax(dim=1)
                
                # Overall accuracy
                correct = (pred_classes == targets).sum()
                total_correct += correct.item()
                total_pixels += targets.numel()
                
                # Per-class metrics
                for class_id in range(6):
                    pred_mask = (pred_classes == class_id)
                    target_mask = (targets == class_id)
                    
                    # IoU computation
                    intersection[class_id] += (pred_mask & target_mask).sum().float()
                    union[class_id] += (pred_mask | target_mask).sum().float()
                    
                    # Class accuracy
                    if target_mask.sum() > 0:
                        class_correct[class_id] += (pred_classes[target_mask] == class_id).sum().float()
                        class_total[class_id] += target_mask.sum().float()
        
        # Compute final metrics
        avg_loss = total_loss / len(dataloader)
        overall_accuracy = total_correct / total_pixels
        
        # Per-class IoU and accuracy
        iou_per_class = intersection / (union + 1e-8)
        miou = iou_per_class.mean().item()
        
        class_accuracy = class_correct / (class_total + 1e-8)
        mean_class_accuracy = class_accuracy.mean().item()
        
        # Prepare detailed metrics
        class_names = ['ground', 'vegetation', 'obstacle', 'water', 'vehicle', 'other']
        detailed_metrics = {
            'loss': avg_loss,
            'miou': miou,
            'accuracy': overall_accuracy,
            'mean_class_accuracy': mean_class_accuracy,
            'iou_per_class': iou_per_class.cpu().numpy(),
            'class_accuracy': class_accuracy.cpu().numpy()
        }
        
        # Add component losses
        for key, value in loss_components.items():
            detailed_metrics[f'{key}_loss'] = value / len(dataloader)
        
        # Log to W&B
        if self.use_wandb:
            val_log = {
                'val/loss': avg_loss,
                'val/miou': miou,
                'val/accuracy': overall_accuracy,
                'val/mean_class_accuracy': mean_class_accuracy
            }
            
            # Add per-class metrics
            for i, class_name in enumerate(class_names):
                val_log[f'val/class_miou_{class_name}'] = iou_per_class[i].item()
                val_log[f'val/class_accuracy_{class_name}'] = class_accuracy[i].item()
            
            # Add component losses
            for key, value in loss_components.items():
                val_log[f'val/{key}_loss'] = value / len(dataloader)
            
            wandb.log(val_log)
        
        return detailed_metrics


def main():
    parser = argparse.ArgumentParser(description='A100 Progressive Multi-Dataset Training')
    
    # Paths
    parser.add_argument('--sdd_data_root', type=str, required=True,
                        help='Path to Semantic Drone Dataset')
    parser.add_argument('--dronedeploy_data_root', type=str, required=True,
                        help='Path to DroneDeploy dataset')
    parser.add_argument('--udd6_data_root', type=str, required=True,
                        help='Path to UDD6 dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='outputs/a100_progressive',
                        help='Checkpoint directory')
    
    # Training parameters
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2, 3, 4],
                        help='Training stage to run (1=SDD, 2=DroneDeploy, 3=UDD6, 4=Joint)')
    parser.add_argument('--model_type', type=str, default='enhanced', choices=['standard', 'enhanced'],
                        help='Edge model type')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    
    # Stage-specific parameters
    parser.add_argument('--stage1_epochs', type=int, default=50, help='Stage 1 epochs')
    parser.add_argument('--stage2_epochs', type=int, default=30, help='Stage 2 epochs')
    parser.add_argument('--stage3_epochs', type=int, default=20, help='Stage 3 epochs')
    
    args = parser.parse_args()
    
    print(f"🚁 A100 Progressive Multi-Dataset Training")
    print(f"   Stage: {args.stage}")
    print(f"   Device: {args.device}")
    print(f"   Model: MobileNetV3 {args.model_type}")
    
    # Create edge-optimized model
    model = create_edge_model(
        model_type=args.model_type,
        num_classes=6,
        use_uncertainty=True,
        pretrained=True
    )
    
    # Create trainer
    trainer = A100ProgressiveTrainer(
        model=model,
        device=args.device,
        use_wandb=args.use_wandb,
        checkpoint_dir=args.checkpoint_dir
    )
    
    try:
        # Run the specified stage
        if args.stage == 1:
            print(f"\n🚀 Running Stage 1: Semantic Foundation")
            results = trainer.train_stage1_semantic_foundation(
                data_root=args.sdd_data_root,
                num_epochs=args.stage1_epochs,
                batch_size=256,  # 🔥 A100 SXM optimization!
                lr=1e-3
            )
            
        elif args.stage == 2:
            print(f"\n🚀 Running Stage 2: Landing Specialization")
            results = trainer.train_stage2_landing_specialization(
                data_root=args.dronedeploy_data_root,
                num_epochs=args.stage2_epochs
            )
            
        elif args.stage == 3:
            print(f"\n🚀 Running Stage 3: Domain Adaptation")
            results = trainer.train_stage3_domain_adaptation(
                data_root=args.udd6_data_root,
                num_epochs=args.stage3_epochs
            )
            
        else:
            print(f"❌ Stage 4 (Joint Refinement) not yet implemented")
            return
        
        print(f"\n🎉 Stage {args.stage} completed successfully!")
        print(f"   Final mIoU: {results['final_miou']:.4f}")
        print(f"   Best model: {args.checkpoint_dir}/stage{args.stage}_best.pth")
        
        # Suggest next stage
        if args.stage < 3:
            next_stage = args.stage + 1
            print(f"\n🚀 Ready for Stage {next_stage}!")
            if next_stage == 2:
                print(f"   Command: python train_a100_progressive_multi_dataset.py --stage 2 --dronedeploy_data_root {args.dronedeploy_data_root}")
            elif next_stage == 3:
                print(f"   Command: python train_a100_progressive_multi_dataset.py --stage 3 --udd6_data_root {args.udd6_data_root}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 