#!/usr/bin/env python3
"""
Progressive UAV Landing Model Training
=====================================

Implements conservative multi-dataset strategy:
1. Start with DroneDeploy (native 6 classes, no mapping conflicts)
2. Optional: Add SDD with careful mapping
3. Optional: Add UDD6 for domain adaptation

Features:
- Stage-wise training with checkpointing
- Multi-dataset loss functions with consistency
- Automatic performance monitoring
- Fallback to single-dataset if conflicts detected
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
import json
import time
import warnings
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import wandb
import cv2
from collections import defaultdict

# Import our components
import sys
sys.path.append('..')
from models.edge_landing_net import EdgeLandingNet, create_edge_model
from datasets.dronedeploy_1024_dataset import DroneDeploy1024Dataset
from datasets.edge_landing_dataset import EdgeLandingDataset

warnings.filterwarnings('ignore')


class MultiDatasetLoss(nn.Module):
    """
    Adaptive loss function for multi-dataset training.
    Handles different dataset characteristics and prevents learning conflicts.
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
        
        # Dataset-specific weights (can be tuned)
        self.dataset_weights = {
            'dronedeploy': {'focal': 0.5, 'dice': 0.3, 'boundary': 0.15, 'consistency': 0.05},
            'semantic_drone': {'focal': 0.4, 'dice': 0.3, 'boundary': 0.2, 'consistency': 0.1},
            'udd6': {'focal': 0.45, 'dice': 0.25, 'boundary': 0.2, 'consistency': 0.1}
        }
        
        # Semantic similarity groups for consistency
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
        # Simple edge loss using Sobel operator
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
        if dataset_source in ['dronedeploy']:  # Single dataset, no consistency needed
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


class ProgressiveLandingTrainer:
    """
    Progressive trainer for multi-dataset UAV landing models.
    Implements conservative strategy with automatic conflict detection.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        use_wandb: bool = True,
        checkpoint_dir: str = 'outputs/progressive_training'
    ):
        self.model = model.to(device)
        self.device = device
        self.use_wandb = use_wandb
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.stage = 1
        self.best_metrics = {}
        self.training_history = defaultdict(list)
        
        # Performance monitoring
        self.performance_threshold = 0.05  # 5% degradation threshold
        self.conflict_detected = False
        
        print(f"🚁 ProgressiveLandingTrainer initialized:")
        print(f"   Device: {device}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Checkpoint dir: {checkpoint_dir}")
    
    def train_stage1_dronedeploy(
        self,
        data_root: str,
        num_epochs: int = 50,
        batch_size: int = 8,
        lr: float = 1e-3
    ) -> Dict[str, float]:
        """
        Stage 1: DroneDeploy training (conservative baseline).
        Native 6 classes, no mapping conflicts.
        """
        
        print(f"\n🎯 Stage 1: DroneDeploy Training")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {lr}")
        
        # Create DroneDeploy datasets
        try:
            from datasets.dronedeploy_1024_dataset import create_dronedeploy_datasets
            
            datasets = create_dronedeploy_datasets(
                data_root=data_root,
                patch_size=1024,
                edge_enhancement=True
            )
            
            if datasets['train'] is None or len(datasets['train']) == 0:
                raise ValueError("No DroneDeploy training data found")
            
            print(f"   Train patches: {len(datasets['train'])}")
            print(f"   Val patches: {len(datasets['val'])}")
            
        except Exception as e:
            print(f"❌ Failed to load DroneDeploy dataset: {e}")
            print(f"🔄 Falling back to EdgeLandingDataset...")
            
            # Fallback to existing dataset
            from datasets.edge_landing_dataset import create_edge_datasets
            datasets = create_edge_datasets(
                data_root=data_root,
                input_size=256,  # Smaller for faster training
                extreme_augmentation=True
            )
        
        # Create data loaders
        train_loader = DataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Get class weights
        class_weights = datasets['train'].get_class_weights().to(self.device)
        print(f"   Class weights: {class_weights}")
        
        # Setup training components
        loss_fn = MultiDatasetLoss(num_classes=6)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project="uav-progressive-landing",
                name=f"stage1_dronedeploy",
                config={
                    'stage': 1,
                    'dataset': 'dronedeploy',
                    'num_epochs': num_epochs,
                    'batch_size': batch_size,
                    'learning_rate': lr,
                    'model': 'EdgeLandingNet'
                }
            )
            wandb.watch(self.model, log_freq=100)
        
        # Training loop
        best_miou = 0.0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Stage 1 - Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
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
            
            # Log metrics
            self.training_history['stage1_train_loss'].append(train_metrics['loss'])
            self.training_history['stage1_val_loss'].append(val_metrics['loss'])
            self.training_history['stage1_val_miou'].append(val_metrics['miou'])
            
            # Save checkpoint
            is_best = val_metrics['miou'] > best_miou
            if is_best:
                best_miou = val_metrics['miou']
                self._save_checkpoint(epoch, val_metrics, f'stage1_best.pth')
            
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
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"   Train Loss: {train_metrics['loss']:.4f}")
            print(f"   Val Loss: {val_metrics['loss']:.4f}")
            print(f"   Val mIoU: {val_metrics['miou']:.4f}")
            print(f"   Best mIoU: {best_miou:.4f}")
        
        # Store stage 1 results
        stage1_metrics = {
            'final_miou': best_miou,
            'final_loss': val_metrics['loss'],
            'epochs': num_epochs
        }
        
        self.best_metrics['stage1'] = stage1_metrics
        
        print(f"\n🎯 Stage 1 Complete!")
        print(f"   Best mIoU: {best_miou:.4f}")
        
        return stage1_metrics
    
    def _train_epoch(self, loader, loss_fn, optimizer, dataset_source, class_weights):
        """Train for one epoch."""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(loader, desc=f'Training')
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            targets = batch['mask'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            if isinstance(outputs, dict):
                predictions = outputs['main']
            else:
                predictions = outputs
            
            # Compute loss
            loss_dict = loss_fn(predictions, targets, dataset_source, class_weights)
            loss = loss_dict['total']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
        
        return {'loss': total_loss / num_batches}
    
    def _validate_epoch(self, loader, loss_fn, dataset_source, class_weights):
        """Validate for one epoch."""
        
        self.model.eval()
        total_loss = 0.0
        
        # Simple IoU computation
        intersection = torch.zeros(6, device=self.device)
        union = torch.zeros(6, device=self.device)
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=f'Validation')
            
            for batch in pbar:
                images = batch['image'].to(self.device)
                targets = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    predictions = outputs['main']
                else:
                    predictions = outputs
                
                # Compute loss
                loss_dict = loss_fn(predictions, targets, dataset_source, class_weights)
                total_loss += loss_dict['total'].item()
                
                # Compute IoU
                pred_classes = predictions.argmax(dim=1)
                
                for class_id in range(6):
                    pred_mask = (pred_classes == class_id)
                    target_mask = (targets == class_id)
                    
                    intersection[class_id] += (pred_mask & target_mask).sum().float()
                    union[class_id] += (pred_mask | target_mask).sum().float()
        
        # Compute mIoU
        iou_per_class = intersection / (union + 1e-8)
        miou = iou_per_class.mean().item()
        
        return {
            'loss': total_loss / len(loader),
            'miou': miou,
            'iou_per_class': iou_per_class.cpu().numpy()
        }
    
    def _save_checkpoint(self, epoch, metrics, filename):
        """Save training checkpoint."""
        
        checkpoint = {
            'epoch': epoch,
            'stage': self.stage,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics,
            'training_history': dict(self.training_history),
            'best_metrics': self.best_metrics
        }
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def evaluate_stage_performance(self, current_metrics, previous_best=None):
        """Evaluate if stage training was successful."""
        
        if previous_best is None:
            return True
        
        # Check for significant degradation
        degradation = previous_best - current_metrics['miou']
        
        if degradation > self.performance_threshold:
            print(f"⚠️  Performance degradation detected: {degradation:.4f}")
            self.conflict_detected = True
            return False
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Progressive UAV Landing Training')
    parser.add_argument('--data_root', default='../datasets', help='Root directory for datasets')
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2, 3], help='Training stage to run')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--model_type', default='standard', choices=['standard', 'fast'], help='Model variant')
    parser.add_argument('--device', default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--checkpoint_dir', default='outputs/progressive_training', help='Checkpoint directory')
    parser.add_argument('--resume', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🚁 Progressive UAV Landing Training")
    print(f"   Stage: {args.stage}")
    print(f"   Device: {device}")
    print(f"   Data root: {args.data_root}")
    print(f"   Model type: {args.model_type}")
    
    # Create model
    model = create_edge_model(
        model_type=args.model_type,
        num_classes=6,
        input_size=256
    )
    
    # Create trainer
    trainer = ProgressiveLandingTrainer(
        model=model,
        device=device,
        use_wandb=args.use_wandb,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.training_history = checkpoint['training_history']
        trainer.best_metrics = checkpoint['best_metrics']
        print(f"📥 Resumed from: {args.resume}")
    
    try:
        if args.stage == 1:
            # Stage 1: DroneDeploy training
            metrics = trainer.train_stage1_dronedeploy(
                data_root=args.data_root,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                lr=args.lr
            )
            
            print(f"\n✅ Stage 1 completed successfully!")
            print(f"   Final mIoU: {metrics['final_miou']:.4f}")
            
            # TODO: Implement stages 2 and 3
            print(f"\n🚀 Ready for Stage 2 (SDD integration)")
            print(f"   Command: python train_progressive_landing.py --stage 2 --resume {args.checkpoint_dir}/stage1_best.pth")
            
        else:
            print(f"⚠️  Stages 2 and 3 not yet implemented")
            print(f"   Current implementation focuses on Stage 1 (DroneDeploy)")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 