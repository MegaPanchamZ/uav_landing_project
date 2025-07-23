#!/usr/bin/env python3
"""
Edge UAV Landing Model Training
===============================

Training script optimized for real-time edge deployment.
Focuses on speed, efficiency, and robustness with limited data.

Key Features:
- MobileNetV3-Small backbone for speed
- 6-class landing-focused classification
- Extreme augmentation for limited data
- Multi-component loss for class imbalance
- Real-time inference optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
import json
import time
import warnings
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import wandb
import cv2

# Import our edge components
import sys
sys.path.append('..')
from models.edge_landing_net import EdgeLandingNet, create_edge_model, benchmark_model
from datasets.edge_landing_dataset import create_edge_datasets
from utils.metrics import SegmentationMetrics

warnings.filterwarnings('ignore')


class EdgeLandingTrainer:
    """
    Trainer optimized for edge UAV landing models.
    Focuses on speed, efficiency, and robust training with limited data.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        use_wandb: bool = True,
        checkpoint_dir: str = 'outputs/edge_training'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_wandb = use_wandb
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.best_miou = 0.0
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'train_miou': [], 'val_miou': [],
            'train_speed_ms': [], 'lr': []
        }
        
        # Initialize metrics
        self.metrics = SegmentationMetrics(num_classes=6)
        
        print(f"🚁 EdgeLandingTrainer initialized:")
        print(f"   Device: {device}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Train samples: {len(train_loader.dataset)}")
        print(f"   Val samples: {len(val_loader.dataset)}")
    
    def create_loss_function(self, class_weights: torch.Tensor) -> nn.Module:
        """Create multi-component loss optimized for landing detection."""
        
        class EdgeLandingLoss(nn.Module):
            def __init__(self, class_weights, alpha=0.25, gamma=2.0):
                super().__init__()
                self.class_weights = class_weights.to(class_weights.device)
                self.alpha = alpha
                self.gamma = gamma
                
                # Component losses
                self.focal_loss = self._focal_loss
                self.dice_loss = self._dice_loss
                self.boundary_loss = self._boundary_loss
            
            def _focal_loss(self, pred, target):
                """Focal loss for class imbalance."""
                ce_loss = F.cross_entropy(pred, target, weight=self.class_weights, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
            
            def _dice_loss(self, pred, target):
                """Dice loss for segmentation quality."""
                smooth = 1e-5
                pred_soft = F.softmax(pred, dim=1)
                
                dice_loss = 0
                for class_id in range(pred.size(1)):
                    pred_class = pred_soft[:, class_id]
                    target_class = (target == class_id).float()
                    
                    intersection = (pred_class * target_class).sum()
                    union = pred_class.sum() + target_class.sum()
                    
                    dice_coeff = (2 * intersection + smooth) / (union + smooth)
                    dice_loss += 1 - dice_coeff
                
                return dice_loss / pred.size(1)
            
            def _boundary_loss(self, pred, target):
                """Boundary loss for edge preservation."""
                # Sobel filters for edge detection
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                     dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(0)
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                     dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(0)
                
                # Compute edges for prediction and target
                pred_edges = torch.sqrt(
                    F.conv2d(pred.argmax(1).float().unsqueeze(1), sobel_x, padding=1) ** 2 +
                    F.conv2d(pred.argmax(1).float().unsqueeze(1), sobel_y, padding=1) ** 2
                )
                
                target_edges = torch.sqrt(
                    F.conv2d(target.float().unsqueeze(1), sobel_x, padding=1) ** 2 +
                    F.conv2d(target.float().unsqueeze(1), sobel_y, padding=1) ** 2
                )
                
                # Boundary loss
                return F.mse_loss(pred_edges, target_edges)
            
            def forward(self, pred, target):
                # Multi-component loss with safety emphasis
                focal = self.focal_loss(pred, target)
                dice = self.dice_loss(pred, target)
                boundary = self.boundary_loss(pred, target)
                
                # Weighted combination (emphasize focal for safety)
                total_loss = 0.6 * focal + 0.3 * dice + 0.1 * boundary
                
                return {
                    'total': total_loss,
                    'focal': focal,
                    'dice': dice,
                    'boundary': boundary
                }
        
        return EdgeLandingLoss(class_weights)
    
    def create_optimizer(self, lr: float = 1e-3, weight_decay: float = 1e-4) -> optim.Optimizer:
        """Create optimizer with different learning rates for backbone and head."""
        
        # Separate backbone and head parameters
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name or 'features' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        # Different learning rates
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': lr * 0.1, 'weight_decay': weight_decay},  # Lower LR for pretrained
            {'params': head_params, 'lr': lr, 'weight_decay': weight_decay}  # Higher LR for new layers
        ])
        
        return optimizer
    
    def create_scheduler(self, optimizer: optim.Optimizer, total_epochs: int):
        """Create learning rate scheduler."""
        
        # Cosine annealing with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=total_epochs // 4,  # First restart
            T_mult=2,  # Double restart period
            eta_min=1e-6
        )
        
        return scheduler
    
    def train_epoch(self, epoch: int, loss_fn, optimizer, scheduler) -> Dict[str, float]:
        """Train for one epoch with speed monitoring."""
        
        self.model.train()
        total_loss = 0.0
        component_losses = {'focal': 0.0, 'dice': 0.0, 'boundary': 0.0}
        
        # Speed monitoring
        inference_times = []
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            targets = batch['mask'].to(self.device)
            
            # Forward pass with timing
            start_time = time.perf_counter()
            
            optimizer.zero_grad()
            outputs = self.model(images)
            
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000 / images.size(0))  # ms per image
            
            # Compute loss
            if isinstance(outputs, dict):
                predictions = outputs['main']
            else:
                predictions = outputs
            
            loss_dict = loss_fn(predictions, targets)
            loss = loss_dict['total']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            for key in component_losses:
                component_losses[key] += loss_dict[key].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'speed': f'{np.mean(inference_times[-10:]):.1f}ms'
            })
            
            # Log batch metrics to wandb
            if self.use_wandb and batch_idx % 50 == 0:
                wandb.log({
                    'batch_loss': loss.item(),
                    'batch_speed_ms': np.mean(inference_times[-10:]),
                    'epoch': epoch,
                    'batch': batch_idx
                })
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch metrics
        num_batches = len(self.train_loader)
        metrics = {
            'loss': total_loss / num_batches,
            'focal_loss': component_losses['focal'] / num_batches,
            'dice_loss': component_losses['dice'] / num_batches,
            'boundary_loss': component_losses['boundary'] / num_batches,
            'avg_speed_ms': np.mean(inference_times),
            'lr': optimizer.param_groups[0]['lr']
        }
        
        return metrics
    
    def validate_epoch(self, epoch: int, loss_fn) -> Dict[str, float]:
        """Validate model with comprehensive metrics."""
        
        self.model.eval()
        total_loss = 0.0
        
        # Reset metrics
        self.metrics.reset()
        
        # Class-wise predictions for detailed analysis
        class_predictions = {i: [] for i in range(6)}
        class_targets = {i: [] for i in range(6)}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Val Epoch {epoch}')
            
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
                loss_dict = loss_fn(predictions, targets)
                total_loss += loss_dict['total'].item()
                
                # Update metrics
                pred_classes = predictions.argmax(dim=1)
                self.metrics.update(pred_classes.cpu(), targets.cpu())
                
                # Collect class-wise predictions
                for class_id in range(6):
                    class_mask = (targets == class_id)
                    if class_mask.any():
                        class_predictions[class_id].extend(pred_classes[class_mask].cpu().numpy())
                        class_targets[class_id].extend(targets[class_mask].cpu().numpy())
        
        # Compute final metrics
        miou = self.metrics.mean_iou()
        accuracy = self.metrics.accuracy()
        class_ious = self.metrics.class_iou()
        
        # Safety-critical metrics (focus on hazard/obstacle detection)
        hazard_recall = self.metrics.class_recall(4)  # Hazard class
        obstacle_recall = self.metrics.class_recall(3)  # Obstacle class
        safe_precision = (self.metrics.class_precision(1) + self.metrics.class_precision(2)) / 2  # Safe classes
        
        metrics = {
            'loss': total_loss / len(self.val_loader),
            'miou': miou,
            'accuracy': accuracy,
            'hazard_recall': hazard_recall,
            'obstacle_recall': obstacle_recall,
            'safe_precision': safe_precision,
            'class_ious': class_ious
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save training checkpoint."""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics,
            'training_history': self.training_history,
            'best_miou': self.best_miou
        }
        
        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"🏆 New best model saved! mIoU: {metrics['miou']:.4f}")
        
        # Keep only last 5 checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        if len(checkpoints) > 5:
            for old_checkpoint in checkpoints[:-5]:
                old_checkpoint.unlink()
    
    def train(self, num_epochs: int, lr: float = 1e-3, class_weights: Optional[torch.Tensor] = None):
        """Main training loop."""
        
        print(f"🚀 Starting edge landing model training...")
        print(f"   Epochs: {num_epochs}")
        print(f"   Learning rate: {lr}")
        print(f"   Device: {self.device}")
        
        # Setup training components
        if class_weights is None:
            class_weights = torch.ones(6)
        class_weights = class_weights.to(self.device)
        
        loss_fn = self.create_loss_function(class_weights)
        optimizer = self.create_optimizer(lr)
        scheduler = self.create_scheduler(optimizer, num_epochs)
        
        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project="uav-edge-landing",
                config={
                    'model': 'EdgeLandingNet',
                    'num_epochs': num_epochs,
                    'learning_rate': lr,
                    'batch_size': self.train_loader.batch_size,
                    'num_classes': 6,
                    'input_size': 256
                }
            )
            wandb.watch(self.model, log_freq=100)
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_metrics = self.train_epoch(epoch, loss_fn, optimizer, scheduler)
            
            # Validate
            val_metrics = self.validate_epoch(epoch, loss_fn)
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_miou'].append(0.0)  # Computed in validation
            self.training_history['val_miou'].append(val_metrics['miou'])
            self.training_history['train_speed_ms'].append(train_metrics['avg_speed_ms'])
            self.training_history['lr'].append(train_metrics['lr'])
            
            # Check for best model
            is_best = val_metrics['miou'] > self.best_miou
            if is_best:
                self.best_miou = val_metrics['miou']
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"   Train Loss: {train_metrics['loss']:.4f}")
            print(f"   Val Loss: {val_metrics['loss']:.4f}")
            print(f"   Val mIoU: {val_metrics['miou']:.4f}")
            print(f"   Hazard Recall: {val_metrics['hazard_recall']:.4f}")
            print(f"   Obstacle Recall: {val_metrics['obstacle_recall']:.4f}")
            print(f"   Safe Precision: {val_metrics['safe_precision']:.4f}")
            print(f"   Speed: {train_metrics['avg_speed_ms']:.1f}ms/image")
            print(f"   Learning Rate: {train_metrics['lr']:.2e}")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'val_miou': val_metrics['miou'],
                    'hazard_recall': val_metrics['hazard_recall'],
                    'obstacle_recall': val_metrics['obstacle_recall'],
                    'safe_precision': val_metrics['safe_precision'],
                    'train_speed_ms': train_metrics['avg_speed_ms'],
                    'learning_rate': train_metrics['lr'],
                    'best_miou': self.best_miou
                })
        
        print(f"\n🎯 Training completed!")
        print(f"   Best mIoU: {self.best_miou:.4f}")
        print(f"   Final model saved: {self.checkpoint_dir / 'best_model.pth'}")
        
        # Final model benchmarking
        self._benchmark_final_model()
        
        return self.training_history
    
    def _benchmark_final_model(self):
        """Benchmark the final model for deployment readiness."""
        
        print(f"\n🚀 Benchmarking final model...")
        
        # Load best model
        self.model.eval()
        
        # CPU benchmarking
        cpu_stats = benchmark_model(self.model, device='cpu', num_runs=50)
        print(f"   CPU Performance: {cpu_stats['mean_ms']:.1f}±{cpu_stats['std_ms']:.1f}ms")
        
        # GPU benchmarking (if available)
        if torch.cuda.is_available():
            cuda_stats = benchmark_model(self.model, device='cuda', num_runs=100)
            print(f"   CUDA Performance: {cuda_stats['mean_ms']:.1f}±{cuda_stats['std_ms']:.1f}ms")
        
        # Model size analysis
        model_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        print(f"   Model Size: {model_size:.1f} MB")
        
        # ONNX conversion test
        try:
            from models.edge_landing_net import convert_to_onnx
            onnx_path = self.checkpoint_dir / 'edge_landing_final.onnx'
            convert_to_onnx(self.model, str(onnx_path))
            print(f"   ONNX Export: ✅ {onnx_path}")
        except Exception as e:
            print(f"   ONNX Export: ❌ {e}")


def main():
    parser = argparse.ArgumentParser(description='Train Edge UAV Landing Model')
    parser.add_argument('--data_root', default='../datasets', help='Root directory for datasets')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--input_size', type=int, default=256, help='Input image size')
    parser.add_argument('--model_type', default='standard', choices=['standard', 'fast'], help='Model variant')
    parser.add_argument('--device', default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--checkpoint_dir', default='outputs/edge_training', help='Checkpoint directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Data loader workers')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🚁 Edge UAV Landing Training")
    print(f"   Device: {device}")
    print(f"   Data root: {args.data_root}")
    print(f"   Model type: {args.model_type}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Input size: {args.input_size}")
    print(f"   Epochs: {args.num_epochs}")
    
    # Create datasets
    print(f"\n📚 Loading datasets...")
    datasets = create_edge_datasets(
        data_root=args.data_root,
        input_size=args.input_size,
        extreme_augmentation=True
    )
    
    if len(datasets['train']) == 0:
        print("❌ No training data found! Check dataset paths.")
        return
    
    # Create data loaders
    train_loader = DataLoader(
        datasets['train'],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        datasets['val'],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    print(f"\n🤖 Creating {args.model_type} model...")
    model = create_edge_model(
        model_type=args.model_type,
        num_classes=6,
        input_size=args.input_size
    )
    
    # Get class weights
    class_weights = datasets['train'].get_class_weights()
    print(f"   Class weights: {class_weights}")
    
    # Create trainer
    trainer = EdgeLandingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        use_wandb=args.use_wandb,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train model
    try:
        training_history = trainer.train(
            num_epochs=args.num_epochs,
            lr=args.lr,
            class_weights=class_weights
        )
        
        print(f"\n✅ Training completed successfully!")
        print(f"   Best model: {args.checkpoint_dir}/best_model.pth")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 