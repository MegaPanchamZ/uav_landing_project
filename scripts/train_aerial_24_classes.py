#!/usr/bin/env python3
"""
Train Aerial Semantic Segmentation with 24 Classes
=================================================

Train a segmentation model on the full 24 semantic classes from the Semantic Drone Dataset.
This preserves semantic richness for neuro-symbolic reasoning with Scallop.

Key differences from previous approach:
- 24 classes instead of 4 (preserves information)
- Aerial-specific training (no Cityscapes pretraining)
- Proper class weighting for 24-class distribution
- Enhanced architectures with sufficient capacity
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import wandb
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from datasets.aerial_semantic_24_dataset import AerialSemantic24Dataset, create_aerial_semantic_transforms
from models.enhanced_architectures import EnhancedBiSeNetV2
from losses.safety_aware_losses import MultiComponentSemanticLoss
from evaluation.safety_metrics import SemanticSegmentationEvaluator


class AerialSemantic24Trainer:
    """Trainer for 24-class aerial semantic segmentation."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = GradScaler('cuda') if self.device.type == 'cuda' else None
        
        # Initialize wandb
        if config.use_wandb:
            wandb.init(
                project="uav-landing-24-class-semantic",
                name=f"aerial_semantic_24_{config.model_name}",
                config=config.__dict__
            )
        
        print(f"🛩️  Aerial Semantic 24-Class Training")
        print(f"   Device: {self.device}")
        print(f"   Model: {config.model_name}")
        print(f"   Classes: 24 semantic classes")
        print(f"   Resolution: {config.input_resolution}")
        print(f"   Batch size: {config.batch_size}")
        
    def setup_data(self):
        """Setup datasets and data loaders."""
        
        # Training transforms with aerial-specific augmentations
        train_transforms = create_aerial_semantic_transforms(
            input_size=self.config.input_resolution,
            is_training=True,
            advanced_augmentation=True
        )
        
        # Validation transforms (no augmentation)
        val_transforms = create_aerial_semantic_transforms(
            input_size=self.config.input_resolution,
            is_training=False,
            advanced_augmentation=False
        )
        
        # Create datasets
        self.train_dataset = AerialSemantic24Dataset(
            data_root=self.config.data_root,
            split="train",
            transform=train_transforms,
            target_resolution=self.config.input_resolution,
            return_confidence=True,
            use_random_crops=True,
            crops_per_image=self.config.crops_per_image
        )
        
        self.val_dataset = AerialSemantic24Dataset(
            data_root=self.config.data_root,
            split="val",
            transform=val_transforms,
            target_resolution=self.config.input_resolution,
            return_confidence=True,
            use_random_crops=False,
            crops_per_image=1
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        print(f"📊 Dataset Summary:")
        print(f"   Training samples: {len(self.train_dataset)}")
        print(f"   Validation samples: {len(self.val_dataset)}")
        print(f"   Training batches: {len(self.train_loader)}")
        print(f"   Validation batches: {len(self.val_loader)}")
        
        # Analyze class distribution for proper weighting
        distribution = self.train_dataset.get_class_distribution()
        relevance_stats = self.train_dataset.get_landing_relevant_stats()
        
        print(f"\n🎯 Landing Relevance Distribution:")
        for relevance, percentage in relevance_stats.items():
            print(f"   {relevance}: {percentage:.1f}%")
        
        return distribution
    
    def setup_model(self, class_distribution):
        """Setup model, loss, and optimizer."""
        
        # Calculate class weights for balanced training
        class_weights = self._compute_class_weights(class_distribution)
        
        # Create model (NO Cityscapes pretraining for proper aerial domain)
        if self.config.model_name == "enhanced_bisenetv2":
            self.model = EnhancedBiSeNetV2(
                num_classes=24,  # Full semantic richness
                input_resolution=self.config.input_resolution,
                backbone=self.config.backbone,
                use_attention=True,
                uncertainty_estimation=True,
                dropout_rate=0.1
            )
        else:
            raise ValueError(f"Unknown model: {self.config.model_name}")
        
        self.model = self.model.to(self.device)
        
        # Multi-component loss for 24-class semantic segmentation
        self.criterion = MultiComponentSemanticLoss(
            num_classes=24,
            class_weights=class_weights.to(self.device),
            use_focal_loss=True,
            use_dice_loss=True,
            use_uncertainty_loss=True,
            semantic_weights={
                'safe': 1.0,      # Standard weight for safe classes
                'caution': 1.5,   # Higher weight for caution (important for assessment)
                'danger': 2.0,    # Highest weight for danger (critical for safety)
                'obstacle': 1.8,  # High weight for obstacles
                'landmark': 0.8,  # Lower weight for landmarks
                'unknown': 0.5    # Low weight for unknown
            }
        )
        
        # Optimizer with differential learning rates
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': self.config.learning_rate * 0.1},  # Lower LR for backbone
            {'params': head_params, 'lr': self.config.learning_rate}  # Higher LR for heads
        ], weight_decay=self.config.weight_decay)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Evaluator for comprehensive metrics
        self.evaluator = SemanticSegmentationEvaluator(
            num_classes=24,
            class_names=[info["name"] for info in self.train_dataset.class_info.values()],
            landing_relevance_map={
                class_id: info["landing_relevance"] 
                for class_id, info in self.train_dataset.class_info.items()
            }
        )
        
        print(f"🏗️  Model Setup Complete:")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Backbone LR: {self.config.learning_rate * 0.1:.2e}")
        print(f"   Head LR: {self.config.learning_rate:.2e}")
        print(f"   Class weights range: {class_weights.min():.3f} - {class_weights.max():.3f}")
    
    def _compute_class_weights(self, class_distribution):
        """Compute class weights for balanced training."""
        
        # Convert distribution to class counts
        total_pixels = sum(stats["count"] for stats in class_distribution.values())
        class_counts = torch.zeros(24)
        
        for class_id, info in self.train_dataset.class_info.items():
            class_name = info["name"]
            if class_name in class_distribution:
                class_counts[class_id] = class_distribution[class_name]["count"]
            else:
                class_counts[class_id] = 1  # Avoid division by zero
        
        # Compute inverse frequency weights
        class_weights = total_pixels / (24 * class_counts)
        
        # Apply landing relevance multipliers
        for class_id, info in self.train_dataset.class_info.items():
            relevance = info["landing_relevance"]
            if relevance == "danger":
                class_weights[class_id] *= 2.0  # Emphasize danger classes
            elif relevance == "caution":
                class_weights[class_id] *= 1.5  # Emphasize caution classes
            elif relevance == "unknown":
                class_weights[class_id] *= 0.5  # De-emphasize unknown classes
        
        # Normalize and clip weights
        class_weights = torch.clamp(class_weights, min=0.1, max=50.0)
        
        return class_weights
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            confidence = batch.get('confidence', None)
            if confidence is not None:
                confidence = confidence.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks, confidence)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks, confidence)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log to wandb
            if self.config.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        self.evaluator.reset()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                confidence = batch.get('confidence', None)
                if confidence is not None:
                    confidence = confidence.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks, confidence)
                
                # Get predictions
                if isinstance(outputs, dict):
                    predictions = outputs['main']
                else:
                    predictions = outputs
                
                predictions = torch.argmax(predictions, dim=1)
                
                # Update evaluator
                self.evaluator.update(predictions.cpu(), masks.cpu())
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({
                    'val_loss': f"{loss.item():.4f}",
                    'avg_val_loss': f"{total_loss / num_batches:.4f}"
                })
        
        # Compute metrics
        metrics = self.evaluator.compute_metrics()
        avg_loss = total_loss / num_batches
        
        print(f"\n📊 Validation Results (Epoch {epoch}):")
        print(f"   Loss: {avg_loss:.4f}")
        print(f"   mIoU: {metrics['miou']:.3f}")
        print(f"   Overall Accuracy: {metrics['overall_accuracy']:.3f}")
        print(f"   Mean Accuracy: {metrics['mean_accuracy']:.3f}")
        
        # Log landing relevance metrics
        if 'landing_relevance_metrics' in metrics:
            relevance_metrics = metrics['landing_relevance_metrics']
            print(f"\n🎯 Landing Relevance Performance:")
            for relevance, metric_dict in relevance_metrics.items():
                print(f"   {relevance}: IoU={metric_dict['iou']:.3f}, Acc={metric_dict['accuracy']:.3f}")
        
        # Log to wandb
        if self.config.use_wandb:
            wandb_metrics = {
                'val/loss': avg_loss,
                'val/miou': metrics['miou'],
                'val/overall_accuracy': metrics['overall_accuracy'],
                'val/mean_accuracy': metrics['mean_accuracy'],
                'epoch': epoch
            }
            
            if 'landing_relevance_metrics' in metrics:
                for relevance, metric_dict in metrics['landing_relevance_metrics'].items():
                    wandb_metrics[f'val/{relevance}_iou'] = metric_dict['iou']
                    wandb_metrics[f'val/{relevance}_accuracy'] = metric_dict['accuracy']
            
            wandb.log(wandb_metrics)
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.output_dir) / f"aerial_semantic_24_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config.output_dir) / "aerial_semantic_24_best.pth"
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model: {best_path}")
        
        print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    def train(self):
        """Main training loop."""
        print(f"\n🚀 Starting training for {self.config.num_epochs} epochs...")
        
        # Setup data and model
        class_distribution = self.setup_data()
        self.setup_model(class_distribution)
        
        best_miou = 0
        
        for epoch in range(1, self.config.num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{self.config.num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, metrics = self.validate_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint
            is_best = metrics['miou'] > best_miou
            if is_best:
                best_miou = metrics['miou']
            
            if epoch % self.config.save_every == 0 or is_best:
                self.save_checkpoint(epoch, metrics, is_best)
            
            print(f"\n📈 Epoch {epoch} Summary:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Val mIoU: {metrics['miou']:.3f} (Best: {best_miou:.3f})")
        
        print(f"\n🎉 Training completed!")
        print(f"   Best mIoU: {best_miou:.3f}")
        
        if self.config.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train 24-class aerial semantic segmentation")
    
    # Data arguments
    parser.add_argument('--data_root', type=str, 
                       default='datasets/Aerial_Semantic_Segmentation_Drone_Dataset/dataset/semantic_drone_dataset',
                       help='Path to semantic drone dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/aerial_semantic_24',
                       help='Output directory for checkpoints')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='enhanced_bisenetv2',
                       choices=['enhanced_bisenetv2'], help='Model architecture')
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet50', 'resnet101'], help='Backbone architecture')
    parser.add_argument('--input_resolution', type=int, nargs=2, default=[512, 512],
                       help='Input resolution [height, width]')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--crops_per_image', type=int, default=6, help='Random crops per image')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train
    trainer = AerialSemantic24Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main() 