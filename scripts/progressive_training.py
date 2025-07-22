#!/usr/bin/env python3
"""
Progressive Fine-Tuning for UAV Landing Detection
================================================

Implements progressive fine-tuning strategy:
1. Stage 1: DroneDeploy (4-channel RGB+Height, coarse labels) + BiSeNetV2 Transfer Learning
2. Stage 2: UDD5 (3-channel, medium granularity) 
3. Stage 3: Aerial Semantic (3-channel, fine granularity, 10-30m aerial)

This matches the user's requested strategy for optimal transfer learning.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any
import wandb

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from datasets.drone_deploy_dataset import DroneDeployDataset, create_drone_deploy_transforms
from datasets.udd_dataset import UDDDataset, create_udd_transforms  
from datasets.semantic_drone_dataset import SemanticDroneDataset, create_semantic_drone_transforms
from datasets.enhanced_augmentation import MultiScaleAugmentedDataset, create_augmented_datasets
from datasets.cached_augmentation import CachedAugmentedDataset
from models.enhanced_architectures import create_enhanced_model
from losses.safety_aware_losses import CombinedSafetyLoss
from safety_evaluation.safety_metrics import SafetyAwareEvaluator
from torch.utils.data import DataLoader


class ProgressiveTrainer:
    """Progressive fine-tuning trainer."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize wandb
        wandb.init(
            project="uav-landing-progressive",
            name=f"progressive_training_{wandb.util.generate_id()}",
            config=config
        )
        
        print("🚀 Progressive UAV Landing Training")
        print("=" * 50)
        print(f"📊 Strategy: DroneDeploy → UDD5 → Aerial Semantic")
        print(f"🏗️ Model: BiSeNetV2 with Cityscapes transfer learning")
        print(f"💾 Device: {self.device}")
        print(f"📁 Output: {config['output_dir']}")
        
    def create_stage1_dataset(self) -> Dict[str, DataLoader]:
        """Stage 1: DroneDeploy with height maps (4-channel)."""
        print("\n🏗️ Stage 1: DroneDeploy Dataset (RGB + Height)")
        
        # Base datasets without heavy transforms (augmentation handles this)
        train_dataset = DroneDeployDataset(
            data_root=self.config['drone_deploy_path'],
            split='train',
            transform=None,  # Raw data for augmentation
            use_height=True
        )
        
        val_dataset = DroneDeployDataset(
            data_root=self.config['drone_deploy_path'],
            split='val',
            transform=create_drone_deploy_transforms(is_training=False),
            use_height=True
        )
        
        # Apply augmentation to training set
        if self.config.get('use_cached_augmentation', False):
            print(f"   💾 Loading cached augmented dataset...")
            train_dataset = CachedAugmentedDataset(
                base_dataset=train_dataset,
                cache_dir=self.config.get('cache_dir', 'cache/augmented_datasets'),
                dataset_name='drone_deploy',
                patch_scales=[(512, 512), (768, 768)],
                augmentation_factor=self.config.get('augmentation_factor', 20),
                min_object_ratio=0.1,
                use_overlapping=True,
                overlap_ratio=0.3,
                uav_augmentations=True,
                force_rebuild=False
            )
        elif self.config.get('use_augmentation', True):
            print(f"   🚀 Applying real-time multi-scale augmentation...")
            train_dataset = MultiScaleAugmentedDataset(
                base_dataset=train_dataset,
                patch_scales=[(512, 512), (768, 768)],
                patches_per_image=self.config.get('augmentation_factor', 15),
                min_object_ratio=0.1,
                use_overlapping=True,
                overlap_ratio=0.3,
                uav_augmentations=True
            )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,  # Disable multiprocessing
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,  # Disable multiprocessing
            pin_memory=False
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'in_channels': 4,  # RGB + Height
            'stage_name': 'drone_deploy'
        }
    
    def create_stage2_dataset(self) -> Dict[str, DataLoader]:
        """Stage 2: UDD5 dataset (3-channel)."""
        print("\n🏗️ Stage 2: UDD5 Dataset (Urban Drone)")
        
        # Base datasets
        train_dataset = UDDDataset(
            data_root=self.config['udd_path'],
            split='train',
            transform=None  # Raw data for augmentation
        )
        
        val_dataset = UDDDataset(
            data_root=self.config['udd_path'],
            split='val',
            transform=create_udd_transforms(is_training=False)
        )
        
        # Apply augmentation to training set
        if self.config.get('use_cached_augmentation', False):
            print(f"   💾 Loading cached augmented dataset...")
            train_dataset = CachedAugmentedDataset(
                base_dataset=train_dataset,
                cache_dir=self.config.get('cache_dir', 'cache/augmented_datasets'),
                dataset_name='udd',
                patch_scales=[(512, 512)],
                augmentation_factor=self.config.get('augmentation_factor', 15),
                min_object_ratio=0.15,
                use_overlapping=True,
                overlap_ratio=0.2,
                uav_augmentations=True,
                force_rebuild=False
            )
        elif self.config.get('use_augmentation', True):
            print(f"   🚀 Applying real-time multi-scale augmentation...")
            train_dataset = MultiScaleAugmentedDataset(
                base_dataset=train_dataset,
                patch_scales=[(512, 512)],
                patches_per_image=self.config.get('augmentation_factor', 8),
                min_object_ratio=0.15,
                use_overlapping=True,
                overlap_ratio=0.2,
                uav_augmentations=True
            )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,  # Disable multiprocessing
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,  # Disable multiprocessing
            pin_memory=False
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'in_channels': 3,  # RGB only
            'stage_name': 'udd5'
        }
    
    def create_stage3_dataset(self) -> Dict[str, DataLoader]:
        """Stage 3: Aerial Semantic dataset (3-channel, fine-tuning)."""
        print("\n🏗️ Stage 3: Aerial Semantic Dataset (10-30m Aerial)")
        
        # Base datasets  
        train_dataset = SemanticDroneDataset(
            data_root=self.config['semantic_drone_path'],
            split='train',
            transform=None,  # Raw data for augmentation
            class_mapping='enhanced_4_class'
        )
        
        val_dataset = SemanticDroneDataset(
            data_root=self.config['semantic_drone_path'],
            split='val',
            transform=create_semantic_drone_transforms(is_training=False),
            class_mapping='enhanced_4_class'
        )
        
        # Apply augmentation to training set (MAXIMUM augmentation for best dataset)
        if self.config.get('use_cached_augmentation', False):
            print(f"   💾 Loading cached augmented dataset (MAXIMUM)...")
            train_dataset = CachedAugmentedDataset(
                base_dataset=train_dataset,
                cache_dir=self.config.get('cache_dir', 'cache/augmented_datasets'),
                dataset_name='semantic_drone',
                patch_scales=[(512, 512), (768, 768), (1024, 1024)],  # All scales
                augmentation_factor=self.config.get('augmentation_factor', 25),  # Maximum patches
                min_object_ratio=0.05,  # More permissive for diversity
                use_overlapping=True,
                overlap_ratio=0.25,
                uav_augmentations=True,  # Full UAV augmentations
                force_rebuild=False
            )
        elif self.config.get('use_augmentation', True):
            print(f"   🚀 Applying MAXIMUM real-time multi-scale augmentation...")
            train_dataset = MultiScaleAugmentedDataset(
                base_dataset=train_dataset,
                patch_scales=[(512, 512), (768, 768), (1024, 1024)],  # All scales
                patches_per_image=self.config.get('augmentation_factor', 25),  # Maximum patches
                min_object_ratio=0.05,  # More permissive for diversity
                use_overlapping=True,
                overlap_ratio=0.25,
                uav_augmentations=True  # Full UAV augmentations
            )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,  # Disable multiprocessing
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,  # Disable multiprocessing
            pin_memory=False
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'in_channels': 3,  # RGB only
            'stage_name': 'aerial_semantic'
        }
    
    def create_model(self, in_channels: int, stage: str) -> nn.Module:
        """Create model with appropriate input channels."""
        print(f"\n🏗️ Creating model for {stage} (in_channels={in_channels})")
        
        model = create_enhanced_model(
            model_type='mmseg_bisenetv2',
            num_classes=4,
            uncertainty_estimation=True,
            in_channels=in_channels
        )
        
        return model.to(self.device)
    
    def train_stage(self, stage_data: Dict, stage_num: int, pretrained_model: nn.Module = None) -> nn.Module:
        """Train a single stage."""
        stage_name = stage_data['stage_name']
        in_channels = stage_data['in_channels']
        
        print(f"\n🚀 Starting Stage {stage_num}: {stage_name}")
        print(f"   Input channels: {in_channels}")
        print(f"   Epochs: {self.config['epochs_per_stage']}")
        
        # Create model
        if pretrained_model is None:
            model = self.create_model(in_channels, stage_name)
        else:
            # Adapt previous model to new input channels
            model = self.adapt_model(pretrained_model, in_channels, stage_name)
        
        # Create loss function
        criterion = CombinedSafetyLoss(
            safety_weights=[1.0, 2.0, 1.5, 3.0]  # Background, Safe, Caution, Danger
        )
        
        # Create optimizer with different learning rates per stage
        lr = self.config['base_lr'] * (0.5 ** (stage_num - 1))  # Decrease LR each stage
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['epochs_per_stage']
        )
        
        # Training loop
        best_val_iou = 0.0
        
        for epoch in range(self.config['epochs_per_stage']):
            print(f"\n--- Stage {stage_num}, Epoch {epoch+1}/{self.config['epochs_per_stage']} ---")
            
            # Train
            train_loss = self.train_epoch(model, stage_data['train'], criterion, optimizer)
            
            # Validate  
            val_loss, val_metrics = self.validate_epoch(model, stage_data['val'], criterion)
            
            # Update scheduler
            scheduler.step()
            
            # Log to wandb
            wandb.log({
                f'stage_{stage_num}/epoch': epoch,
                f'stage_{stage_num}/train_loss': train_loss,
                f'stage_{stage_num}/val_loss': val_loss,
                f'stage_{stage_num}/val_iou': val_metrics['mean_iou'],
                f'stage_{stage_num}/learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # Save best model
            if val_metrics['mean_iou'] > best_val_iou:
                best_val_iou = val_metrics['mean_iou']
                self.save_model(model, stage_num, stage_name, epoch, val_metrics)
            
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Val IoU: {val_metrics['mean_iou']:.4f}")
            print(f"   Best IoU: {best_val_iou:.4f}")
        
        print(f"✅ Stage {stage_num} completed. Best IoU: {best_val_iou:.4f}")
        return model
    
    def adapt_model(self, model: nn.Module, new_in_channels: int, stage_name: str) -> nn.Module:
        """Adapt model input channels for next stage."""
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'detail'):
            current_in_channels = model.backbone.detail.detail_branch[0][0].conv.in_channels
            
            if current_in_channels != new_in_channels:
                print(f"   🔄 Adapting input channels: {current_in_channels} → {new_in_channels}")
                
                # Create new model with correct input channels
                new_model = self.create_model(new_in_channels, stage_name)
                
                # Copy compatible weights
                self.copy_compatible_weights(model, new_model)
                
                return new_model
        
        return model
    
    def copy_compatible_weights(self, source_model: nn.Module, target_model: nn.Module):
        """Copy compatible weights between models."""
        source_dict = source_model.state_dict()
        target_dict = target_model.state_dict()
        
        copied = 0
        total = len(target_dict)
        
        for name, param in target_dict.items():
            if name in source_dict and source_dict[name].shape == param.shape:
                param.copy_(source_dict[name])
                copied += 1
        
        print(f"   📋 Copied {copied}/{total} compatible weights ({100*copied/total:.1f}%)")
    
    def train_epoch(self, model, dataloader, criterion, optimizer):
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"   Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        return total_loss / len(dataloader)
    
    def validate_epoch(self, model, dataloader, criterion):
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        
        # Simple IoU calculation
        intersection = torch.zeros(4)
        union = torch.zeros(4)
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                total_loss += loss.item()
                
                # Calculate IoU
                pred = torch.argmax(outputs, dim=1)
                
                for class_id in range(4):
                    pred_mask = (pred == class_id)
                    true_mask = (masks == class_id)
                    
                    intersection[class_id] += (pred_mask & true_mask).sum().item()
                    union[class_id] += (pred_mask | true_mask).sum().item()
        
        # Calculate mean IoU
        iou_per_class = intersection / (union + 1e-8)
        mean_iou = iou_per_class.mean().item()
        
        return total_loss / len(dataloader), {'mean_iou': mean_iou, 'iou_per_class': iou_per_class}
    
    def save_model(self, model, stage_num, stage_name, epoch, metrics):
        """Save model checkpoint."""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'stage': stage_num,
            'stage_name': stage_name,
            'epoch': epoch,
            'metrics': metrics,
            'config': self.config
        }
        
        save_path = output_dir / f"stage_{stage_num}_{stage_name}_best.pth"
        torch.save(checkpoint, save_path)
        print(f"   💾 Saved: {save_path}")
    
    def run_progressive_training(self):
        """Run the complete progressive training pipeline."""
        print("\n🚀 Starting Progressive Fine-Tuning Pipeline")
        
        # Stage 1: DroneDeploy (4-channel with height maps)
        stage1_data = self.create_stage1_dataset()
        model = self.train_stage(stage1_data, stage_num=1)
        
        # Stage 2: UDD5 (3-channel, adapt from 4-channel)  
        stage2_data = self.create_stage2_dataset()
        model = self.train_stage(stage2_data, stage_num=2, pretrained_model=model)
        
        # Stage 3: Aerial Semantic (3-channel, fine-tuning)
        stage3_data = self.create_stage3_dataset()
        model = self.train_stage(stage3_data, stage_num=3, pretrained_model=model)
        
        # Final evaluation
        print("\n🎯 Progressive Training Complete!")
        print("Models saved in:", self.config['output_dir'])
        
        wandb.finish()
        return model


def main():
    parser = argparse.ArgumentParser(description="Progressive Fine-Tuning for UAV Landing Detection")
    
    # Dataset paths
    parser.add_argument('--drone-deploy-path', type=str, required=True,
                        help='Path to DroneDeploy dataset')
    parser.add_argument('--udd-path', type=str, required=True,
                        help='Path to UDD5 dataset')
    parser.add_argument('--semantic-drone-path', type=str, required=True,
                        help='Path to Aerial Semantic dataset')
    
    # Training parameters
    parser.add_argument('--epochs-per-stage', type=int, default=10,
                        help='Epochs per training stage')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--base-lr', type=float, default=1e-3,
                        help='Base learning rate (decreases each stage)')
    
    # Augmentation parameters
    parser.add_argument('--use-augmentation', action='store_true', default=True,
                        help='Use multi-scale augmentation')
    parser.add_argument('--augmentation-factor', type=int, default=20,
                        help='Augmentation factor for patch extraction')
    parser.add_argument('--use-cached-augmentation', action='store_true', default=False,
                        help='Use pre-cached augmented datasets (much faster)')
    parser.add_argument('--cache-dir', type=str, default='cache/augmented_datasets',
                        help='Directory containing cached augmented datasets')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/progressive_training',
                        help='Output directory for models')
    
    args = parser.parse_args()
    
    # Create config
    config = {
        'drone_deploy_path': args.drone_deploy_path,
        'udd_path': args.udd_path,
        'semantic_drone_path': args.semantic_drone_path,
        'epochs_per_stage': args.epochs_per_stage,
        'batch_size': args.batch_size,
        'base_lr': args.base_lr,
        'output_dir': args.output_dir,
        'use_augmentation': args.use_augmentation,
        'augmentation_factor': args.augmentation_factor,
        'use_cached_augmentation': args.use_cached_augmentation,
        'cache_dir': args.cache_dir
    }
    
    # Create trainer and run
    trainer = ProgressiveTrainer(config)
    final_model = trainer.run_progressive_training()
    
    print("🎉 Progressive training completed successfully!")


if __name__ == "__main__":
    main() 