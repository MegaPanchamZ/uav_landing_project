#!/usr/bin/env python3
"""
Fixed Training Strategy
======================

This script implements several fixes for the catastrophic failure in progressive training:
1. Individual dataset training (baseline test)
2. Reverse progressive training (general→specific)
3. Better regularization and early stopping
4. Class balancing strategies
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any
import wandb
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from datasets.drone_deploy_dataset import DroneDeployDataset, create_drone_deploy_transforms
from datasets.udd_dataset import UDDDataset, create_udd_transforms  
from datasets.semantic_drone_dataset import SemanticDroneDataset, create_semantic_drone_transforms
from datasets.cached_augmentation import CachedAugmentedDataset
from models.enhanced_architectures import create_enhanced_model
from losses.safety_aware_losses import CombinedSafetyLoss
from safety_evaluation.safety_metrics import SafetyAwareEvaluator
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        return self.counter >= self.patience
    
    def save_checkpoint(self, model):
        """Save model weights."""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()
    
    def restore(self, model):
        """Restore best weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def adaptive_collate_fn(batch):
    """Custom collate function that resizes all patches to 512x512."""
    import torch.nn.functional as F
    
    images = [item['image'] for item in batch]
    masks = [item['mask'] for item in batch]
    
    target_size = (512, 512)
    
    # Resize all images and masks to target size
    resized_images = []
    resized_masks = []
    
    for img, mask in zip(images, masks):
        if img.shape[-2:] != target_size:
            img_resized = F.interpolate(
                img.unsqueeze(0),
                size=target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        else:
            img_resized = img
            
        if mask.shape[-2:] != target_size:
            mask_resized = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=target_size,
                mode='nearest'
            ).squeeze(0).squeeze(0).long()
        else:
            mask_resized = mask
            
        resized_images.append(img_resized)
        resized_masks.append(mask_resized)
    
    # Stack resized tensors
    batch_dict = {
        'image': torch.stack(resized_images),
        'mask': torch.stack(resized_masks),
    }
    
    # Add other metadata
    for key in ['image_path', 'base_idx', 'quality_score', 'scale']:
        if key in batch[0]:
            batch_dict[key] = [item[key] for item in batch]
    
    return batch_dict


class FixedTrainer:
    """Improved trainer with better regularization and strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("🔧 Fixed UAV Landing Training")
        print("=" * 50)
        print(f"💾 Device: {self.device}")
        print(f"🎯 Strategy: {config['strategy']}")
        
        # Initialize wandb
        wandb.init(
            project="uav-landing-fixed",
            name=f"{config['strategy']}_training_{wandb.util.generate_id()}",
            config=config,
            tags=["fixed", config['strategy'], "regularized"]
        )
    
    def compute_class_weights(self, dataset, dataset_name: str) -> torch.Tensor:
        """Compute class weights to handle class imbalance."""
        print(f"   📊 Computing class weights for {dataset_name}...")
        
        # Sample dataset to get class distribution
        sample_size = min(50, len(dataset))
        class_counts = np.zeros(4)
        
        for i in range(sample_size):
            try:
                sample = dataset[i]
                mask = sample['mask']
                if isinstance(mask, torch.Tensor):
                    mask = mask.numpy()
                
                unique, counts = np.unique(mask, return_counts=True)
                for cls, count in zip(unique, counts):
                    if cls < 4:  # Ensure class is valid
                        class_counts[cls] += count
            except:
                continue
        
        # Compute weights (inverse frequency)
        total_pixels = class_counts.sum()
        if total_pixels > 0:
            class_weights = total_pixels / (4 * class_counts + 1e-8)  # Avoid division by zero
            class_weights = class_weights / class_weights.mean()  # Normalize
        else:
            class_weights = np.ones(4)
        
        weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        
        print(f"   📈 Class distribution: {class_counts}")
        print(f"   ⚖️ Class weights: {class_weights}")
        
        return weights_tensor
    
    def create_dataset(self, dataset_name: str, dataset_path: str, dataset_class, use_cache: bool = True) -> Dict[str, Any]:
        """Create dataset with proper configuration."""
        print(f"\n🏗️ Creating {dataset_name} Dataset")
        
        # Base datasets
        if dataset_name == 'DroneDeploy':
            train_dataset = dataset_class(
                data_root=dataset_path,
                split='train',
                transform=None,
                use_height=True
            )
            val_dataset = dataset_class(
                data_root=dataset_path,
                split='val',
                transform=create_drone_deploy_transforms(is_training=False),
                use_height=True
            )
            in_channels = 4
        else:
            train_dataset = dataset_class(
                data_root=dataset_path,
                split='train',
                transform=None
            )
            val_dataset = dataset_class(
                data_root=dataset_path,
                split='val',
                transform=create_udd_transforms(is_training=False) if dataset_name == 'UDD'
                          else create_semantic_drone_transforms(is_training=False)
            )
            in_channels = 3
        
        # Use cached dataset if requested
        if use_cache:
            cache_name_mapping = {
                'DroneDeploy': 'drone_deploy',
                'UDD': 'udd',
                'Semantic Drone': 'semantic_drone'
            }
            
            # Cache parameters matching existing cache
            if dataset_name == 'DroneDeploy':
                cache_params = {
                    'patch_scales': [(512, 512), (768, 768)],
                    'augmentation_factor': 10,  # Reduced from 20
                    'min_object_ratio': 0.1,
                    'overlap_ratio': 0.3
                }
            elif dataset_name == 'UDD':
                cache_params = {
                    'patch_scales': [(512, 512)],
                    'augmentation_factor': 8,  # Reduced from 15
                    'min_object_ratio': 0.15,
                    'overlap_ratio': 0.2
                }
            else:  # Semantic Drone
                cache_params = {
                    'patch_scales': [(512, 512), (768, 768), (1024, 1024)],
                    'augmentation_factor': 10,  # Reduced from 25
                    'min_object_ratio': 0.05,
                    'overlap_ratio': 0.25
                }
            
            train_dataset = CachedAugmentedDataset(
                base_dataset=train_dataset,
                cache_dir=self.config.get('cache_dir', 'cache/augmented_datasets'),
                dataset_name=cache_name_mapping[dataset_name],
                patch_scales=cache_params['patch_scales'],
                augmentation_factor=cache_params['augmentation_factor'],
                min_object_ratio=cache_params['min_object_ratio'],
                use_overlapping=True,
                overlap_ratio=cache_params['overlap_ratio'],
                uav_augmentations=True,
                force_rebuild=False
            )
        
        # Compute class weights
        class_weights = self.compute_class_weights(train_dataset, dataset_name)
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=adaptive_collate_fn,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'in_channels': in_channels,
            'class_weights': class_weights,
            'dataset_name': dataset_name.lower().replace(' ', '_')
        }
    
    def create_model(self, in_channels: int, class_weights: torch.Tensor, pretrained_model=None):
        """Create model with proper initialization."""
        
        if pretrained_model is None:
            # Create new model
            model = create_enhanced_model(
                model_type="mmseg_bisenetv2",
                num_classes=4,
                in_channels=in_channels,
                uncertainty_estimation=True,
                pretrained_path="../model_pths/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes_20210902_015551-bcf10f09.pth"
            )
        else:
            # Adapt existing model for different input channels
            if in_channels != pretrained_model.backbone.backbone.stem.conv_3x3.in_channels:
                model = self._adapt_model_channels(pretrained_model, in_channels)
            else:
                model = pretrained_model
        
        model = model.to(self.device)
        
        # Create loss with class weights
        criterion = CombinedSafetyLoss(num_classes=4, safety_weights=class_weights.tolist())
        
        return model, criterion
    
    def _adapt_model_channels(self, pretrained_model, new_in_channels):
        """Adapt model for different input channels."""
        print(f"   🔧 Adapting input channels to {new_in_channels}")
        
        # Create new model with correct input channels
        new_model = create_enhanced_model(
            model_type="mmseg_bisenetv2",
            num_classes=4,
            in_channels=new_in_channels,
            uncertainty_estimation=True,
            pretrained_path=None
        )
        
        # Copy compatible weights
        new_state_dict = new_model.state_dict()
        old_state_dict = pretrained_model.state_dict()
        
        copied_layers = 0
        for name, param in old_state_dict.items():
            if name in new_state_dict:
                if 'conv_3x3' not in name:  # Skip input convolution
                    if new_state_dict[name].shape == param.shape:
                        new_state_dict[name] = param
                        copied_layers += 1
        
        new_model.load_state_dict(new_state_dict, strict=False)
        print(f"   ✅ Copied {copied_layers} layers from pretrained model")
        
        return new_model
    
    def train_single_dataset(self, dataset_name: str, dataset_path: str, dataset_class):
        """Train on a single dataset to establish baseline."""
        print(f"\n🎯 Training {dataset_name} (Individual)")
        
        # Create dataset
        dataset_info = self.create_dataset(dataset_name, dataset_path, dataset_class)
        
        # Create model
        model, criterion = self.create_model(
            dataset_info['in_channels'], 
            dataset_info['class_weights']
        )
        
        # Optimizer with better regularization
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.config['base_lr'],
            weight_decay=1e-3,  # Stronger weight decay
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Early stopping
        early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        
        print(f"   📈 Learning rate: {self.config['base_lr']:.2e}")
        print(f"   🏋️ Class weights: {dataset_info['class_weights']}")
        
        # Training loop
        best_iou = 0.0
        for epoch in range(self.config['max_epochs']):
            print(f"\n--- {dataset_name} Epoch {epoch+1}/{self.config['max_epochs']} ---")
            
            # Train
            train_loss = self.train_epoch(model, dataset_info['train'], criterion, optimizer)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(model, dataset_info['val'], criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Log metrics
            wandb.log({
                f'{dataset_name}/train_loss': train_loss,
                f'{dataset_name}/val_loss': val_loss,
                f'{dataset_name}/val_iou': val_metrics.get('miou', 0.0),
                f'{dataset_name}/lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch
            })
            
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Val IoU: {val_metrics.get('miou', 0.0):.4f}")
            
            # Early stopping check
            if early_stopping(val_loss, model):
                print(f"   🛑 Early stopping at epoch {epoch+1}")
                early_stopping.restore(model)
                break
            
            # Save best model
            if val_metrics.get('miou', 0.0) > best_iou:
                best_iou = val_metrics.get('miou', 0.0)
                self.save_model(model, dataset_name, epoch, val_metrics, 'individual')
        
        return model, best_iou
    
    def train_epoch(self, model, dataloader, criterion, optimizer):
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss_dict = criterion(outputs, masks)
            loss = loss_dict['total_loss']
            
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"     Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        return total_loss / len(dataloader)
    
    def validate_epoch(self, model, dataloader, criterion):
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        evaluator = SafetyAwareEvaluator(num_classes=4)
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = model(images)
                loss_dict = criterion(outputs, masks)
                loss = loss_dict['total_loss']
                
                total_loss += loss.item()
                
                # Update evaluator
                if isinstance(outputs, dict):
                    logits = outputs['main']
                else:
                    logits = outputs
                
                predictions = torch.argmax(logits, dim=1)
                evaluator.update(predictions, masks)
        
        metrics = evaluator.compute_metrics()
        return total_loss / len(dataloader), metrics
    
    def save_model(self, model, dataset_name, epoch, metrics, strategy):
        """Save model checkpoint."""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'dataset_name': dataset_name,
            'epoch': epoch,
            'metrics': metrics,
            'config': self.config,
            'strategy': strategy
        }
        
        save_path = output_dir / f"{strategy}_{dataset_name.lower().replace(' ', '_')}_best.pth"
        torch.save(checkpoint, save_path)
        print(f"   💾 Saved: {save_path}")
    
    def run_individual_training(self):
        """Train each dataset individually to establish baselines."""
        print("\n🎯 Individual Dataset Training Strategy")
        print("=" * 50)
        
        datasets_config = {
            'DroneDeploy': {
                'path': self.config['drone_deploy_path'],
                'class': DroneDeployDataset
            },
            'UDD': {
                'path': self.config['udd_path'],
                'class': UDDDataset
            },
            'Semantic Drone': {
                'path': self.config['semantic_drone_path'],
                'class': SemanticDroneDataset
            }
        }
        
        results = {}
        
        for dataset_name, config in datasets_config.items():
            try:
                model, best_iou = self.train_single_dataset(
                    dataset_name, 
                    config['path'], 
                    config['class']
                )
                results[dataset_name] = {
                    'best_iou': best_iou,
                    'model': model
                }
                print(f"✅ {dataset_name}: Best IoU = {best_iou:.4f}")
            except Exception as e:
                print(f"❌ Failed to train {dataset_name}: {e}")
                results[dataset_name] = {'error': str(e)}
        
        return results
    
    def run_reverse_progressive(self):
        """Run reverse progressive training: Semantic Drone → UDD → DroneDeploy."""
        print("\n🔄 Reverse Progressive Training Strategy")
        print("=" * 50)
        
        # Stage 1: Semantic Drone (largest, most general)
        print("🎯 Stage 1: Semantic Drone (Base)")
        dataset_info = self.create_dataset('Semantic Drone', self.config['semantic_drone_path'], SemanticDroneDataset)
        model, criterion = self.create_model(dataset_info['in_channels'], dataset_info['class_weights'])
        model = self.train_stage(model, dataset_info, criterion, stage_num=1, stage_name='semantic_drone')
        
        # Stage 2: UDD (medium size)
        print("🎯 Stage 2: UDD (Transfer)")
        dataset_info = self.create_dataset('UDD', self.config['udd_path'], UDDDataset)
        model, criterion = self.create_model(dataset_info['in_channels'], dataset_info['class_weights'], model)
        model = self.train_stage(model, dataset_info, criterion, stage_num=2, stage_name='udd')
        
        # Stage 3: DroneDeploy (smallest, most specific)
        print("🎯 Stage 3: DroneDeploy (Fine-tune)")
        dataset_info = self.create_dataset('DroneDeploy', self.config['drone_deploy_path'], DroneDeployDataset)
        model, criterion = self.create_model(dataset_info['in_channels'], dataset_info['class_weights'], model)
        model = self.train_stage(model, dataset_info, criterion, stage_num=3, stage_name='dronedeploy')
        
        print("✅ Reverse progressive training completed!")
        return model
    
    def train_stage(self, model, dataset_info, criterion, stage_num, stage_name):
        """Train a single stage with early stopping."""
        # Very conservative learning rate for transfer
        lr = self.config['base_lr'] * (0.1 ** (stage_num - 1))
        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr,
            weight_decay=1e-3
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        early_stopping = EarlyStopping(patience=8, min_delta=0.001)
        
        print(f"   📈 Learning rate: {lr:.2e}")
        
        best_iou = 0.0
        for epoch in range(self.config['epochs_per_stage']):
            print(f"\n--- Stage {stage_num} ({stage_name}), Epoch {epoch+1} ---")
            
            train_loss = self.train_epoch(model, dataset_info['train'], criterion, optimizer)
            val_loss, val_metrics = self.validate_epoch(model, dataset_info['val'], criterion)
            
            scheduler.step(val_loss)
            
            wandb.log({
                f'reverse_stage_{stage_num}/train_loss': train_loss,
                f'reverse_stage_{stage_num}/val_loss': val_loss,
                f'reverse_stage_{stage_num}/val_iou': val_metrics.get('miou', 0.0),
                f'reverse_stage_{stage_num}/lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch + (stage_num - 1) * self.config['epochs_per_stage']
            })
            
            print(f"   Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_metrics.get('miou', 0.0):.4f}")
            
            if early_stopping(val_loss, model):
                print(f"   🛑 Early stopping at epoch {epoch+1}")
                early_stopping.restore(model)
                break
            
            if val_metrics.get('miou', 0.0) > best_iou:
                best_iou = val_metrics.get('miou', 0.0)
                self.save_model(model, stage_name, epoch, val_metrics, f'reverse_stage_{stage_num}')
        
        return model


def main():
    parser = argparse.ArgumentParser(description="Fixed Training Strategy")
    
    # Dataset paths
    parser.add_argument('--drone-deploy-path', type=str, 
                        default='../datasets/drone_deploy_dataset_intermediate/dataset-medium')
    parser.add_argument('--udd-path', type=str, 
                        default='../datasets/UDD/UDD/UDD5')
    parser.add_argument('--semantic-drone-path', type=str, 
                        default='../datasets/Aerial_Semantic_Segmentation_Drone_Dataset/dataset/semantic_drone_dataset')
    
    # Training strategy
    parser.add_argument('--strategy', type=str, choices=['individual', 'reverse_progressive', 'both'], 
                        default='both', help='Training strategy to use')
    
    # Training parameters
    parser.add_argument('--max-epochs', type=int, default=20, help='Max epochs for individual training')
    parser.add_argument('--epochs-per-stage', type=int, default=15, help='Epochs per stage for progressive')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--base-lr', type=float, default=1e-4, help='Base learning rate')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/fixed_training')
    parser.add_argument('--cache-dir', type=str, default='cache/augmented_datasets')
    
    args = parser.parse_args()
    
    config = {
        'drone_deploy_path': args.drone_deploy_path,
        'udd_path': args.udd_path,
        'semantic_drone_path': args.semantic_drone_path,
        'strategy': args.strategy,
        'max_epochs': args.max_epochs,
        'epochs_per_stage': args.epochs_per_stage,
        'batch_size': args.batch_size,
        'base_lr': args.base_lr,
        'output_dir': args.output_dir,
        'cache_dir': args.cache_dir
    }
    
    trainer = FixedTrainer(config)
    
    if args.strategy == 'individual' or args.strategy == 'both':
        print("🎯 Running individual dataset training...")
        individual_results = trainer.run_individual_training()
        
        print("\n📊 INDIVIDUAL TRAINING RESULTS:")
        for dataset, result in individual_results.items():
            if 'best_iou' in result:
                print(f"   {dataset}: IoU = {result['best_iou']:.4f}")
            else:
                print(f"   {dataset}: FAILED - {result.get('error', 'Unknown error')}")
    
    if args.strategy == 'reverse_progressive' or args.strategy == 'both':
        print("\n🔄 Running reverse progressive training...")
        final_model = trainer.run_reverse_progressive()
    
    wandb.finish()
    print("\n🎉 Fixed training strategy completed!")


if __name__ == "__main__":
    main() 