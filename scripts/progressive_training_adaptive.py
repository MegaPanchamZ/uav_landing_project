#!/usr/bin/env python3
"""
Progressive Training with Adaptive Size Handling
===============================================

Modified progressive training that can handle the existing mixed-size cached dataset
using adaptive collation to resize all patches to a common size during training.
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
from datasets.cached_augmentation import CachedAugmentedDataset
from models.enhanced_architectures import create_enhanced_model
from losses.safety_aware_losses import CombinedSafetyLoss
from safety_evaluation.safety_metrics import SafetyAwareEvaluator
from training.variable_size_training import create_variable_size_dataloader
from torch.utils.data import DataLoader


def adaptive_collate_fn(batch):
    """Custom collate function that resizes all patches to 512x512."""
    import torch.nn.functional as F
    
    images = [item['image'] for item in batch]
    masks = [item['mask'] for item in batch]
    
    target_size = (512, 512)  # Fixed target size
    
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


class AdaptiveProgressiveTrainer:
    """Progressive trainer that handles mixed-size cached datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Enhanced device info
        if torch.cuda.is_available():
            print(f"🚀 GPU Available: {torch.cuda.get_device_name()}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  Running on CPU - training will be very slow!")
            print("💡 Consider enabling GPU for 50-100x speedup")
        
        # Initialize wandb with better configuration
        wandb.init(
            project="uav-landing-adaptive",
            name=f"adaptive_training_{wandb.util.generate_id()}",
            config=config,
            tags=["progressive", "transfer-learning", "gpu"],
            notes="Progressive fine-tuning with Cityscapes pretrained weights"
        )
        
        print("🔄 Adaptive Progressive UAV Landing Training")
        print("=" * 50)
        print(f"📊 Strategy: DroneDeploy → UDD5 → Aerial Semantic (Adaptive Size)")
        print(f"🏗️ Model: BiSeNetV2 with adaptive size handling")
        print(f"💾 Device: {self.device}")
        print(f"📁 Output: {config['output_dir']}")
        
    def create_stage_dataset(self, stage_num: int, dataset_name: str, dataset_path: str, dataset_class) -> Dict[str, DataLoader]:
        """Create dataset for a specific stage with adaptive size handling."""
        print(f"\n🏗️ Stage {stage_num}: {dataset_name} (Adaptive Size)")
        
        # Base datasets with appropriate parameters for each dataset type
        if dataset_name == 'DroneDeploy':
            train_dataset = dataset_class(
                data_root=dataset_path,
                split='train',
                transform=None,  # Raw data for caching
                use_height=True  # DroneDeploy uses height maps
            )
            val_dataset = dataset_class(
                data_root=dataset_path,
                split='val',
                transform=create_drone_deploy_transforms(is_training=False),
                use_height=True
            )
        else:
            # UDD and Semantic Drone datasets don't use height parameter
            train_dataset = dataset_class(
                data_root=dataset_path,
                split='train',
                transform=None  # Raw data for caching
            )
            val_dataset = dataset_class(
                data_root=dataset_path,
                split='val',
                transform=create_udd_transforms(is_training=False) if dataset_name == 'UDD'
                          else create_semantic_drone_transforms(is_training=False)
            )
        
        # Load cached dataset (with mixed sizes)
        print(f"   💾 Loading cached dataset with adaptive resizing...")
        # Map dataset names to match existing cache
        cache_name_mapping = {
            'DroneDeploy': 'drone_deploy',
            'UDD': 'udd', 
            'Semantic Drone': 'semantic_drone'
        }
        
        # Use EXACT parameters from existing cache to avoid rebuilding
        if dataset_name == 'DroneDeploy':
            cache_params = {
                'patch_scales': [(512, 512), (768, 768)],
                'augmentation_factor': 20,
                'min_object_ratio': 0.1,
                'overlap_ratio': 0.3
            }
        elif dataset_name == 'UDD':
            cache_params = {
                'patch_scales': [(512, 512)],
                'augmentation_factor': 15,
                'min_object_ratio': 0.15,
                'overlap_ratio': 0.2
            }
        else:  # Semantic Drone
            cache_params = {
                'patch_scales': [(512, 512), (768, 768), (1024, 1024)],
                'augmentation_factor': 25,
                'min_object_ratio': 0.05,
                'overlap_ratio': 0.25
            }
        
        train_dataset = CachedAugmentedDataset(
            base_dataset=train_dataset,
            cache_dir=self.config.get('cache_dir', 'cache/augmented_datasets'),
            dataset_name=cache_name_mapping[dataset_name],
            # Use EXACT cache parameters to match existing cache
            patch_scales=cache_params['patch_scales'],
            augmentation_factor=cache_params['augmentation_factor'],
            min_object_ratio=cache_params['min_object_ratio'],
            use_overlapping=True,
            overlap_ratio=cache_params['overlap_ratio'],
            uav_augmentations=True,
            force_rebuild=False
        )
        
        # Create DataLoaders with adaptive collation
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4 if torch.cuda.is_available() else 0,  # More workers for GPU
            pin_memory=torch.cuda.is_available(),  # Pin memory for GPU
            collate_fn=adaptive_collate_fn,  # Custom collation for size handling
            drop_last=True  # Prevent BatchNorm errors with batch_size=1
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available(),
            drop_last=True  # Prevent BatchNorm errors with batch_size=1
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'in_channels': 4 if dataset_name == 'DroneDeploy' else 3,
            'stage_name': dataset_name.lower().replace(' ', '_')
        }
    
    def train_stage(self, stage_data: Dict, stage_num: int, pretrained_model=None):
        """Train a single stage."""
        print(f"\n🚀 Starting Stage {stage_num}: {stage_data['stage_name']}")
        print(f"   Input channels: {stage_data['in_channels']}")
        print(f"   Epochs: {self.config['epochs_per_stage']}")
        
        # Create or adapt model
        if pretrained_model is None:
            # First stage - create new model
            print(f"🏗️ Creating model for {stage_data['stage_name']} (in_channels={stage_data['in_channels']})")
            model = create_enhanced_model(
                model_type="mmseg_bisenetv2",
                num_classes=4,
                in_channels=stage_data['in_channels'],
                uncertainty_estimation=True,
                pretrained_path="../model_pths/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes_20210902_015551-bcf10f09.pth"
            )
        else:
            # Subsequent stages - adapt existing model
            print(f"🔄 Adapting model for {stage_data['stage_name']} (in_channels={stage_data['in_channels']})")
            # For stages 2+ (UDD, Semantic), we need to adapt from 4-channel to 3-channel
            if stage_data['in_channels'] == 3:
                # Need to adapt input channels from 4-channel (DroneDeploy) to 3-channel
                model = self._adapt_model_channels(pretrained_model, stage_data['in_channels'])
            else:
                model = pretrained_model
        
        model = model.to(self.device)
        
        # Loss and optimizer
        criterion = CombinedSafetyLoss(num_classes=4)
        
        # Learning rate decreases with each stage (more conservative for pretrained models)
        lr = self.config['base_lr'] * (0.5 ** (stage_num - 1))
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Add learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.config['epochs_per_stage'],
            eta_min=lr * 0.01
        )
        
        print(f"   📈 Learning rate: {lr:.2e}")
        
        # Training loop
        best_iou = 0.0
        for epoch in range(self.config['epochs_per_stage']):
            print(f"\n--- Stage {stage_num}, Epoch {epoch+1}/{self.config['epochs_per_stage']} ---")
            
            # Train
            train_loss = self.train_epoch(model, stage_data['train'], criterion, optimizer)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(model, stage_data['val'], criterion)
            
            # Update learning rate scheduler
            scheduler.step()
            
            # Log metrics
            wandb.log({
                f'stage_{stage_num}/train_loss': train_loss,
                f'stage_{stage_num}/val_loss': val_loss,
                f'stage_{stage_num}/val_iou': val_metrics.get('miou', 0.0),
                f'stage_{stage_num}/lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch + (stage_num - 1) * self.config['epochs_per_stage']
            })
            
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Val IoU: {val_metrics.get('miou', 0.0):.4f}")
            print(f"   Available metrics: {list(val_metrics.keys())}")
            
            # Save best model
            if val_metrics.get('miou', 0.0) > best_iou:
                best_iou = val_metrics.get('miou', 0.0)
                self.save_model(model, stage_num, stage_data['stage_name'], epoch, val_metrics)
        
        return model
    
    def _adapt_model_channels(self, model, new_in_channels):
        """Adapt model for different input channels."""
        # This is a simplified adaptation - in practice you'd want more sophisticated handling
        if new_in_channels == 3 and model.backbone.backbone.stem.conv_3x3.in_channels == 4:
            # Adapt from 4-channel to 3-channel by removing height channel adaptation
            print("   🔄 Adapting 4-channel model to 3-channel")
            # Copy first 3 channels of the first conv layer weights
            with torch.no_grad():
                old_weight = model.backbone.backbone.stem.conv_3x3.weight
                new_weight = old_weight[:, :3, :, :].clone()
                model.backbone.backbone.stem.conv_3x3 = nn.Conv2d(3, old_weight.shape[0], 
                                                                  kernel_size=3, stride=2, padding=1, bias=False)
                model.backbone.backbone.stem.conv_3x3.weight.data = new_weight
        
        return model
    
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
            loss = loss_dict['total_loss']  # Extract total loss from dictionary
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
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
                loss = loss_dict['total_loss']  # Extract total loss from dictionary
                
                total_loss += loss.item()
                
                # Update evaluator with class predictions (not logits)
                if isinstance(outputs, dict):
                    logits = outputs['main']
                else:
                    logits = outputs
                
                # Convert logits to class predictions
                predictions = torch.argmax(logits, dim=1)
                evaluator.update(predictions, masks)
        
        metrics = evaluator.compute_metrics()
        return total_loss / len(dataloader), metrics
    
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
        
        save_path = output_dir / f"adaptive_stage_{stage_num}_{stage_name}_best.pth"
        torch.save(checkpoint, save_path)
        print(f"   💾 Saved: {save_path}")
    
    def _adapt_model_channels(self, pretrained_model, new_in_channels):
        """Adapt model input channels (e.g., 4-channel to 3-channel)."""
        print(f"   🔧 Adapting input channels to {new_in_channels}")
        
        # For now, create a new model with the correct input channels
        # In a full implementation, you'd want to copy weights and adapt the first layer
        new_model = create_enhanced_model(
            model_type="mmseg_bisenetv2",
            num_classes=4,
            in_channels=new_in_channels,
            uncertainty_estimation=True,
            pretrained_path=None  # Don't reload pretrained, use the adapted model
        )
        
        # Copy all weights except the first layer
        new_state_dict = new_model.state_dict()
        old_state_dict = pretrained_model.state_dict()
        
        for name, param in old_state_dict.items():
            if name in new_state_dict and 'conv_3x3' not in name:
                # Copy all layers except input convolution
                if new_state_dict[name].shape == param.shape:
                    new_state_dict[name] = param
        
        new_model.load_state_dict(new_state_dict, strict=False)
        return new_model
    
    def run_progressive_training(self):
        """Run the complete progressive training pipeline."""
        print("\n🚀 Starting Adaptive Progressive Training Pipeline")
        
        # Stage 1: DroneDeploy
        stage1_data = self.create_stage_dataset(1, "DroneDeploy", self.config['drone_deploy_path'], DroneDeployDataset)
        model = self.train_stage(stage1_data, stage_num=1)
        
        # Stage 2: UDD
        stage2_data = self.create_stage_dataset(2, "UDD", self.config['udd_path'], UDDDataset)
        model = self.train_stage(stage2_data, stage_num=2, pretrained_model=model)
        
        # Stage 3: Semantic Drone
        stage3_data = self.create_stage_dataset(3, "Semantic Drone", self.config['semantic_drone_path'], SemanticDroneDataset)
        model = self.train_stage(stage3_data, stage_num=3, pretrained_model=model)
        
        print("\n Adaptive Progressive Training Complete!")
        print("Models saved in:", self.config['output_dir'])
        
        wandb.finish()
        return model


def main():
    parser = argparse.ArgumentParser(description="Adaptive Progressive Training")
    
    # Dataset paths
    parser.add_argument('--drone-deploy-path', type=str, 
                        default='../datasets/drone_deploy_dataset_intermediate/dataset-medium',
                        help='Path to DroneDeploy dataset')
    parser.add_argument('--udd-path', type=str, 
                        default='../datasets/UDD/UDD/UDD5',
                        help='Path to UDD5 dataset')
    parser.add_argument('--semantic-drone-path', type=str, 
                        default='../datasets/Aerial_Semantic_Segmentation_Drone_Dataset/dataset/semantic_drone_dataset',
                        help='Path to Semantic Drone dataset')
    
    # Training parameters
    parser.add_argument('--epochs-per-stage', type=int, default=5,
                        help='Epochs per training stage')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--base-lr', type=float, default=1e-3,
                        help='Base learning rate')
    
    # Cache parameters
    parser.add_argument('--cache-dir', type=str, default='cache/augmented_datasets',
                        help='Directory containing cached datasets')
    parser.add_argument('--augmentation-factor', type=int, default=20,
                        help='Augmentation factor')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/adaptive_training',
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
        'cache_dir': args.cache_dir,
        'augmentation_factor': args.augmentation_factor
    }
    
    # Create trainer and run
    trainer = AdaptiveProgressiveTrainer(config)
    final_model = trainer.run_progressive_training()
    
    print("🎉 Adaptive progressive training completed successfully!")


if __name__ == "__main__":
    main() 