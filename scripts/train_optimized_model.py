#!/usr/bin/env python3
"""
Optimized Training Script for Semantic Segmentation
==================================================

This script uses the optimized dataset with multi-scale support and
addresses all the data loading bottlenecks identified.

Key optimizations:
- Pre-processed data caching
- Multi-scale training (crops + context)
- Optimized DataLoader settings
- GPU memory optimization
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from pathlib import Path
from typing import Dict, Any
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets.optimized_semantic_dataset import OptimizedSemanticDroneDataset, create_optimized_transforms
from models.enhanced_architectures import create_enhanced_model
from losses.safety_aware_losses import CombinedSafetyLoss
from safety_evaluation.safety_metrics import SafetyAwareEvaluator

def print_header():
    """Prints a formatted header for the training script."""
    print("=" * 80)
    print("🚀 OPTIMIZED Semantic Segmentation Training")
    print("=" * 80)
    print("Features:")
    print("  • Pre-processed data caching for instant loading")
    print("  • Multi-scale: crops (detail) + full images (context)")
    print("  • GPU-optimized data pipeline")
    print("  • Faster augmentation pipeline")
    print("-" * 80)

class OptimizedTrainer:
    """
    Optimized trainer with fast data loading and multi-scale support.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize wandb
        wandb.init(
            project=config['wandb_project'],
            name=f"optimized-{config['model_type']}-{wandb.util.generate_id()}",
            config=config,
            tags=["optimized", "multi-scale", "fast-loading", config['model_type']],
            notes="Optimized training with pre-processed caching and multi-scale support."
        )

        self._log_system_info()

    def _log_system_info(self):
        """Logs system and configuration details."""
        print(f"Configuration:")
        for key, val in self.config.items():
            print(f"  - {key}: {val}")
        print("-" * 80)
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"   - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"   - CUDA Version: {torch.version.cuda}")
        else:
            print("WARNING: Running on CPU. Training will be extremely slow.")
        print("-" * 80)

    def _get_dataloaders(self) -> Dict[str, DataLoader]:
        """Creates optimized dataloaders with pre-processing."""
        print("🔄 Setting up optimized datasets...")
        
        train_transforms = create_optimized_transforms(
            input_size=self.config['input_size'], 
            is_training=True
        )
        val_transforms = create_optimized_transforms(
            input_size=self.config['input_size'], 
            is_training=False
        )

        # Training dataset with multi-scale support
        train_dataset = OptimizedSemanticDroneDataset(
            data_root=self.config['dataset_path'],
            split="train",
            transform=train_transforms,
            target_resolution=self.config['input_size'],
            use_multi_scale=self.config.get('use_multi_scale', True),
            crops_per_image=self.config.get('crops_per_image', 4),
            preprocess_on_init=True
        )
        
        # Validation dataset
        val_dataset = OptimizedSemanticDroneDataset(
            data_root=self.config['dataset_path'],
            split="val",
            transform=val_transforms,
            target_resolution=self.config['input_size'],
            use_multi_scale=False,  # Only full images for validation
            preprocess_on_init=True
        )

        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")

        # Optimized DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,  # Keeps workers alive between epochs
            prefetch_factor=4,       # Pre-fetch more batches
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        return {'train': train_loader, 'val': val_loader}

    def _get_model(self) -> nn.Module:
        """Creates and returns the segmentation model."""
        print(f"Creating model: {self.config['model_type']}")
        model = create_enhanced_model(
            model_type=self.config['model_type'],
            num_classes=4,
            in_channels=3,
            uncertainty_estimation=True,
            pretrained_path=self.config['pretrained_path']
        )
        return model.to(self.device)

    def train(self):
        """Runs the optimized training pipeline."""
        dataloaders = self._get_dataloaders()
        model = self._get_model()
        
        criterion = CombinedSafetyLoss(num_classes=4)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config['epochs'], eta_min=1e-6
        )

        scaler = GradScaler()
        best_iou = 0.0
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n🚀 Starting optimized training...\n")
        
        # Training loop with timing
        for epoch in range(self.config['epochs']):
            print(f"--- Epoch {epoch+1}/{self.config['epochs']} ---")
            
            # Time the training
            start_time = time.time()
            train_loss = self._train_epoch(model, dataloaders['train'], criterion, optimizer, scaler)
            train_time = time.time() - start_time
            
            # Time the validation
            start_time = time.time()
            val_loss, val_metrics = self._validate_epoch(model, dataloaders['val'], criterion)
            val_time = time.time() - start_time
            
            scheduler.step()

            # Logging with timing info
            current_lr = optimizer.param_groups[0]['lr']
            log_metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr,
                'train_time': train_time,
                'val_time': val_time,
                **{f"val/{k}": v for k, v in val_metrics.items()}
            }
            wandb.log(log_metrics)
            
            print(f"  - Train Loss: {train_loss:.4f} ({train_time:.1f}s)")
            print(f"  - Val Loss:   {val_loss:.4f} ({val_time:.1f}s)")
            print(f"  - Val mIoU:   {val_metrics.get('miou', 0.0):.4f}")
            print(f"  - Val Acc:    {val_metrics.get('accuracy', 0.0):.4f}")

            # Save best model
            if val_metrics.get('miou', 0.0) > best_iou:
                best_iou = val_metrics.get('miou', 0.0)
                save_path = output_dir / f"optimized_{self.config['model_type']}_best.pth"
                torch.save(model.state_dict(), save_path)
                print(f"   New best model saved to {save_path} (mIoU: {best_iou:.4f})")

        wandb.finish()
        print("\n🎉 Optimized training complete!")
        print(f"Best model saved in '{output_dir}'")

    def _train_epoch(self, model, dataloader, criterion, optimizer, scaler):
        """Trains the model for one epoch with timing."""
        model.train()
        total_loss = 0.0
        
        # Pre-fetch first batch to hide loading latency
        dataloader_iter = iter(dataloader)
        
        progress_bar = tqdm(dataloader_iter, desc="Training", leave=False)
        
        for batch in progress_bar:
            # Move to GPU efficiently
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss_dict = criterion(outputs, masks)
                loss = loss_dict['total_loss']
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        return total_loss / len(dataloader)

    def _validate_epoch(self, model, dataloader, criterion):
        """Validates the model for one epoch."""
        model.eval()
        total_loss = 0.0
        evaluator = SafetyAwareEvaluator(num_classes=4)
        
        progress_bar = tqdm(dataloader, desc="Validating", leave=False)

        with torch.no_grad():
            for batch in progress_bar:
                images = batch['image'].to(self.device, non_blocking=True)
                masks = batch['mask'].to(self.device, non_blocking=True)
                
                outputs = model(images)
                loss_dict = criterion(outputs, masks)
                loss = loss_dict['total_loss']
                
                total_loss += loss.item()
                
                logits = outputs['main'] if isinstance(outputs, dict) else outputs
                predictions = torch.argmax(logits, dim=1)
                evaluator.update(predictions, masks)
        
        metrics = evaluator.compute_metrics()
        return total_loss / len(dataloader), metrics

def main():
    parser = argparse.ArgumentParser(description="Optimized Semantic Segmentation Training")
    
    # Get the absolute path to the project root
    project_root = Path(__file__).parent.parent.parent
    default_dataset_path = project_root / 'datasets' / 'Aerial_Semantic_Segmentation_Drone_Dataset' / 'dataset' / 'semantic_drone_dataset'
    default_pretrained_path = project_root / 'model_pths' / 'bisenetv2_fcn_4x4_1024x1024_160k_cityscapes_20210902_015551-bcf10f09.pth'

    # Paths
    parser.add_argument('--dataset-path', type=str, 
                        default=str(default_dataset_path),
                        help='Path to the root of Semantic Drone Dataset.')
    parser.add_argument('--pretrained-path', type=str, 
                        default=str(default_pretrained_path),
                        help='Path to the pretrained Cityscapes weights.')
    parser.add_argument('--output-dir', type=str, default='outputs/optimized_training',
                        help='Directory to save model checkpoints.')

    # Model and Training Parameters
    parser.add_argument('--model-type', type=str, default='mmseg_bisenetv2',
                        choices=['mmseg_bisenetv2', 'deeplabv3plus_resnet50'],
                        help='The model architecture to use.')
    parser.add_argument('--input-size', type=int, nargs=2, default=[512, 512],
                        help='Input image size (height width).')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Total number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=16,  # Increased default
                        help='Training batch size.')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Initial learning rate.')
    parser.add_argument('--num-workers', type=int, default=8,  # Increased default
                        help='Number of workers for DataLoader.')

    # Optimization parameters
    parser.add_argument('--crops-per-image', type=int, default=4,
                        help='Number of pre-computed crops per training image.')
    parser.add_argument('--use-multi-scale', action='store_true', default=True,
                        help='Use multi-scale training (crops + context).')

    # W&B
    parser.add_argument('--wandb-project', type=str, default='uav-landing-optimized',
                        help='Weights & Biases project name.')

    args = parser.parse_args()
    config = vars(args)
    
    print_header()
    trainer = OptimizedTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main() 