#!/usr/bin/env python3
"""
Memory-Efficient Training Script for 8GB GPU
============================================

Optimized training for systems with limited GPU memory:
- Memory-efficient dataset
- Conservative batch sizes
- GPU memory monitoring
- Gradient accumulation for effective larger batches
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
import gc

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets.memory_efficient_dataset import MemoryEfficientSemanticDataset
from models.enhanced_architectures import create_enhanced_model
from losses.safety_aware_losses import CombinedSafetyLoss
from safety_evaluation.safety_metrics import SafetyAwareEvaluator

def print_header():
    """Prints a formatted header for the training script."""
    print("=" * 80)
    print("🧠 MEMORY-EFFICIENT Semantic Segmentation Training")
    print("=" * 80)
    print("Features:")
    print("  • Optimized for 8GB GPU memory")
    print("  • Memory-efficient data pipeline")
    print("  • Gradient accumulation for effective larger batches")
    print("  • GPU memory monitoring")
    print("-" * 80)

class MemoryEfficientTrainer:
    """
    Memory-efficient trainer for 8GB GPU systems.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize wandb
        wandb.init(
            project=config['wandb_project'],
            name=f"memory-efficient-{config['model_type']}-{wandb.util.generate_id()}",
            config=config,
            tags=["memory-efficient", "8gb-gpu", config['model_type']],
            notes="Memory-efficient training for 8GB GPU systems."
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
            self._log_gpu_memory()
        else:
            print("WARNING: Running on CPU.")
        print("-" * 80)

    def _log_gpu_memory(self):
        """Log current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"   - Memory allocated: {allocated:.2f} GB")
            print(f"   - Memory reserved: {reserved:.2f} GB")

    def _get_dataloaders(self) -> Dict[str, DataLoader]:
        """Creates memory-efficient dataloaders."""
        print("🧠 Setting up memory-efficient datasets...")
        
        # Training dataset
        train_dataset = MemoryEfficientSemanticDataset(
            data_root=self.config['dataset_path'],
            split="train",
            target_resolution=self.config['input_size'],
            crops_per_image=self.config.get('crops_per_image', 4),
            max_memory_gb=self.config.get('max_memory_gb', 6.0),
            preprocess_on_init=True
        )
        
        # Validation dataset
        val_dataset = MemoryEfficientSemanticDataset(
            data_root=self.config['dataset_path'],
            split="val",
            target_resolution=self.config['input_size'],
            crops_per_image=1,  # Only 1 variant for validation
            max_memory_gb=self.config.get('max_memory_gb', 6.0),
            preprocess_on_init=True
        )

        print(f"  ✅ Training samples: {len(train_dataset)}")
        print(f"  ✅ Validation samples: {len(val_dataset)}")

        # Conservative DataLoaders for 8GB GPU
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,  # Reduced workers for memory
            pin_memory=False,  # Disable for memory efficiency
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,  # Reduced workers
            pin_memory=False,  # Disable for memory efficiency
        )
        return {'train': train_loader, 'val': val_loader}

    def _get_model(self) -> nn.Module:
        """Creates and returns the segmentation model."""
        print(f"Creating model: {self.config['model_type']}")
        model = create_enhanced_model(
            model_type=self.config['model_type'],
            num_classes=4,
            in_channels=3,
            uncertainty_estimation=False,  # Disable for memory efficiency
            pretrained_path=self.config['pretrained_path']
        )
        return model.to(self.device)

    def train(self):
        """Runs the memory-efficient training pipeline."""
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

        print("\n🧠 Starting memory-efficient training...\n")
        
        # Training loop with memory management
        for epoch in range(self.config['epochs']):
            print(f"--- Epoch {epoch+1}/{self.config['epochs']} ---")
            
            # Aggressive memory cleanup
            torch.cuda.empty_cache()
            gc.collect()
            
            # Time the training
            start_time = time.time()
            train_loss, train_throughput = self._train_epoch(model, dataloaders['train'], criterion, optimizer, scaler)
            train_time = time.time() - start_time
            
            # Memory cleanup before validation
            torch.cuda.empty_cache()
            gc.collect()
            
            # Time the validation
            start_time = time.time()
            val_loss, val_metrics, val_throughput = self._validate_epoch(model, dataloaders['val'], criterion)
            val_time = time.time() - start_time
            
            scheduler.step()

            # Logging
            current_lr = optimizer.param_groups[0]['lr']
            log_metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr,
                'train_time': train_time,
                'val_time': val_time,
                'train_throughput': train_throughput,
                'val_throughput': val_throughput,
                **{f"val/{k}": v for k, v in val_metrics.items()}
            }
            wandb.log(log_metrics)
            
            print(f"  - Train Loss: {train_loss:.4f} ({train_time:.1f}s, {train_throughput:.1f} samples/s)")
            print(f"  - Val Loss:   {val_loss:.4f} ({val_time:.1f}s, {val_throughput:.1f} samples/s)")
            print(f"  - Val mIoU:   {val_metrics.get('miou', 0.0):.4f}")
            print(f"  - Val Acc:    {val_metrics.get('accuracy', 0.0):.4f}")
            
            # Memory status
            self._log_gpu_memory()

            # Save best model
            if val_metrics.get('miou', 0.0) > best_iou:
                best_iou = val_metrics.get('miou', 0.0)
                save_path = output_dir / f"memory_efficient_{self.config['model_type']}_best.pth"
                torch.save(model.state_dict(), save_path)
                print(f"  ✅ New best model saved to {save_path} (mIoU: {best_iou:.4f})")

        wandb.finish()
        print("\n🧠 Memory-efficient training complete!")
        print(f"Best model saved in '{output_dir}'")

    def _train_epoch(self, model, dataloader, criterion, optimizer, scaler):
        """Trains the model for one epoch with memory management."""
        model.train()
        total_loss = 0.0
        total_samples = 0
        start_time = time.time()
        
        # Gradient accumulation for effective larger batch size
        accumulation_steps = self.config.get('accumulation_steps', 2)
        
        progress_bar = tqdm(dataloader, desc="Training", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to GPU efficiently
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)
            
            batch_size = images.size(0)
            total_samples += batch_size
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss_dict = criterion(outputs, masks)
                loss = loss_dict['total_loss'] / accumulation_steps  # Scale loss for accumulation
            
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps  # Unscale for logging
            progress_bar.set_postfix(loss=f"{loss.item() * accumulation_steps:.4f}")
            
            # Periodic memory cleanup
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()
        
        # Handle remaining gradients
        if len(dataloader) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_time = time.time() - start_time
        throughput = total_samples / total_time
        
        return total_loss / len(dataloader), throughput

    def _validate_epoch(self, model, dataloader, criterion):
        """Validates the model for one epoch."""
        model.eval()
        total_loss = 0.0
        total_samples = 0
        evaluator = SafetyAwareEvaluator(num_classes=4)
        start_time = time.time()
        
        progress_bar = tqdm(dataloader, desc="Validating", leave=False)

        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                images = batch['image'].to(self.device, non_blocking=True)
                masks = batch['mask'].to(self.device, non_blocking=True)
                
                batch_size = images.size(0)
                total_samples += batch_size
                
                outputs = model(images)
                loss_dict = criterion(outputs, masks)
                loss = loss_dict['total_loss']
                
                total_loss += loss.item()
                
                logits = outputs['main'] if isinstance(outputs, dict) else outputs
                predictions = torch.argmax(logits, dim=1)
                evaluator.update(predictions, masks)
                
                # Periodic memory cleanup
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
        
        total_time = time.time() - start_time
        throughput = total_samples / total_time
        
        metrics = evaluator.compute_metrics()
        return total_loss / len(dataloader), metrics, throughput

def main():
    parser = argparse.ArgumentParser(description="Memory-Efficient Semantic Segmentation Training")
    
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
    parser.add_argument('--output-dir', type=str, default='outputs/memory_efficient_training',
                        help='Directory to save model checkpoints.')

    # Model and Training Parameters (Conservative for 8GB)
    parser.add_argument('--model-type', type=str, default='mmseg_bisenetv2',
                        choices=['mmseg_bisenetv2', 'deeplabv3plus_resnet50'],
                        help='The model architecture to use.')
    parser.add_argument('--input-size', type=int, nargs=2, default=[512, 512],
                        help='Input image size (height width).')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Total number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=4,  # Conservative for 8GB
                        help='Training batch size.')
    parser.add_argument('--accumulation-steps', type=int, default=4,  # Effective batch size = 4*4=16
                        help='Gradient accumulation steps.')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Initial learning rate.')

    # Memory optimization parameters
    parser.add_argument('--crops-per-image', type=int, default=4,
                        help='Number of crops per training image.')
    parser.add_argument('--max-memory-gb', type=float, default=5.0,  # Conservative for 8GB GPU
                        help='Maximum memory to use for dataset caching.')

    # W&B
    parser.add_argument('--wandb-project', type=str, default='uav-landing-memory-efficient',
                        help='Weights & Biases project name.')

    args = parser.parse_args()
    config = vars(args)
    
    print_header()
    trainer = MemoryEfficientTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main() 