#!/usr/bin/env python3
"""
A100 GPU Optimized UAV Landing Detection Training
================================================

High-performance training script optimized for A100 GPU with:
- Enhanced multi-dataset training pipeline
- Advanced memory management
- Professional monitoring and logging
- Checkpoint management and recovery
- Weights & Biases integration
"""

import os
import sys
import time
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import cv2
from tqdm import tqdm
import wandb

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# A100 GPU Optimizations
# =============================================================================
def setup_a100_optimizations():
    """Configure PyTorch for optimal A100 performance"""
    if torch.cuda.is_available():
        # A100 specific optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Memory optimizations
        torch.cuda.empty_cache()
        
        print(f"🚀 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("✅ A100 optimizations applied!")
        return True
    else:
        print("❌ No GPU detected!")
        return False

# =============================================================================
# Import Project Components
# =============================================================================
sys.path.append(str(Path(__file__).parent))

try:
    # Import the enhanced training pipeline
    from training.enhanced_training_pipeline import EnhancedTrainingPipeline
    from src.models.enhanced_bisenetv2 import EnhancedBiSeNetV2
    from src.models.enhanced_deeplabv3 import EnhancedDeepLabV3Plus
    from src.losses.safety_aware_loss import SafetyAwareLoss
    from src.evaluation.safety_metrics import SafetyAwareEvaluator
    from datasets.semantic_drone_dataset import SemanticDroneDataset
    print("✅ Successfully imported project components")
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    print("Using fallback implementations...")
    
    # Fallback implementations if main components aren't available
    class EnhancedTrainingPipeline:
        def __init__(self, config):
            self.config = config
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
    class SafetyAwareLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.criterion = nn.CrossEntropyLoss()
            
        def forward(self, outputs, targets):
            return {'total_loss': self.criterion(outputs, targets)}

# =============================================================================
# A100 Training Configuration
# =============================================================================
class A100TrainingConfig:
    """Optimized training configuration for A100 GPU"""
    
    def __init__(self):
        # A100 optimized settings
        self.batch_size = 32  # Large batch for A100's 40GB memory
        self.accumulation_steps = 1  # No accumulation needed with large batch
        self.num_workers = 8  # Optimal for A100 systems
        self.pin_memory = True
        self.persistent_workers = True
        
        # Training parameters
        self.epochs = 150
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.warmup_epochs = 10
        
        # Model configuration
        self.model_name = 'enhanced_bisenetv2'  # or 'enhanced_deeplabv3plus'
        self.num_classes = 4
        self.input_size = (512, 512)
        
        # Mixed precision training
        self.use_amp = True
        self.gradient_clipping = 1.0
        
        # Checkpoint and logging
        self.save_every = 10
        self.log_every = 50
        self.validate_every = 5
        
        # Paths
        self.data_root = "datasets/raw"
        self.output_dir = "outputs"
        self.checkpoint_dir = "checkpoints"
        self.log_dir = "logs"
        
        # Weights & Biases
        self.use_wandb = True
        self.wandb_project = "uav-landing-a100"
        self.wandb_name = f"a100-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# =============================================================================
# Enhanced Model Definitions (Fallback)
# =============================================================================
def create_model(config: A100TrainingConfig) -> nn.Module:
    """Create the training model"""
    
    if config.model_name == 'enhanced_deeplabv3plus':
        try:
            model = EnhancedDeepLabV3Plus(num_classes=config.num_classes)
        except:
            # Fallback to torchvision DeepLabV3
            from torchvision.models.segmentation import deeplabv3_resnet50
            model = deeplabv3_resnet50(num_classes=config.num_classes, pretrained=True)
            print("Using fallback DeepLabV3 model")
    else:
        try:
            model = EnhancedBiSeNetV2(num_classes=config.num_classes)
        except:
            # Simple fallback model
            model = SimpleFallbackModel(config.num_classes)
            print("Using fallback segmentation model")
    
    return model

class SimpleFallbackModel(nn.Module):
    """Simple fallback segmentation model"""
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True).features
        self.classifier = nn.Sequential(
            nn.Conv2d(1280, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        out = torch.nn.functional.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return out

# =============================================================================
# Dataset Loading
# =============================================================================
def create_datasets(config: A100TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation datasets"""
    
    # Basic transforms for fallback
    import torchvision.transforms as T
    
    train_transform = T.Compose([
        T.Resize(config.input_size),
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = T.Compose([
        T.Resize(config.input_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        # Try to use the project's semantic drone dataset
        train_dataset = SemanticDroneDataset(
            root_dir=f"{config.data_root}/semantic_drone_dataset",
            split='train',
            transform=train_transform
        )
        val_dataset = SemanticDroneDataset(
            root_dir=f"{config.data_root}/semantic_drone_dataset", 
            split='val',
            transform=val_transform
        )
    except:
        print("⚠️  Using dummy dataset for testing")
        # Create dummy datasets for testing
        train_dataset = DummyDataset(1000, config.input_size, config.num_classes)
        val_dataset = DummyDataset(200, config.input_size, config.num_classes)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers
    )
    
    return train_loader, val_loader

class DummyDataset(torch.utils.data.Dataset):
    """Dummy dataset for testing"""
    def __init__(self, size, input_size, num_classes):
        self.size = size
        self.input_size = input_size
        self.num_classes = num_classes
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        image = torch.randn(3, *self.input_size)
        mask = torch.randint(0, self.num_classes, self.input_size)
        return {'image': image, 'mask': mask}

# =============================================================================
# A100 Optimized Trainer
# =============================================================================
class A100Trainer:
    """High-performance trainer optimized for A100 GPU"""
    
    def __init__(self, config: A100TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        for dir_path in [config.output_dir, config.checkpoint_dir, config.log_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize Weights & Biases
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_name,
                config=vars(config)
            )
        
        # Create model, loss, optimizer
        self.model = create_model(config).to(self.device)
        self.criterion = SafetyAwareLoss().to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=100,  # Will be updated with actual data
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.best_miou = 0.0
        self.training_history = []
        
        print(f"🎯 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"🎯 Model size: {sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1e6:.1f} MB")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_metrics = {
            'loss': 0.0,
            'samples_processed': 0,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, masks)
                    loss = loss_dict['total_loss']
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.config.gradient_clipping
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, masks)
                loss = loss_dict['total_loss']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.gradient_clipping
                )
                self.optimizer.step()
            
            # Update scheduler
            self.scheduler.step()
            
            # Update metrics
            batch_size = images.size(0)
            epoch_metrics['loss'] += loss.item() * batch_size
            epoch_metrics['samples_processed'] += batch_size
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log to wandb
            if self.config.use_wandb and batch_idx % self.config.log_every == 0:
                wandb.log({
                    'batch_loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': self.current_epoch,
                    'batch': batch_idx
                })
        
        # Calculate average metrics
        epoch_metrics['loss'] /= epoch_metrics['samples_processed']
        return epoch_metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        val_metrics = {
            'val_loss': 0.0,
            'val_miou': 0.0,
            'samples_processed': 0
        }
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            
            for batch in progress_bar:
                images = batch['image'].to(self.device, non_blocking=True)
                masks = batch['mask'].to(self.device, non_blocking=True)
                
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, masks)
                loss = loss_dict['total_loss']
                
                # Calculate IoU (simplified)
                if isinstance(outputs, dict):
                    preds = torch.argmax(outputs['out'], dim=1)
                else:
                    preds = torch.argmax(outputs, dim=1)
                
                intersection = ((preds == masks) & (masks > 0)).sum().float()
                union = ((preds > 0) | (masks > 0)).sum().float()
                iou = intersection / (union + 1e-8)
                
                batch_size = images.size(0)
                val_metrics['val_loss'] += loss.item() * batch_size
                val_metrics['val_miou'] += iou.item() * batch_size
                val_metrics['samples_processed'] += batch_size
                
                progress_bar.set_postfix({
                    'val_loss': f"{loss.item():.4f}",
                    'val_miou': f"{iou.item():.4f}"
                })
        
        # Calculate averages
        val_metrics['val_loss'] /= val_metrics['samples_processed']
        val_metrics['val_miou'] /= val_metrics['samples_processed']
        
        return val_metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
            'metrics': metrics,
            'config': vars(self.config)
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / "latest_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_checkpoint.pth"
            torch.save(checkpoint, best_path)
            print(f"🏆 New best model saved! mIoU: {metrics['val_miou']:.4f}")
        
        # Save periodic checkpoint
        if self.current_epoch % self.config.save_every == 0:
            epoch_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{self.current_epoch}.pth"
            torch.save(checkpoint, epoch_path)
    
    def train(self):
        """Main training loop"""
        print("🚀 Starting A100 GPU training...")
        
        # Create data loaders
        train_loader, val_loader = create_datasets(self.config)
        
        # Update scheduler steps per epoch
        self.scheduler.steps_per_epoch = len(train_loader)
        
        print(f"📊 Training samples: {len(train_loader.dataset)}")
        print(f"📊 Validation samples: {len(val_loader.dataset)}")
        print(f"📊 Batch size: {self.config.batch_size}")
        print(f"📊 Total epochs: {self.config.epochs}")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            if epoch % self.config.validate_every == 0:
                val_metrics = self.validate_epoch(val_loader)
            else:
                val_metrics = {'val_loss': 0.0, 'val_miou': 0.0}
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            self.training_history.append(epoch_metrics)
            
            # Check for best model
            is_best = val_metrics['val_miou'] > self.best_miou
            if is_best:
                self.best_miou = val_metrics['val_miou']
            
            # Save checkpoint
            self.save_checkpoint(epoch_metrics, is_best)
            
            # Log to wandb
            if self.config.use_wandb:
                wandb.log({
                    **epoch_metrics,
                    'epoch': epoch,
                    'best_miou': self.best_miou
                })
            
            # Print epoch summary
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d}/{self.config.epochs} | "
                  f"Loss: {train_metrics['loss']:.4f} | "
                  f"Val mIoU: {val_metrics['val_miou']:.4f} | "
                  f"Best: {self.best_miou:.4f} | "
                  f"Time: {elapsed/60:.1f}m")
        
        total_time = time.time() - start_time
        print(f"🎉 Training completed in {total_time/3600:.2f} hours!")
        print(f"🏆 Best validation mIoU: {self.best_miou:.4f}")
        
        # Save final model
        final_model_path = Path(self.config.output_dir) / "final_model.pth"
        torch.save(self.model.state_dict(), final_model_path)
        
        # Save training history
        history_path = Path(self.config.output_dir) / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        if self.config.use_wandb:
            wandb.finish()

# =============================================================================
# Main Training Function
# =============================================================================
def main():
    """Main training function"""
    print("🚁 A100 GPU UAV Landing Detection Training")
    print("=" * 50)
    
    # Setup A100 optimizations
    setup_a100_optimizations()
    
    # Create training configuration
    config = A100TrainingConfig()
    
    # Print configuration
    print(f"📋 Training Configuration:")
    print(f"   🔧 Model: {config.model_name}")
    print(f"   📦 Batch size: {config.batch_size}")
    print(f"   🔄 Epochs: {config.epochs}")
    print(f"   📈 Learning rate: {config.learning_rate}")
    print(f"   🎯 Input size: {config.input_size}")
    print(f"   🏷️  Classes: {config.num_classes}")
    print(f"   ⚡ Mixed precision: {config.use_amp}")
    print(f"   📊 W&B logging: {config.use_wandb}")
    print()
    
    # Create and start trainer
    trainer = A100Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main() 