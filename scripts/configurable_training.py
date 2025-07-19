#!/usr/bin/env python3
"""
Configurable UAV Landing Model Training Script
==============================================

Advanced training pipeline supporting:
- Multiple input resolutions (256x256 to 1024x1024)
- Configurable training parameters
- Multiple use cases (racing, commercial, precision, research)
- Automatic model architecture adaptation
- Dynamic loss monitoring and checkpoints
- Early stopping and learning rate scheduling

Author: UAV Landing Detection Team
Version: 2.0.0
Date: 2025-07-20
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
import time
import json
import argparse
from tqdm import tqdm
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any

# Try to import training components
try:
    from staged_training import (
        DroneDeployDataset, UDDDataset, STAGE1_CLASSES, STAGE2_CLASSES,
        DRONE_DEPLOY_MAPPING, UDD6_MAPPING, UDD_TO_LANDING
    )
    TRAINING_AVAILABLE = True
except ImportError:
    print("⚠️  Training datasets not available, running in config-only mode")
    TRAINING_AVAILABLE = False

class ConfigurableBiSeNet(nn.Module):
    """
    Configurable BiSeNet architecture that adapts based on input resolution and use case.
    
    Architecture scales from ultra-lightweight (256x256, racing) to high-capacity (1024x1024, research).
    """
    
    def __init__(self, num_classes: int = 4, input_resolution: Tuple[int, int] = (512, 512), 
                 use_case: str = "commercial"):
        super().__init__()
        
        self.input_resolution = input_resolution
        self.use_case = use_case
        
        # Determine architecture complexity based on resolution and use case
        complexity = self._determine_complexity()
        
        # Adaptive channel configuration
        if complexity == "ultra_light":
            channels = [32, 64, 128, 64, 32]
            self.dropout_rate = 0.1
        elif complexity == "light":
            channels = [64, 128, 256, 128, 64]
            self.dropout_rate = 0.2
        elif complexity == "medium":
            channels = [96, 192, 384, 192, 96]
            self.dropout_rate = 0.3
        elif complexity == "heavy":
            channels = [128, 256, 512, 256, 128]
            self.dropout_rate = 0.4
        else:  # ultra_heavy
            channels = [160, 320, 640, 320, 160]
            self.dropout_rate = 0.5
        
        # Backbone encoder
        self.backbone = nn.Sequential(
            # Initial feature extraction
            nn.Conv2d(3, channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            
            # Downsample 1
            nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout_rate),
            
            # Downsample 2
            nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            
            # Deep feature processing (more layers for higher resolutions)
            *self._create_feature_layers(channels[2], complexity),
        )
        
        # Feature fusion and decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout_rate),
            
            nn.Conv2d(channels[3], channels[4], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[4]),
            nn.ReLU(inplace=True),
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Conv2d(channels[4], channels[4] // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[4] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[4] // 2, num_classes, 1)
        )
        
        # Calculate model size for logging
        self.param_count = sum(p.numel() for p in self.parameters())
        
    def _determine_complexity(self) -> str:
        """Determine model complexity based on resolution and use case"""
        res = self.input_resolution[0]  # Assume square
        
        if self.use_case == "racing":
            if res <= 256:
                return "ultra_light"
            elif res <= 512:
                return "light"
            else:
                return "medium"
        elif self.use_case == "commercial":
            if res <= 256:
                return "light"
            elif res <= 512:
                return "medium"
            elif res <= 768:
                return "heavy"
            else:
                return "ultra_heavy"
        elif self.use_case == "precision":
            if res <= 256:
                return "medium"
            elif res <= 512:
                return "heavy"
            else:
                return "ultra_heavy"
        else:  # research
            if res <= 512:
                return "heavy"
            else:
                return "ultra_heavy"
    
    def _create_feature_layers(self, channels: int, complexity: str) -> List[nn.Module]:
        """Create additional feature processing layers based on complexity"""
        layers = []
        
        if complexity in ["ultra_light", "light"]:
            # Minimal processing for speed
            layers.extend([
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ])
        elif complexity == "medium":
            # Moderate processing
            layers.extend([
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ])
        else:  # heavy, ultra_heavy
            # Deep processing for quality
            layers.extend([
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 1, bias=False),  # 1x1 conv for feature mixing
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ])
            
            # Extra depth for ultra_heavy
            if complexity == "ultra_heavy":
                layers.extend([
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                ])
        
        return layers
    
    def forward(self, x):
        # Encode
        features = self.backbone(x)
        
        # Decode
        x = self.decoder(features)
        
        # Upsample to target resolution
        x = F.interpolate(x, size=self.input_resolution, mode='bilinear', align_corners=False)
        
        # Classify
        x = self.classifier(x)
        
        return x

class TrainingConfig:
    """Configuration class for training parameters"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Default configuration
        self.config = {
            # Model configuration
            "input_resolution": (512, 512),
            "num_classes": 4,
            "use_case": "commercial",  # racing, commercial, precision, research
            
            # Training parameters
            "epochs": {
                "stage1": 15,
                "stage2": 20
            },
            "batch_size": 8,
            "learning_rate": {
                "stage1": 2e-4,
                "stage2": 1e-4
            },
            "weight_decay": 1e-4,
            "mixed_precision": True,
            
            # Data augmentation
            "augmentation": {
                "horizontal_flip": 0.5,
                "rotation": 0.3,
                "color_jitter": 0.2,
                "blur": 0.1,
                "noise": 0.1
            },
            
            # Training optimization
            "gradient_accumulation": 2,
            "early_stopping": {
                "enabled": True,
                "patience": 5,
                "min_delta": 0.001
            },
            "learning_rate_schedule": {
                "type": "onecycle",  # onecycle, step, cosine
                "factor": 2.0
            },
            
            # Checkpointing
            "save_every_n_epochs": 5,
            "save_best_only": True,
            
            # Hardware
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True
        }
        
        # Load custom config if provided
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        
        # Deep merge configurations
        self._deep_update(self.config, user_config)
        print(f"✅ Configuration loaded from {config_path}")
    
    def save_config(self, config_path: str):
        """Save current configuration to JSON file"""
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"💾 Configuration saved to {config_path}")
    
    def _deep_update(self, base_dict, update_dict):
        """Recursively update nested dictionaries"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get_model_name(self) -> str:
        """Generate model name based on configuration"""
        res = f"{self.config['input_resolution'][0]}x{self.config['input_resolution'][1]}"
        use_case = self.config['use_case']
        return f"bisenetv2_uav_{use_case}_{res}"
    
    def get_transforms(self):
        """Create data transforms based on configuration"""
        res = self.config['input_resolution']
        aug = self.config['augmentation']
        
        train_transform = A.Compose([
            A.Resize(res[1], res[0]),
            A.HorizontalFlip(p=aug['horizontal_flip']),
            A.RandomRotate90(p=aug['rotation']),
            A.ColorJitter(brightness=aug['color_jitter'], contrast=aug['color_jitter'], p=0.3),
            A.GaussianBlur(p=aug['blur']),
            A.GaussNoise(p=aug['noise']),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        val_transform = A.Compose([
            A.Resize(res[1], res[0]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        return train_transform, val_transform

class ConfigurableTrainer:
    """Configurable trainer with advanced features"""
    
    def __init__(self, config: TrainingConfig, output_dir: str = "outputs"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if (
            config.config['mixed_precision'] and self.device.type == 'cuda'
        ) else None
        
        if self.scaler:
            print("⚡ Mixed precision training enabled")
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_iou': [],
            'learning_rates': []
        }
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def create_model(self) -> ConfigurableBiSeNet:
        """Create model based on configuration"""
        model = ConfigurableBiSeNet(
            num_classes=self.config.config['num_classes'],
            input_resolution=self.config.config['input_resolution'],
            use_case=self.config.config['use_case']
        )
        
        model.to(self.device)
        
        print(f"🏗️  Model created:")
        print(f"   Architecture: {model.use_case} ({model._determine_complexity()})")
        print(f"   Parameters: {model.param_count:,}")
        print(f"   Input resolution: {model.input_resolution}")
        print(f"   Model size: ~{model.param_count * 4 / 1024 / 1024:.1f} MB")
        
        return model
    
    def train_stage(self, stage: int, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader) -> Dict[str, List[float]]:
        """Train a single stage with full configuration support"""
        
        stage_name = f"Stage {stage}"
        epochs = self.config.config['epochs'][f'stage{stage}']
        lr = self.config.config['learning_rate'][f'stage{stage}']
        
        print(f"\n🚀 {stage_name}: Training for {epochs} epochs")
        print("=" * 50)
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=self.config.config['weight_decay']
        )
        
        scheduler = self._create_scheduler(optimizer, epochs, len(train_loader))
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Reset best loss for this stage
        stage_best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training phase
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion, epoch)
            
            # Validation phase
            val_loss, val_iou = self._validate_epoch(model, val_loader, criterion)
            
            # Update learning rate
            if scheduler:
                scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_iou'].append(val_iou)
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Logging
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1:2d}/{epochs}: "
                  f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
                  f"IoU: {val_iou:.3f}, LR: {optimizer.param_groups[0]['lr']:.2e}, "
                  f"Time: {epoch_time:.1f}s")
            
            # Save checkpoints
            is_best = val_loss < stage_best_loss
            if is_best:
                stage_best_loss = val_loss
                self._save_checkpoint(model, optimizer, epoch, val_loss, 
                                    f"{self.config.get_model_name()}_stage{stage}_best.pth")
            
            # Regular checkpoint
            if (epoch + 1) % self.config.config['save_every_n_epochs'] == 0:
                self._save_checkpoint(model, optimizer, epoch, val_loss,
                                    f"{self.config.get_model_name()}_stage{stage}_epoch{epoch+1}.pth")
            
            # Early stopping
            if self._check_early_stopping(val_loss):
                print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                break
        
        print(f"✅ {stage_name} completed! Best validation loss: {stage_best_loss:.4f}")
        return self.history
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                    optimizer: optim.Optimizer, criterion: nn.Module, epoch: int) -> float:
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        accumulation_steps = self.config.config['gradient_accumulation']
        
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Training")
        for i, batch in enumerate(pbar):
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks) / accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (i + 1) % accumulation_steps == 0:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(images)
                loss = criterion(outputs, masks) / accumulation_steps
                loss.backward()
                
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'loss': loss.item() * accumulation_steps})
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                       criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch"""
        model.eval()
        total_loss = 0.0
        iou_scores = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                images = batch['image'].to(self.device, non_blocking=True)
                masks = batch['mask'].to(self.device, non_blocking=True)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                total_loss += loss.item()
                
                # Calculate IoU
                predictions = torch.argmax(outputs, dim=1)
                iou = self._calculate_iou(predictions, masks)
                iou_scores.append(iou)
        
        avg_loss = total_loss / len(val_loader)
        avg_iou = torch.mean(torch.stack(iou_scores)) if iou_scores else 0.0
        
        return avg_loss, avg_iou.item() if isinstance(avg_iou, torch.Tensor) else avg_iou
    
    def _calculate_iou(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate IoU for segmentation"""
        # Focus on non-background classes
        valid_mask = targets > 0
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=predictions.device)
        
        predictions_valid = predictions[valid_mask]
        targets_valid = targets[valid_mask]
        
        intersection = (predictions_valid == targets_valid).sum().float()
        union = valid_mask.sum().float()
        
        return intersection / union if union > 0 else torch.tensor(0.0, device=predictions.device)
    
    def _create_scheduler(self, optimizer: optim.Optimizer, epochs: int, 
                         steps_per_epoch: int):
        """Create learning rate scheduler based on configuration"""
        schedule_config = self.config.config['learning_rate_schedule']
        
        if schedule_config['type'] == 'onecycle':
            return optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=optimizer.param_groups[0]['lr'] * schedule_config['factor'],
                epochs=epochs,
                steps_per_epoch=steps_per_epoch
            )
        elif schedule_config['type'] == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=epochs // 3,
                gamma=0.1
            )
        elif schedule_config['type'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs
            )
        else:
            return None
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check early stopping conditions"""
        if not self.config.config['early_stopping']['enabled']:
            return False
        
        patience = self.config.config['early_stopping']['patience']
        min_delta = self.config.config['early_stopping']['min_delta']
        
        if val_loss < self.best_val_loss - min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= patience
    
    def _save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                        epoch: int, val_loss: float, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config.config,
            'history': self.history
        }
        
        torch.save(checkpoint, self.output_dir / filename)
    
    def plot_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        if not self.history['train_loss']:
            print("No training history to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        epochs = range(1, len(self.history['train_loss']) + 1)
        ax1.plot(epochs, self.history['train_loss'], label='Training Loss')
        ax1.plot(epochs, self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # IoU curve
        ax2.plot(epochs, self.history['val_iou'], label='Validation IoU', color='green')
        ax2.set_title('Validation IoU')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('IoU')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate
        ax3.plot(epochs, self.history['learning_rates'], label='Learning Rate', color='orange')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True)
        
        # Loss comparison
        ax4.plot(epochs, self.history['train_loss'], label='Train', alpha=0.7)
        ax4.plot(epochs, self.history['val_loss'], label='Validation', alpha=0.7)
        ax4.fill_between(epochs, self.history['train_loss'], alpha=0.3)
        ax4.fill_between(epochs, self.history['val_loss'], alpha=0.3)
        ax4.set_title('Loss Comparison')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training plots saved to {save_path}")
        
        plt.show()

def create_sample_configs():
    """Create sample configuration files for different use cases"""
    
    configs = {
        "racing_256x256.json": {
            "input_resolution": (256, 256),
            "use_case": "racing",
            "epochs": {"stage1": 8, "stage2": 10},
            "batch_size": 12,
            "learning_rate": {"stage1": 3e-4, "stage2": 1.5e-4},
            "mixed_precision": True,
            "early_stopping": {"enabled": True, "patience": 3}
        },
        
        "commercial_512x512.json": {
            "input_resolution": (512, 512),
            "use_case": "commercial", 
            "epochs": {"stage1": 15, "stage2": 20},
            "batch_size": 8,
            "learning_rate": {"stage1": 2e-4, "stage2": 1e-4},
            "mixed_precision": True,
            "early_stopping": {"enabled": True, "patience": 5}
        },
        
        "precision_768x768.json": {
            "input_resolution": (768, 768),
            "use_case": "precision",
            "epochs": {"stage1": 20, "stage2": 25},
            "batch_size": 4,
            "learning_rate": {"stage1": 1.5e-4, "stage2": 7e-5},
            "mixed_precision": True,
            "gradient_accumulation": 4,
            "early_stopping": {"enabled": True, "patience": 7}
        },
        
        "research_1024x1024.json": {
            "input_resolution": (1024, 1024),
            "use_case": "research",
            "epochs": {"stage1": 25, "stage2": 30},
            "batch_size": 2,
            "learning_rate": {"stage1": 1e-4, "stage2": 5e-5},
            "mixed_precision": True,
            "gradient_accumulation": 8,
            "early_stopping": {"enabled": True, "patience": 10}
        }
    }
    
    config_dir = Path("configs/training")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, config in configs.items():
        config_path = config_dir / filename
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"📄 Created config: {config_path}")

def main():
    """Main training function with CLI support"""
    parser = argparse.ArgumentParser(description="Configurable UAV Landing Model Training")
    parser.add_argument("--config", type=str, help="Path to training configuration JSON")
    parser.add_argument("--use-case", type=str, choices=["racing", "commercial", "precision", "research"],
                       default="commercial", help="Training use case")
    parser.add_argument("--resolution", type=int, default=512, 
                       help="Input resolution (will be used for both width and height)")
    parser.add_argument("--epochs", type=int, help="Override epochs for both stages")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--create-configs", action="store_true", help="Create sample configurations and exit")
    
    args = parser.parse_args()
    
    if args.create_configs:
        create_sample_configs()
        return
    
    print("🚀 Configurable UAV Landing Model Training")
    print("=" * 50)
    
    # Load or create configuration
    if args.config:
        config = TrainingConfig(args.config)
    else:
        config = TrainingConfig()
        
        # Apply CLI overrides
        config.config['use_case'] = args.use_case
        config.config['input_resolution'] = (args.resolution, args.resolution)
        
        if args.epochs:
            config.config['epochs']['stage1'] = args.epochs
            config.config['epochs']['stage2'] = args.epochs
        
        if args.batch_size:
            config.config['batch_size'] = args.batch_size
    
    # Print configuration
    print(f"📋 Training Configuration:")
    print(f"   Use case: {config.config['use_case']}")
    print(f"   Resolution: {config.config['input_resolution']}")
    print(f"   Epochs: Stage1={config.config['epochs']['stage1']}, Stage2={config.config['epochs']['stage2']}")
    print(f"   Batch size: {config.config['batch_size']}")
    print(f"   Mixed precision: {config.config['mixed_precision']}")
    
    # Save configuration for reproducibility
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    config.save_config(output_dir / "training_config.json")
    
    # Create trainer
    trainer = ConfigurableTrainer(config, args.output_dir)
    
    # Create model
    model = trainer.create_model()
    
    if not TRAINING_AVAILABLE:
        print("⚠️  Training datasets not available - created model architecture only")
        print(f"✅ Model architecture: {model.param_count:,} parameters")
        return
    
    # TODO: Add dataset loading and training loop here
    print("🏗️  Model created successfully!")
    print("📈 Ready for training (dataset integration needed)")
    
    # Plot sample architecture info
    print(f"\n📊 Model Architecture Summary:")
    print(f"   Complexity: {model._determine_complexity()}")
    print(f"   Parameters: {model.param_count:,}")
    print(f"   Estimated size: ~{model.param_count * 4 / 1024 / 1024:.1f} MB")
    print(f"   Dropout rate: {model.dropout_rate}")

if __name__ == "__main__":
    main()
