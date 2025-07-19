#!/usr/bin/env python3
"""
Practical Fine-Tuning Script for BiSeNetV2 UAV Landing Detection

This script provides a simplified, practical approach to fine-tuning your BiSeNetV2 model
using the DroneDeploy dataset structure you already have.

Usage:
    python practical_fine_tuning.py --data_path /path/to/dataset-medium --epochs 50
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from pathlib import Path
import cv2
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime
import logging
from tqdm import tqdm
import random
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DroneDeploy RGB color mappings to our landing classes
DRONEDEPLOY_RGB_TO_LANDING = {
    # RGB colors from DroneDeploy labels -> Our 4-class system
    (75, 25, 230): 2,    # BUILDING -> obstacle
    (180, 30, 145): 3,   # CLUTTER -> unsafe
    (75, 180, 60): 1,    # VEGETATION -> suitable (if flat)
    (48, 130, 245): 3,   # WATER -> unsafe
    (255, 255, 255): 1,  # GROUND -> suitable
    (200, 130, 0): 2,    # CAR -> obstacle
    (255, 0, 255): 0     # IGNORE -> background
}

LANDING_CLASSES = {
    0: "background",
    1: "suitable",      # Safe landing zones
    2: "obstacle",      # Buildings, cars, etc.
    3: "unsafe"         # Water, clutter, etc.
}

class SimpleBiSeNetV2(nn.Module):
    """
    Simplified BiSeNetV2 implementation that can load your pre-trained weights
    and be fine-tuned for our 4-class landing detection task.
    """
    
    def __init__(self, num_classes=4, pretrained_path=None):
        super(SimpleBiSeNetV2, self).__init__()
        
        self.num_classes = num_classes
        
        # Encoder (simplified ResNet-like backbone)
        self.encoder = self._make_encoder()
        
        # Decoder
        self.decoder = self._make_decoder()
        
        # Final classifier
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)
        
        # Auxiliary classifier for training
        self.aux_classifier = nn.Conv2d(512, num_classes, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
        
        # Load pre-trained weights if provided
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained_weights(pretrained_path)
            
    def _make_encoder(self):
        """Create encoder (backbone) network."""
        layers = []
        
        # Initial convolution
        layers.extend([
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        
        # Stage 1: 64 -> 128
        layers.extend(self._make_stage(64, 128, 2, stride=1))
        
        # Stage 2: 128 -> 256  
        layers.extend(self._make_stage(128, 256, 2, stride=2))
        
        # Stage 3: 256 -> 512
        layers.extend(self._make_stage(256, 512, 2, stride=2))
        
        return nn.Sequential(*layers)
        
    def _make_stage(self, in_channels, out_channels, num_blocks, stride=1):
        """Create a stage with residual blocks."""
        layers = []
        
        # First block with potential downsampling
        layers.append(BasicBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock(out_channels, out_channels, 1))
            
        return layers
        
    def _make_decoder(self):
        """Create decoder network."""
        return nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _load_pretrained_weights(self, pretrained_path):
        """Load pre-trained weights and adapt for our task."""
        try:
            logger.info(f"Loading pre-trained weights from {pretrained_path}")
            
            # Load the checkpoint
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Extract state dict (handle different checkpoint formats)
            if 'model' in checkpoint:
                pretrained_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
            else:
                pretrained_dict = checkpoint
                
            # Get current model state dict
            model_dict = self.state_dict()
            
            # Filter out classifier layers (we'll retrain these for our classes)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and 'classifier' not in k and 'aux' not in k}
            
            # Update model dict
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            
            logger.info(f"Successfully loaded {len(pretrained_dict)} pre-trained parameters")
            
        except Exception as e:
            logger.warning(f"Could not load pre-trained weights: {e}")
            logger.info("Training from scratch")
            
    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        
        # Decoder
        decoded = self.decoder(features)
        
        # Main output
        output = self.classifier(decoded)
        output = nn.functional.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        if self.training:
            # Auxiliary output for training
            aux_output = self.aux_classifier(features)
            aux_output = nn.functional.interpolate(aux_output, size=x.shape[2:], mode='bilinear', align_corners=False)
            return output, aux_output
        else:
            return output

class BasicBlock(nn.Module):
    """Basic residual block."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class DroneDeployDataset(Dataset):
    """
    Dataset for loading DroneDeploy images and labels.
    Converts RGB color labels to our 4-class integer labels.
    """
    
    def __init__(self, data_root, transform=None, target_size=(512, 512)):
        self.data_root = Path(data_root)
        self.transform = transform
        self.target_size = target_size
        
        # Find image and label files
        self.image_dir = self.data_root / "images"
        self.label_dir = self.data_root / "labels"
        
        # Get all image files
        self.image_files = list(self.image_dir.glob("*.tif"))
        
        # Filter to only include files that have corresponding labels
        self.samples = []
        for img_path in self.image_files:
            # Convert image filename to label filename
            label_name = img_path.stem.replace("-ortho", "-label") + ".png"
            label_path = self.label_dir / label_name
            
            if label_path.exists():
                self.samples.append((str(img_path), str(label_path)))
                
        logger.info(f"Found {len(self.samples)} image-label pairs in {data_root}")
        
        if len(self.samples) == 0:
            logger.warning("No samples found! Check data paths and file naming.")
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load label (RGB format)
        label_rgb = cv2.imread(label_path)
        label_rgb = cv2.cvtColor(label_rgb, cv2.COLOR_BGR2RGB)
        
        # Convert RGB label to integer classes
        label = self._rgb_to_class_label(label_rgb)
        
        # Resize to target size
        image = cv2.resize(image, self.target_size)
        label = cv2.resize(label, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']
        else:
            # Convert to tensors
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            label = torch.from_numpy(label).long()
            
        return image, label
        
    def _rgb_to_class_label(self, rgb_label):
        """Convert RGB color label to integer class label."""
        h, w, _ = rgb_label.shape
        class_label = np.zeros((h, w), dtype=np.int64)  # Ensure int64 for PyTorch
        
        for rgb_color, class_id in DRONEDEPLOY_RGB_TO_LANDING.items():
            # Find pixels matching this RGB color
            mask = np.all(rgb_label == np.array(rgb_color).reshape(1, 1, 3), axis=2)
            class_label[mask] = class_id
            
        return class_label

class UAVFineTuner:
    """Main fine-tuning class for the UAV landing detection model."""
    
    def __init__(self, 
                 data_path: str,
                 pretrained_model_path: str = None,
                 output_dir: str = "models/fine_tuned",
                 device: str = "auto"):
        
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Model setup
        self.model = SimpleBiSeNetV2(
            num_classes=4, 
            pretrained_path=pretrained_model_path
        ).to(self.device)
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
        
    def create_data_loaders(self, batch_size=8, val_split=0.2):
        """Create train and validation data loaders."""
        
        # Define transforms
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),  # Aerial view can be flipped vertically
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        val_transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        # Create dataset
        full_dataset = DroneDeployDataset(self.data_path, transform=None)
        
        # Split into train and validation
        total_samples = len(full_dataset)
        val_samples = int(total_samples * val_split)
        train_samples = total_samples - val_samples
        
        # Random split
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        train_indices = indices[:train_samples]
        val_indices = indices[train_samples:]
        
        # Create subset datasets
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        
        # Apply transforms
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Created data loaders: {len(train_dataset)} train, {len(val_dataset)} val samples")
        
        return train_loader, val_loader
        
    def train(self, 
              epochs=50,
              batch_size=8, 
              learning_rate=1e-4,
              val_split=0.2,
              save_best_only=True):
        """Train the model."""
        
        logger.info("🚀 Starting fine-tuning...")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(batch_size, val_split)
        
        # Loss function with class weights (emphasize landing-relevant classes)
        class_weights = torch.tensor([1.0, 2.0, 1.5, 1.5]).to(self.device)  # Emphasize suitable areas
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
        
        # Optimizer with different learning rates for different parts
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'classifier' in name or 'aux' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
                
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': learning_rate * 0.1},  # Lower LR for backbone
            {'params': head_params, 'lr': learning_rate}  # Higher LR for classifier heads
        ], weight_decay=1e-4)
        
        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=learning_rate * 0.01
        )
        
        # Training loop
        best_miou = 0.0
        train_losses = []
        val_mious = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for batch_idx, (images, targets) in enumerate(train_pbar):
                images, targets = images.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                if self.model.training:
                    main_out, aux_out = self.model(images)
                    main_loss = criterion(main_out, targets)
                    aux_loss = criterion(aux_out, targets)
                    loss = main_loss + 0.4 * aux_loss
                else:
                    main_out = self.model(images)
                    loss = criterion(main_out, targets)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            val_miou = self._validate(val_loader, criterion)
            val_mious.append(val_miou)
            
            # Update scheduler
            scheduler.step()
            
            # Save model if best
            if val_miou > best_miou:
                best_miou = val_miou
                self._save_model(epoch, val_miou, optimizer, scheduler, "best_model.pth")
                logger.info(f"✅ New best model saved! mIoU: {val_miou:.4f}")
                
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self._save_model(epoch, val_miou, optimizer, scheduler, f"checkpoint_epoch_{epoch+1}.pth")
                
            logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                       f"Val mIoU: {val_miou:.4f}, Best mIoU: {best_miou:.4f}")
                       
        logger.info(f"🎉 Training completed! Best mIoU: {best_miou:.4f}")
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_mious': val_mious,
            'best_miou': best_miou
        }
        
        with open(self.output_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
            
        return history
        
    def _validate(self, val_loader, criterion):
        """Validate the model and compute mIoU."""
        self.model.eval()
        
        total_loss = 0.0
        intersection = torch.zeros(4).to(self.device)
        union = torch.zeros(4).to(self.device)
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for images, targets in val_pbar:
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Use main output only for validation
                    
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                # Compute IoU
                preds = torch.argmax(outputs, dim=1)
                
                for cls in range(4):
                    pred_mask = (preds == cls)
                    target_mask = (targets == cls)
                    
                    intersection[cls] += (pred_mask & target_mask).sum().float()
                    union[cls] += (pred_mask | target_mask).sum().float()
                    
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
        # Calculate mIoU
        iou_per_class = intersection / (union + 1e-8)
        miou = iou_per_class.mean().item()
        
        # Log per-class IoU
        for cls in range(4):
            class_name = LANDING_CLASSES[cls]
            logger.info(f"  {class_name}: IoU = {iou_per_class[cls].item():.4f}")
            
        return miou
        
    def _save_model(self, epoch, miou, optimizer, scheduler, filename):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'miou': miou,
            'num_classes': 4,
            'class_names': LANDING_CLASSES
        }
        
        torch.save(checkpoint, self.output_dir / filename)
        
    def export_to_onnx(self, model_path=None, output_path=None):
        """Export the trained model to ONNX format."""
        
        if model_path:
            # Load specific model
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if output_path is None:
            output_path = self.output_dir / "bisenetv2_uav_landing.onnx"
            
        logger.info(f"Exporting model to ONNX: {output_path}")
        
        self.model.eval()
        dummy_input = torch.randn(1, 3, 512, 512).to(self.device)
        
        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'output': {0: 'batch_size', 2: 'height', 3: 'width'}
                }
            )
            logger.info(f"✅ ONNX export successful: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ ONNX export failed: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description="Practical Fine-Tuning for UAV Landing Detection")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to DroneDeploy dataset directory")
    parser.add_argument("--pretrained_model", type=str, 
                       help="Path to pre-trained BiSeNetV2 model")
    parser.add_argument("--output_dir", type=str, default="models/fine_tuned",
                       help="Output directory for trained models")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.2,
                       help="Validation split ratio")
    parser.add_argument("--device", type=str, default="auto",
                       help="Training device (auto/cpu/cuda)")
    parser.add_argument("--export_onnx", action="store_true",
                       help="Export best model to ONNX after training")
    
    args = parser.parse_args()
    
    # Initialize fine-tuner
    trainer = UAVFineTuner(
        data_path=args.data_path,
        pretrained_model_path=args.pretrained_model,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Train the model
    history = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split
    )
    
    # Export to ONNX if requested
    if args.export_onnx:
        best_model_path = Path(args.output_dir) / "best_model.pth"
        if best_model_path.exists():
            onnx_path = trainer.export_to_onnx(str(best_model_path))
            if onnx_path:
                print(f"\n🎯 Model exported to ONNX: {onnx_path}")
                
    print(f"\n🎉 Fine-tuning completed!")
    print(f"📊 Final best mIoU: {history['best_miou']:.4f}")
    print(f"💾 Models saved in: {args.output_dir}")

if __name__ == "__main__":
    main()
