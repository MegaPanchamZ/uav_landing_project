#!/usr/bin/env python3
"""
Three-Step Fine-Tuning Pipeline for UAV Landing Zone Detection

Step 1: BiSeNetV2 Base Model (Cityscapes pre-trained)
Step 2: Intermediate Fine-Tuning (DroneDeploy - General Aerial View)  
Step 3: Task-Specific Fine-Tuning (UDD-6 - Low-Altitude Drone View)

This creates a robust model progression from general semantic segmentation 
to specialized UAV landing zone detection.
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BiSeNetV2(nn.Module):
    """
    BiSeNetV2 implementation for semantic segmentation.
    Optimized for real-time performance on UAV platforms.
    """
    
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super(BiSeNetV2, self).__init__()
        
        self.num_classes = num_classes
        
        # Detail Branch - High resolution, low-level features
        self.detail_branch = self._make_detail_branch()
        
        # Semantic Branch - Low resolution, high-level features  
        self.semantic_branch = self._make_semantic_branch()
        
        # Bilateral Guided Aggregation
        self.bga = BilateralGuidedAggregation(128, 128)
        
        # Segmentation Head
        self.seg_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )
        
        # Auxiliary heads for training
        self.aux_head_detail = nn.Conv2d(64, num_classes, 1)
        self.aux_head_semantic = nn.Conv2d(128, num_classes, 1)
        
        if pretrained:
            self._load_pretrained_weights()
            
    def _make_detail_branch(self):
        """Detail branch for high-resolution feature extraction."""
        return nn.Sequential(
            # Stage 1
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Stage 2  
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Stage 3
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
    def _make_semantic_branch(self):
        """Semantic branch with lightweight encoder."""
        return nn.Sequential(
            # Stem block
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Stage 3 (1/8)
            self._make_stage(16, 32, 2, stride=2),
            
            # Stage 4 (1/16)  
            self._make_stage(32, 64, 2, stride=2),
            
            # Stage 5 (1/32)
            self._make_stage(64, 128, 2, stride=2),
            
            # Global Average Pool
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int, stride: int = 1):
        """Create a stage with multiple residual blocks."""
        blocks = []
        blocks.append(GatherAndExpansionBlock(in_channels, out_channels, stride))
        for _ in range(num_blocks - 1):
            blocks.append(GatherAndExpansionBlock(out_channels, out_channels, 1))
        return nn.Sequential(*blocks)
        
    def _load_pretrained_weights(self):
        """Load Cityscapes pre-trained weights if available."""
        try:
            # In practice, you would download from:
            # https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/bisenetv2.pth
            checkpoint_path = "pretrained/bisenetv2_cityscapes.pth"
            if Path(checkpoint_path).exists():
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                self.load_state_dict(checkpoint, strict=False)
                logger.info(f"Loaded pre-trained weights from {checkpoint_path}")
            else:
                logger.warning("Pre-trained weights not found, training from scratch")
        except Exception as e:
            logger.warning(f"Could not load pre-trained weights: {e}")
            
    def forward(self, x):
        # Detail branch
        detail_feat = self.detail_branch(x)  # 1/4 resolution
        
        # Semantic branch  
        semantic_feat = self.semantic_branch(x)  # 1/32 resolution
        semantic_feat = nn.functional.interpolate(
            semantic_feat, size=detail_feat.shape[2:], mode='bilinear', align_corners=False
        )
        
        # Bilateral guided aggregation
        fused_feat = self.bga(detail_feat, semantic_feat)
        
        # Main segmentation output
        seg_out = self.seg_head(fused_feat)
        seg_out = nn.functional.interpolate(
            seg_out, size=x.shape[2:], mode='bilinear', align_corners=False
        )
        
        if self.training:
            # Auxiliary outputs for training
            aux_detail = self.aux_head_detail(detail_feat)
            aux_detail = nn.functional.interpolate(
                aux_detail, size=x.shape[2:], mode='bilinear', align_corners=False
            )
            
            aux_semantic = self.aux_head_semantic(semantic_feat) 
            aux_semantic = nn.functional.interpolate(
                aux_semantic, size=x.shape[2:], mode='bilinear', align_corners=False
            )
            
            return seg_out, aux_detail, aux_semantic
        else:
            return seg_out

class GatherAndExpansionBlock(nn.Module):
    """Gather-and-Expansion block for BiSeNetV2."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expansion: int = 6):
        super().__init__()
        
        mid_channels = in_channels * expansion
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.dwconv = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.dwconv(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn3(out)
        
        out = out + residual
        out = self.relu(out)
        
        return out

class BilateralGuidedAggregation(nn.Module):
    """Bilateral Guided Aggregation module."""
    
    def __init__(self, detail_channels: int, semantic_channels: int):
        super().__init__()
        
        self.detail_dwconv = nn.Conv2d(detail_channels, detail_channels, 3, 1, 1, 
                                      groups=detail_channels, bias=False)
        self.detail_conv = nn.Conv2d(detail_channels, detail_channels, 1, bias=False)
        
        self.semantic_conv = nn.Sequential(
            nn.Conv2d(semantic_channels, detail_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(detail_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(detail_channels, detail_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(detail_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, detail_feat, semantic_feat):
        # Detail branch
        detail_dwconv = self.detail_dwconv(detail_feat)
        detail_conv = self.detail_conv(detail_feat)
        detail_out = detail_dwconv + detail_conv
        
        # Semantic branch
        semantic_out = self.semantic_conv(semantic_feat)
        
        # Fusion
        fused = detail_out * semantic_out
        out = self.conv_out(fused)
        
        return out

class UAVDataset(Dataset):
    """
    Generic dataset class for UAV imagery with different annotation formats.
    Supports DroneDeploy and UDD-6 dataset structures.
    """
    
    def __init__(self, 
                 data_root: str,
                 split: str = "train",
                 dataset_type: str = "dronedeploy",  # "dronedeploy" or "udd6"
                 transform: Optional[transforms.Compose] = None,
                 target_size: Tuple[int, int] = (512, 512)):
        
        self.data_root = Path(data_root)
        self.split = split
        self.dataset_type = dataset_type
        self.transform = transform
        self.target_size = target_size
        
        # Dataset-specific configurations
        if dataset_type == "dronedeploy":
            self.num_classes = 10  # Adjust based on DroneDeploy classes
            self.class_mapping = self._get_dronedeploy_mapping()
        elif dataset_type == "udd6":
            self.num_classes = 6   # UDD-6 classes
            self.class_mapping = self._get_udd6_mapping()
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
            
        # Load image and annotation paths
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples from {dataset_type} {split} split")
        
    def _get_dronedeploy_mapping(self) -> Dict[int, int]:
        """Map DroneDeploy classes to our 6-class system."""
        return {
            # Map original classes to our target classes:
            # 0: background -> 0: background
            # 1-3: vegetation/suitable -> 1: suitable  
            # 4-6: buildings/obstacles -> 3: obstacles
            # 7-9: water/unsafe -> 4: unsafe
            0: 0,  # background
            1: 1, 2: 1, 3: 1,  # suitable areas
            4: 3, 5: 3, 6: 3,  # obstacles  
            7: 4, 8: 4, 9: 4   # unsafe areas
        }
        
    def _get_udd6_mapping(self) -> Dict[int, int]:
        """Map UDD-6 classes to our system."""
        return {
            0: 0,  # other -> background
            1: 3,  # facade -> obstacles
            2: 1,  # road -> suitable
            3: 1,  # vegetation -> suitable (if flat)
            4: 4,  # vehicle -> unsafe
            5: 3,  # roof -> obstacles  
        }
        
    def _load_samples(self) -> List[Tuple[str, str]]:
        """Load image and annotation file paths."""
        samples = []
        
        if self.dataset_type == "dronedeploy":
            img_dir = self.data_root / "images" / self.split
            ann_dir = self.data_root / "annotations" / self.split
            
        elif self.dataset_type == "udd6":
            img_dir = self.data_root / self.split / "src"
            ann_dir = self.data_root / self.split / "gt"
            
        if not img_dir.exists() or not ann_dir.exists():
            logger.warning(f"Dataset directories not found: {img_dir}, {ann_dir}")
            return []
            
        # Find matching image and annotation pairs
        for img_path in sorted(img_dir.glob("*.jpg")):
            # Look for corresponding annotation
            ann_path = ann_dir / f"{img_path.stem}.png"
            if ann_path.exists():
                samples.append((str(img_path), str(ann_path)))
                
        return samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotation
        annotation = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize to target size
        image = cv2.resize(image, self.target_size)
        annotation = cv2.resize(annotation, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Apply class mapping
        mapped_annotation = np.zeros_like(annotation)
        for orig_class, target_class in self.class_mapping.items():
            mapped_annotation[annotation == orig_class] = target_class
            
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        annotation = torch.from_numpy(mapped_annotation).long()
        
        # Apply transforms
        if self.transform:
            # Note: This is simplified - in practice you'd want paired transforms
            image = self.transform(image)
            
        return image, annotation

class ThreeStepTrainer:
    """
    Implements the three-step fine-tuning pipeline:
    Step 1: BiSeNetV2 (Cityscapes pre-trained)
    Step 2: Intermediate fine-tuning on DroneDeploy  
    Step 3: Task-specific fine-tuning on UDD-6
    """
    
    def __init__(self, 
                 output_dir: str = "models/fine_tuning",
                 device: str = "auto"):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Training configuration
        self.config = {
            "step1": {
                "epochs": 50,
                "lr": 1e-3,
                "batch_size": 8,
                "dataset": "dronedeploy"
            },
            "step2": {
                "epochs": 30, 
                "lr": 5e-4,
                "batch_size": 12,
                "dataset": "udd6"
            },
            "step3": {
                "epochs": 20,
                "lr": 1e-4,
                "batch_size": 16,
                "dataset": "udd6"  # Fine-tune further on UDD-6
            }
        }
        
    def step1_base_training(self, 
                           dronedeploy_path: str = "data/dronedeploy",
                           resume: bool = False) -> str:
        """
        Step 1: Train BiSeNetV2 on DroneDeploy for general aerial view understanding.
        """
        logger.info("🚀 Step 1: Base training on DroneDeploy dataset")
        
        # Initialize model
        model = BiSeNetV2(num_classes=6, pretrained=True)
        model = model.to(self.device)
        
        # Data loading
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = UAVDataset(dronedeploy_path, "train", "dronedeploy", train_transform)
        val_dataset = UAVDataset(dronedeploy_path, "val", "dronedeploy", val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config["step1"]["batch_size"], 
                                 shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.config["step1"]["batch_size"], 
                               shuffle=False, num_workers=4)
        
        # Training setup
        optimizer = optim.AdamW(model.parameters(), lr=self.config["step1"]["lr"], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config["step1"]["epochs"])
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        # Training loop
        best_iou = 0.0
        for epoch in range(self.config["step1"]["epochs"]):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                images, targets = images.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                
                if model.training:
                    main_out, aux_detail, aux_semantic = model(images)
                    
                    # Multi-scale loss
                    main_loss = criterion(main_out, targets)
                    aux_loss1 = criterion(aux_detail, targets)
                    aux_loss2 = criterion(aux_semantic, targets)
                    
                    loss = main_loss + 0.4 * aux_loss1 + 0.4 * aux_loss2
                else:
                    main_out = model(images)
                    loss = criterion(main_out, targets)
                    
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                    
            # Validation
            val_iou = self._validate_model(model, val_loader, criterion)
            scheduler.step()
            
            # Save best model
            if val_iou > best_iou:
                best_iou = val_iou
                step1_path = self.output_dir / "step1_dronedeploy_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iou': val_iou
                }, step1_path)
                
            logger.info(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
                       f"Val IoU: {val_iou:.4f}, Best IoU: {best_iou:.4f}")
                       
        logger.info(f"✅ Step 1 completed. Best IoU: {best_iou:.4f}")
        return str(step1_path)
        
    def step2_intermediate_finetuning(self, 
                                    step1_model_path: str,
                                    udd6_path: str = "data/udd6") -> str:
        """
        Step 2: Intermediate fine-tuning on UDD-6 for low-altitude drone view.
        """
        logger.info("🚀 Step 2: Intermediate fine-tuning on UDD-6")
        
        # Load Step 1 model
        model = BiSeNetV2(num_classes=6, pretrained=False)
        checkpoint = torch.load(step1_model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        # Freeze early layers for fine-tuning
        for param in model.detail_branch[:2].parameters():
            param.requires_grad = False
            
        # Data loading
        train_dataset = UAVDataset(udd6_path, "train", "udd6")
        val_dataset = UAVDataset(udd6_path, "val", "udd6")
        
        train_loader = DataLoader(train_dataset, batch_size=self.config["step2"]["batch_size"], 
                                 shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.config["step2"]["batch_size"], 
                               shuffle=False, num_workers=4)
        
        # Fine-tuning setup
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                               lr=self.config["step2"]["lr"], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        # Fine-tuning loop
        best_iou = 0.0
        for epoch in range(self.config["step2"]["epochs"]):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                images, targets = images.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                
                if model.training:
                    main_out, aux_detail, aux_semantic = model(images)
                    main_loss = criterion(main_out, targets)
                    aux_loss1 = criterion(aux_detail, targets)
                    aux_loss2 = criterion(aux_semantic, targets)
                    loss = main_loss + 0.2 * aux_loss1 + 0.2 * aux_loss2
                else:
                    main_out = model(images)
                    loss = criterion(main_out, targets)
                    
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            # Validation
            val_iou = self._validate_model(model, val_loader, criterion)
            scheduler.step()
            
            # Save best model
            if val_iou > best_iou:
                best_iou = val_iou
                step2_path = self.output_dir / "step2_udd6_intermediate.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iou': val_iou
                }, step2_path)
                
            logger.info(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
                       f"Val IoU: {val_iou:.4f}, Best IoU: {best_iou:.4f}")
                       
        logger.info(f"✅ Step 2 completed. Best IoU: {best_iou:.4f}")
        return str(step2_path)
        
    def step3_task_specific_finetuning(self, 
                                     step2_model_path: str,
                                     udd6_path: str = "data/udd6") -> str:
        """
        Step 3: Task-specific fine-tuning for UAV landing zone detection.
        """
        logger.info("🚀 Step 3: Task-specific fine-tuning for landing zones")
        
        # Load Step 2 model
        model = BiSeNetV2(num_classes=6, pretrained=False)
        checkpoint = torch.load(step2_model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        # Unfreeze all layers for final fine-tuning
        for param in model.parameters():
            param.requires_grad = True
            
        # Data loading with landing-specific augmentations
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),  # Aerial view specific
            transforms.RandomRotation(360),      # Any rotation for aerial
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = UAVDataset(udd6_path, "train", "udd6", train_transform)
        val_dataset = UAVDataset(udd6_path, "val", "udd6")
        
        train_loader = DataLoader(train_dataset, batch_size=self.config["step3"]["batch_size"], 
                                 shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.config["step3"]["batch_size"], 
                               shuffle=False, num_workers=4)
        
        # Final fine-tuning setup
        optimizer = optim.AdamW(model.parameters(), lr=self.config["step3"]["lr"], 
                               weight_decay=5e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        
        # Class-weighted loss for landing zone focus
        class_weights = torch.tensor([1.0, 2.0, 1.5, 2.0, 3.0, 1.0]).to(self.device)  
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
        
        # Final training loop
        best_iou = 0.0
        for epoch in range(self.config["step3"]["epochs"]):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                images, targets = images.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                
                if model.training:
                    main_out, aux_detail, aux_semantic = model(images)
                    main_loss = criterion(main_out, targets)
                    aux_loss1 = criterion(aux_detail, targets)
                    aux_loss2 = criterion(aux_semantic, targets)
                    loss = main_loss + 0.1 * aux_loss1 + 0.1 * aux_loss2
                else:
                    main_out = model(images)
                    loss = criterion(main_out, targets)
                    
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            # Validation
            val_iou = self._validate_model(model, val_loader, criterion)
            scheduler.step()
            
            # Save best model
            if val_iou > best_iou:
                best_iou = val_iou
                step3_path = self.output_dir / "step3_udd6_final.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iou': val_iou
                }, step3_path)
                
            logger.info(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
                       f"Val IoU: {val_iou:.4f}, Best IoU: {best_iou:.4f}")
                       
        logger.info(f"✅ Step 3 completed. Best IoU: {best_iou:.4f}")
        return str(step3_path)
        
    def export_to_onnx(self, pytorch_model_path: str, onnx_output_path: str = None):
        """Export the final trained model to ONNX for deployment."""
        
        if onnx_output_path is None:
            onnx_output_path = str(self.output_dir / "bisenetv2_udd6_final.onnx")
            
        logger.info(f"🔄 Exporting model to ONNX: {onnx_output_path}")
        
        # Load trained model
        model = BiSeNetV2(num_classes=6, pretrained=False)
        checkpoint = torch.load(pytorch_model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 512, 512).to(self.device)
        
        # Export to ONNX
        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_output_path,
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
            logger.info(f"✅ ONNX export successful: {onnx_output_path}")
            return onnx_output_path
            
        except Exception as e:
            logger.error(f"❌ ONNX export failed: {e}")
            return None
            
    def _validate_model(self, model, val_loader, criterion):
        """Validate model and compute mean IoU."""
        model.eval()
        val_loss = 0.0
        
        # IoU calculation
        intersection = torch.zeros(6).to(self.device)
        union = torch.zeros(6).to(self.device)
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Use main output only
                    
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Compute IoU
                preds = torch.argmax(outputs, dim=1)
                
                for cls in range(6):
                    pred_mask = (preds == cls)
                    target_mask = (targets == cls)
                    
                    intersection[cls] += (pred_mask & target_mask).sum().float()
                    union[cls] += (pred_mask | target_mask).sum().float()
                    
        # Calculate mean IoU
        iou_per_class = intersection / (union + 1e-8)
        mean_iou = iou_per_class.mean().item()
        
        return mean_iou
        
    def run_full_pipeline(self, 
                         dronedeploy_path: str = "data/dronedeploy",
                         udd6_path: str = "data/udd6") -> str:
        """Run the complete three-step fine-tuning pipeline."""
        
        logger.info("🚀 Starting three-step fine-tuning pipeline")
        
        # Step 1: Base training on DroneDeploy
        step1_model = self.step1_base_training(dronedeploy_path)
        
        # Step 2: Intermediate fine-tuning on UDD-6
        step2_model = self.step2_intermediate_finetuning(step1_model, udd6_path)
        
        # Step 3: Task-specific fine-tuning
        step3_model = self.step3_task_specific_finetuning(step2_model, udd6_path)
        
        # Export to ONNX
        onnx_path = self.export_to_onnx(step3_model)
        
        logger.info("🎉 Three-step fine-tuning pipeline completed!")
        logger.info(f"Final ONNX model: {onnx_path}")
        
        return onnx_path

def main():
    parser = argparse.ArgumentParser(description="Three-step fine-tuning for UAV landing detection")
    parser.add_argument("--dronedeploy_path", type=str, default="data/dronedeploy",
                       help="Path to DroneDeploy dataset")
    parser.add_argument("--udd6_path", type=str, default="data/udd6", 
                       help="Path to UDD-6 dataset")
    parser.add_argument("--output_dir", type=str, default="models/fine_tuning",
                       help="Output directory for models")
    parser.add_argument("--step", type=int, choices=[1, 2, 3], default=None,
                       help="Run specific step only (default: run all)")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Training device (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ThreeStepTrainer(output_dir=args.output_dir, device=args.device)
    
    if args.step is None:
        # Run full pipeline
        onnx_path = trainer.run_full_pipeline(args.dronedeploy_path, args.udd6_path)
        print(f"\n🎉 Pipeline completed! Final model: {onnx_path}")
        
    elif args.step == 1:
        # Run Step 1 only
        model_path = trainer.step1_base_training(args.dronedeploy_path)
        print(f"\n✅ Step 1 completed: {model_path}")
        
    elif args.step == 2:
        # Run Step 2 only (requires Step 1 model)
        step1_model = args.output_dir + "/step1_dronedeploy_best.pth"
        model_path = trainer.step2_intermediate_finetuning(step1_model, args.udd6_path)
        print(f"\n✅ Step 2 completed: {model_path}")
        
    elif args.step == 3:
        # Run Step 3 only (requires Step 2 model) 
        step2_model = args.output_dir + "/step2_udd6_intermediate.pth"
        model_path = trainer.step3_task_specific_finetuning(step2_model, args.udd6_path)
        onnx_path = trainer.export_to_onnx(model_path)
        print(f"\n✅ Step 3 completed: {onnx_path}")

if __name__ == "__main__":
    main()
