#!/usr/bin/env python3
"""
Natural Semantic Segmentation Training
====================================

Train on the original 24 semantic classes from Semantic Drone Dataset.
This follows the dataset's natural structure instead of forcing artificial
"safe/caution/danger" mappings.

Classes (24 total):
0: unlabeled, 1: paved-area, 2: dirt, 3: grass, 4: gravel, 5: water,
6: rocks, 7: pool, 8: vegetation, 9: roof, 10: wall, 11: window,
12: door, 13: fence, 14: fence-pole, 15: person, 16: dog, 17: car,
18: bicycle, 19: tree, 20: bald-tree, 21: ar-marker, 22: obstacle, 23: conflicting
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import wandb
from typing import Dict, List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.mmseg_bisenetv2 import MMSegBiSeNetV2

# Define the natural class mapping from RGB to class indices
RGB_TO_CLASS = {
    (0, 0, 0): 0,           # unlabeled
    (128, 64, 128): 1,      # paved-area  
    (130, 76, 0): 2,        # dirt
    (0, 102, 0): 3,         # grass
    (112, 103, 87): 4,      # gravel
    (28, 42, 168): 5,       # water
    (48, 41, 30): 6,        # rocks
    (0, 50, 89): 7,         # pool
    (107, 142, 35): 8,      # vegetation
    (70, 70, 70): 9,        # roof
    (102, 102, 156): 10,    # wall
    (254, 228, 12): 11,     # window
    (254, 148, 12): 12,     # door
    (190, 153, 153): 13,    # fence
    (153, 153, 153): 14,    # fence-pole
    (255, 22, 96): 15,      # person
    (102, 51, 0): 16,       # dog
    (9, 143, 150): 17,      # car
    (119, 11, 32): 18,      # bicycle
    (51, 51, 0): 19,        # tree
    (190, 250, 190): 20,    # bald-tree
    (112, 150, 146): 21,    # ar-marker
    (2, 135, 115): 22,      # obstacle
    (255, 0, 0): 23,        # conflicting
}

CLASS_NAMES = [
    'unlabeled', 'paved-area', 'dirt', 'grass', 'gravel', 'water',
    'rocks', 'pool', 'vegetation', 'roof', 'wall', 'window',
    'door', 'fence', 'fence-pole', 'person', 'dog', 'car',
    'bicycle', 'tree', 'bald-tree', 'ar-marker', 'obstacle', 'conflicting'
]

class NaturalSemanticDataset(torch.utils.data.Dataset):
    """Dataset for natural 24-class semantic segmentation."""
    
    def __init__(self, data_root: str, split: str = 'train', image_size: int = 512):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.split = split
        
        # Get image and label paths
        if split == 'train':
            self.image_dir = self.data_root / "original_images"
            self.label_dir = self.data_root / "label_images_semantic"
        else:  # validation - use same for now
            self.image_dir = self.data_root / "original_images" 
            self.label_dir = self.data_root / "label_images_semantic"
            
        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob("*.jpg")))
        
        # Limit dataset size for faster iteration
        if split == 'train':
            self.image_files = self.image_files[:300]  # Use first 300 for training
        else:
            self.image_files = self.image_files[300:350]  # Use next 50 for validation
            
        print(f"Found {len(self.image_files)} {split} images")
        
    def rgb_to_class_index(self, rgb_label):
        """Convert RGB label to class indices."""
        h, w, c = rgb_label.shape
        class_label = np.zeros((h, w), dtype=np.uint8)
        
        for rgb, class_idx in RGB_TO_CLASS.items():
            mask = np.all(rgb_label == rgb, axis=2)
            class_label[mask] = class_idx
            
        return class_label
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        label_path = self.label_dir / img_path.name
        
        import cv2
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = cv2.imread(str(label_path))
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        label = cv2.resize(label, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        # Convert label to class indices
        label = self.rgb_to_class_index(label)
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(label).long()
        
        return image, label

def calculate_class_weights(dataloader, num_classes=24):
    """Calculate class weights based on frequency."""
    class_counts = torch.zeros(num_classes)
    
    print("Calculating class weights...")
    for batch_idx, (_, labels) in enumerate(tqdm(dataloader)):
        if batch_idx > 50:  # Sample from first 50 batches
            break
            
        for class_idx in range(num_classes):
            class_counts[class_idx] += (labels == class_idx).sum()
    
    # Calculate weights (inverse frequency)
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (num_classes * class_counts + 1e-8)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print("\nClass weights calculated:")
    for i, (name, weight) in enumerate(zip(CLASS_NAMES, class_weights)):
        print(f"{name:12}: {weight:.4f} (count: {class_counts[i]:,})")
    
    return class_weights

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Memory cleanup
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    return total_loss / num_batches

def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    class_correct = torch.zeros(24)
    class_total = torch.zeros(24)
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.argmax(outputs, dim=1)
            correct_pixels += (predictions == labels).sum().item()
            total_pixels += labels.numel()
            
            # Per-class accuracy
            for class_idx in range(24):
                class_mask = (labels == class_idx)
                if class_mask.sum() > 0:
                    class_correct[class_idx] += (predictions[class_mask] == class_idx).sum().item()
                    class_total[class_idx] += class_mask.sum().item()
    
    accuracy = correct_pixels / total_pixels
    avg_loss = total_loss / len(dataloader)
    
    # Calculate per-class accuracies
    class_accuracies = class_correct / (class_total + 1e-8)
    
    print(f"\nValidation Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Mean Class Accuracy: {class_accuracies.mean():.4f}")
    
    # Print top performing classes
    top_classes = torch.argsort(class_accuracies, descending=True)[:10]
    print("\nTop 10 performing classes:")
    for i in top_classes:
        if class_total[i] > 0:
            print(f"{CLASS_NAMES[i]:12}: {class_accuracies[i]:.4f} ({class_total[i]:,} pixels)")
    
    return avg_loss, accuracy, class_accuracies.mean().item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, 
                       default=r'H:\landing-system\datasets\Aerial_Semantic_Segmentation_Drone_Dataset\dataset\semantic_drone_dataset',
                       help='Path to dataset root')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--image-size', type=int, default=512, help='Image size')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(
        project="uav-natural-semantic",
        config=vars(args),
        name=f"natural_24class_{args.image_size}px"
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = NaturalSemanticDataset(args.data_root, 'train', args.image_size)
    val_dataset = NaturalSemanticDataset(args.data_root, 'val', args.image_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_loader, 24)
    
    # Create model (24 classes)
    model = MMSegBiSeNetV2(num_classes=24).to(device)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Training loop
    best_accuracy = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        
        # Validate
        val_loss, val_accuracy, mean_class_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'mean_class_accuracy': mean_class_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.4f}, Mean Class Acc: {mean_class_acc:.4f}")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'class_weights': class_weights,
            }, 'outputs/natural_semantic_best.pth')
            print(f"New best model saved! Accuracy: {best_accuracy:.4f}")
    
    print(f"Training completed! Best accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    main() 