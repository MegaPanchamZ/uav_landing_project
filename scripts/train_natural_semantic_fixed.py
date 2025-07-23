#!/usr/bin/env python3
"""
Fixed Natural Semantic Segmentation Training
===========================================

Fixed version that handles missing label files and validates dataset integrity.
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

def validate_dataset(data_root: str):
    """Validate dataset and return list of valid image-label pairs."""
    data_root = Path(data_root)
    image_dir = data_root / "original_images"
    label_dir = data_root / "label_images_semantic"
    
    print(f"Validating dataset in {data_root}")
    print(f"Image dir: {image_dir}")
    print(f"Label dir: {label_dir}")
    
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    
    # Get all image files
    image_files = list(image_dir.glob("*.jpg"))
    print(f"Found {len(image_files)} image files")
    
    valid_pairs = []
    missing_labels = []
    
    import cv2
    
    for img_path in tqdm(image_files, desc="Validating dataset"):
        # Check corresponding label file - labels are .png, images are .jpg
        label_path = label_dir / (img_path.stem + ".png")
        
        # Try to read both files
        try:
            # Test image reading
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            # Test label reading  
            label = cv2.imread(str(label_path))
            if label is None:
                missing_labels.append(label_path.name)
                continue
                
            # Both files are readable
            valid_pairs.append((img_path, label_path))
            
        except Exception as e:
            print(f"Error reading {img_path.name}: {e}")
            continue
    
    print(f"\nValidation Results:")
    print(f"Total images: {len(image_files)}")
    print(f"Valid pairs: {len(valid_pairs)}")
    print(f"Missing labels: {len(missing_labels)}")
    
    if len(missing_labels) > 0:
        print(f"First 10 missing labels: {missing_labels[:10]}")
    
    if len(valid_pairs) < 10:
        raise ValueError(f"Too few valid pairs ({len(valid_pairs)}). Need at least 10 for training.")
    
    return valid_pairs

class FixedNaturalSemanticDataset(torch.utils.data.Dataset):
    """Fixed dataset that only uses validated image-label pairs."""
    
    def __init__(self, valid_pairs: List, split: str = 'train', image_size: int = 512):
        self.valid_pairs = valid_pairs
        self.image_size = image_size
        self.split = split
        
        # Split the valid pairs
        total_pairs = len(valid_pairs)
        if split == 'train':
            # Use first 80% for training
            end_idx = int(0.8 * total_pairs)
            self.pairs = valid_pairs[:end_idx]
        else:
            # Use last 20% for validation
            start_idx = int(0.8 * total_pairs)
            self.pairs = valid_pairs[start_idx:]
        
        print(f"Created {split} dataset with {len(self.pairs)} pairs")
        
    def rgb_to_class_index(self, rgb_label):
        """Convert RGB label to class indices."""
        h, w, c = rgb_label.shape
        class_label = np.zeros((h, w), dtype=np.uint8)
        
        for rgb, class_idx in RGB_TO_CLASS.items():
            mask = np.all(rgb_label == rgb, axis=2)
            class_label[mask] = class_idx
            
        return class_label
        
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        img_path, label_path = self.pairs[idx]
        
        import cv2
        
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError(f"Cannot read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load label
            label = cv2.imread(str(label_path))
            if label is None:
                raise ValueError(f"Cannot read label: {label_path}")
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
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            # Return a dummy sample to avoid crashing
            dummy_image = torch.zeros(3, self.image_size, self.image_size)
            dummy_label = torch.zeros(self.image_size, self.image_size, dtype=torch.long)
            return dummy_image, dummy_label

def calculate_class_weights(dataloader, num_classes=24):
    """Calculate class weights based on frequency."""
    class_counts = torch.zeros(num_classes)
    
    print("Calculating class weights...")
    total_batches = min(50, len(dataloader))  # Limit to prevent long wait
    
    for batch_idx, (_, labels) in enumerate(tqdm(dataloader, total=total_batches)):
        if batch_idx >= total_batches:
            break
            
        for class_idx in range(num_classes):
            class_counts[class_idx] += (labels == class_idx).sum()
    
    # Calculate weights (inverse frequency)
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (num_classes * class_counts + 1e-8)
    
    # Normalize weights and cap extreme values
    class_weights = torch.clamp(class_weights, min=0.1, max=10.0)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print("\nClass weights calculated:")
    for i, (name, weight) in enumerate(zip(CLASS_NAMES, class_weights)):
        count = class_counts[i].item()
        percentage = (count / total_pixels.item()) * 100 if total_pixels > 0 else 0
        print(f"{name:12}: weight={weight:.4f}, count={count:,} ({percentage:.2f}%)")
    
    return class_weights

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (images, labels) in enumerate(pbar):
        try:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                # Handle MMSeg model output format - model returns {'main': tensor}
                if isinstance(outputs, dict):
                    outputs = outputs.get('main', outputs.get('out', outputs.get('seg', outputs.get('pred', outputs))))
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
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    return total_loss / max(num_batches, 1)

def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    class_correct = torch.zeros(24)
    class_total = torch.zeros(24)
    num_batches = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            try:
                images = images.to(device)
                labels = labels.to(device)
                
                with autocast():
                    outputs = model(images)
                    # Handle MMSeg model output format - model returns {'main': tensor}
                    if isinstance(outputs, dict):
                        outputs = outputs.get('main', outputs.get('out', outputs.get('seg', outputs.get('pred', outputs))))
                    loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
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
                    
            except Exception as e:
                print(f"Error in validation batch: {e}")
                continue
    
    accuracy = correct_pixels / max(total_pixels, 1)
    avg_loss = total_loss / max(num_batches, 1)
    
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
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loader workers (set to 0 if having issues)')
    
    args = parser.parse_args()
    
    # Validate dataset first
    print("Step 1: Validating dataset...")
    try:
        valid_pairs = validate_dataset(args.data_root)
    except Exception as e:
        print(f"Dataset validation failed: {e}")
        return
    
    # Initialize wandb
    wandb.init(
        project="uav-natural-semantic-fixed",
        config=vars(args),
        name=f"natural_24class_fixed_{args.image_size}px"
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets using validated pairs
    print("Step 2: Creating datasets...")
    train_dataset = FixedNaturalSemanticDataset(valid_pairs, 'train', args.image_size)
    val_dataset = FixedNaturalSemanticDataset(valid_pairs, 'val', args.image_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Calculate class weights
    print("Step 3: Calculating class weights...")
    class_weights = calculate_class_weights(train_loader, 24)
    
    # Create model (24 classes)
    print("Step 4: Creating model...")
    model = MMSegBiSeNetV2(num_classes=24).to(device)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Training loop
    print("Step 5: Starting training...")
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
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'class_weights': class_weights,
                'valid_pairs_count': len(valid_pairs)
            }
            
            torch.save(checkpoint, 'outputs/natural_semantic_best_fixed.pth')
            print(f"New best model saved! Accuracy: {best_accuracy:.4f}")
    
    print(f"Training completed! Best accuracy: {best_accuracy:.4f}")
    print(f"Model saved to: outputs/natural_semantic_best_fixed.pth")

if __name__ == "__main__":
    main() 