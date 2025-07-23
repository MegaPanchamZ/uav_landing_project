#!/usr/bin/env python3
"""
Simple Segmentation Demo
=======================

A simplified working demo with standard segmentation model to show actual results.
Uses DeepLabV3 which is known to work well out of the box.
"""

import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import segmentation
from torch.utils.data import DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import wandb

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Use the same RGB to class mapping from our fixed script
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

class SimpleSemanticDataset(torch.utils.data.Dataset):
    """Simplified dataset for demo."""
    
    def __init__(self, data_root: str, split: str = 'train', image_size: int = 256):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.split = split
        
        # Get image paths
        image_dir = self.data_root / "original_images"
        label_dir = self.data_root / "label_images_semantic"
        
        image_files = list(image_dir.glob("*.jpg"))
        
        # Validate and get valid pairs (same logic as fixed script)
        valid_pairs = []
        for img_path in image_files:
            label_path = label_dir / (img_path.stem + ".png")
            if label_path.exists():
                valid_pairs.append((img_path, label_path))
        
        # Split dataset
        total_pairs = len(valid_pairs)
        if split == 'train':
            self.pairs = valid_pairs[:int(0.8 * total_pairs)]
        else:
            self.pairs = valid_pairs[int(0.8 * total_pairs):]
        
        print(f"Simple dataset {split}: {len(self.pairs)} pairs")
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
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
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load label
        label = cv2.imread(str(label_path))
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        label = cv2.resize(label, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        # Convert label to class indices
        label = self.rgb_to_class_index(label)
        
        # Transform image
        image = self.transform(image)
        label = torch.from_numpy(label).long()
        
        return image, label

def create_simple_model(num_classes=24):
    """Create a simple DeepLabV3 model."""
    model = segmentation.deeplabv3_resnet50(pretrained=True)
    
    # Replace classifier for our number of classes
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    
    return model

def train_simple_demo():
    """Train the simple demo."""
    
    # Initialize wandb
    wandb.init(project="uav-simple-demo", name="deeplabv3_demo")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    data_root = r'H:\landing-system\datasets\Aerial_Semantic_Segmentation_Drone_Dataset\dataset\semantic_drone_dataset'
    
    train_dataset = SimpleSemanticDataset(data_root, 'train', 256)
    val_dataset = SimpleSemanticDataset(data_root, 'val', 256)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # Model
    model = create_simple_model(24).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore unlabeled class
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    epochs = 3
    best_acc = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Training")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)['out']  # DeepLabV3 returns dict with 'out' key
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy (ignoring unlabeled class)
            preds = torch.argmax(outputs, dim=1)
            mask = labels != 0  # Ignore unlabeled pixels
            if mask.sum() > 0:
                train_correct += (preds[mask] == labels[mask]).sum().item()
                train_total += mask.sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        class_correct = torch.zeros(24)
        class_total = torch.zeros(24)
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)['out']
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                mask = labels != 0  # Ignore unlabeled pixels
                if mask.sum() > 0:
                    val_correct += (preds[mask] == labels[mask]).sum().item()
                    val_total += mask.sum().item()
                
                # Per-class accuracy
                for class_idx in range(1, 24):  # Skip unlabeled class
                    class_mask = (labels == class_idx)
                    if class_mask.sum() > 0:
                        class_correct[class_idx] += (preds[class_mask] == class_idx).sum().item()
                        class_total[class_idx] += class_mask.sum().item()
        
        # Calculate metrics
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        
        # Per-class accuracies
        class_accs = class_correct / (class_total + 1e-8)
        mean_class_acc = class_accs[1:].mean()  # Exclude unlabeled class
        
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
        print(f"Mean Class Acc: {mean_class_acc:.4f}")
        
        # Print top performing classes
        non_zero_classes = [(i, acc.item()) for i, acc in enumerate(class_accs) if class_total[i] > 0 and i > 0]
        non_zero_classes.sort(key=lambda x: x[1], reverse=True)
        
        print("Top performing classes:")
        for class_idx, acc in non_zero_classes[:5]:
            print(f"  {CLASS_NAMES[class_idx]}: {acc:.4f} ({class_total[class_idx]:,} pixels)")
        
        # Log to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss/len(train_loader),
            'train_acc': train_acc,
            'val_loss': val_loss/len(val_loader),
            'val_acc': val_acc,
            'mean_class_acc': mean_class_acc
        })
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'mean_class_acc': mean_class_acc.item()
            }, 'outputs/simple_demo_best.pth')
            print(f"New best model saved! Val Acc: {val_acc:.4f}")
    
    print(f"\nTraining completed! Best validation accuracy: {best_acc:.4f}")

def create_prediction_visualization():
    """Create a visualization of model predictions."""
    print("Creating prediction visualization...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = create_simple_model(24).to(device)
    checkpoint = torch.load('outputs/simple_demo_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load a test image
    data_root = r'H:\landing-system\datasets\Aerial_Semantic_Segmentation_Drone_Dataset\dataset\semantic_drone_dataset'
    test_dataset = SimpleSemanticDataset(data_root, 'val', 256)
    
    # Get first image
    image, label = test_dataset[0]
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)['out']
        prediction = torch.argmax(output, dim=1).cpu().numpy()[0]
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image (denormalize)
    img_np = image.cpu().numpy()[0]
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_np = img_np * std + mean
    img_np = np.clip(img_np.transpose(1, 2, 0), 0, 1)
    
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(label.numpy(), cmap='tab20', vmin=0, vmax=23)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(prediction, cmap='tab20', vmin=0, vmax=23)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/simple_demo_prediction.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to outputs/simple_demo_prediction.png")

if __name__ == "__main__":
    print("🛠️ Starting Simple Segmentation Demo")
    train_simple_demo()
    create_prediction_visualization()
    print("✅ Simple demo completed!") 