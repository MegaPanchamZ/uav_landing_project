#!/usr/bin/env python3
"""
Visualization Tools for UAV Landing Detection Dataset and Training

This script provides visualization utilities to:
1. Visualize dataset samples
2. Monitor training progress
3. Evaluate model predictions
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import json
import torch
from pathlib import Path
import argparse
from practical_fine_tuning import DroneDeployDataset, SimpleBiSeNetV2, LANDING_CLASSES, DRONEDEPLOY_RGB_TO_LANDING

# Color map for visualization
LANDING_COLORS = {
    0: [0, 0, 0],        # background - black
    1: [0, 255, 0],      # suitable - green
    2: [255, 0, 0],      # obstacle - red  
    3: [255, 255, 0]     # unsafe - yellow
}

def visualize_dataset_samples(data_path, num_samples=6, save_path=None):
    """Visualize random samples from the dataset."""
    
    dataset = DroneDeployDataset(data_path, transform=None)
    
    if len(dataset) == 0:
        print("❌ No samples found in dataset!")
        return
        
    # Select random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # Create subplot layout
    rows = 2
    cols = min(num_samples, 3)
    fig, axes = plt.subplots(rows * 2, cols, figsize=(15, 10))
    fig.suptitle("DroneDeploy Dataset Samples", fontsize=16)
    
    for idx, sample_idx in enumerate(indices[:6]):
        if idx >= 6:
            break
            
        # Load sample
        image, label = dataset[sample_idx]
        
        # Convert tensors back to numpy if needed
        if torch.is_tensor(image):
            image = image.permute(1, 2, 0).numpy()
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
        if torch.is_tensor(label):
            label = label.numpy()
            
        # Create colored mask
        colored_mask = np.zeros((*label.shape, 3), dtype=np.uint8)
        for class_id, color in LANDING_COLORS.items():
            mask = (label == class_id)
            colored_mask[mask] = color
            
        # Plot image and mask
        row = (idx // cols) * 2
        col = idx % cols
        
        # Original image
        axes[row, col].imshow(image)
        axes[row, col].set_title(f"Sample {sample_idx} - Original")
        axes[row, col].axis('off')
        
        # Segmentation mask
        axes[row + 1, col].imshow(colored_mask)
        axes[row + 1, col].set_title(f"Sample {sample_idx} - Labels")
        axes[row + 1, col].axis('off')
        
    # Add legend
    legend_elements = [
        patches.Patch(color=[c/255.0 for c in color], label=f"{class_id}: {LANDING_CLASSES[class_id]}")
        for class_id, color in LANDING_COLORS.items()
    ]
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.15, 0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Dataset visualization saved to {save_path}")
    else:
        plt.show()
        
def plot_training_history(history_path, save_path=None):
    """Plot training history from saved JSON file."""
    
    if not Path(history_path).exists():
        print(f"❌ History file not found: {history_path}")
        return
        
    with open(history_path, 'r') as f:
        history = json.load(f)
        
    epochs = range(1, len(history['train_losses']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training loss
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Validation mIoU
    ax2.plot(epochs, history['val_mious'], 'r-', label='Validation mIoU')
    ax2.axhline(y=history['best_miou'], color='g', linestyle='--', label=f'Best mIoU: {history["best_miou"]:.4f}')
    ax2.set_title('Validation Mean IoU')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mIoU')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Training history plot saved to {save_path}")
    else:
        plt.show()
        
def visualize_model_predictions(model_path, data_path, num_samples=4, save_path=None):
    """Visualize model predictions on test samples."""
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleBiSeNetV2(num_classes=4)
    
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f" Loaded model from {model_path}")
    else:
        print(f"❌ Model file not found: {model_path}")
        return
        
    model.to(device)
    model.eval()
    
    # Load dataset
    dataset = DroneDeployDataset(data_path, transform=None)
    
    if len(dataset) == 0:
        print("❌ No samples found in dataset!")
        return
        
    # Select random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # Create subplot layout
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 10))
    fig.suptitle("Model Predictions vs Ground Truth", fontsize=16)
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            # Load sample
            image, true_label = dataset[sample_idx]
            
            # Prepare input
            if not torch.is_tensor(image):
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            else:
                image_tensor = image
                
            image_tensor = image_tensor.unsqueeze(0).to(device)
            
            # Get prediction
            pred_logits = model(image_tensor)
            if isinstance(pred_logits, tuple):
                pred_logits = pred_logits[0]
                
            pred_label = torch.argmax(pred_logits, dim=1).squeeze().cpu().numpy()
            
            # Convert for visualization
            if torch.is_tensor(image):
                vis_image = image.permute(1, 2, 0).numpy()
                if vis_image.max() <= 1.0:
                    vis_image = (vis_image * 255).astype(np.uint8)
            else:
                vis_image = image
                
            if torch.is_tensor(true_label):
                true_label = true_label.numpy()
                
            # Create colored masks
            true_mask = np.zeros((*true_label.shape, 3), dtype=np.uint8)
            pred_mask = np.zeros((*pred_label.shape, 3), dtype=np.uint8)
            
            for class_id, color in LANDING_COLORS.items():
                true_mask[true_label == class_id] = color
                pred_mask[pred_label == class_id] = color
                
            # Plot
            if num_samples == 1:
                axes[0].imshow(vis_image)
                axes[0].set_title("Original Image")
                axes[0].axis('off')
                
                axes[1].imshow(true_mask)
                axes[1].set_title("Ground Truth")
                axes[1].axis('off')
                
                axes[2].imshow(pred_mask)
                axes[2].set_title("Prediction")
                axes[2].axis('off')
            else:
                axes[0, idx].imshow(vis_image)
                axes[0, idx].set_title(f"Sample {sample_idx}")
                axes[0, idx].axis('off')
                
                axes[1, idx].imshow(true_mask)
                axes[1, idx].set_title("Ground Truth")
                axes[1, idx].axis('off')
                
                axes[2, idx].imshow(pred_mask)
                axes[2, idx].set_title("Prediction")
                axes[2, idx].axis('off')
                
    # Add legend
    legend_elements = [
        patches.Patch(color=[c/255.0 for c in color], label=f"{LANDING_CLASSES[class_id]}")
        for class_id, color in LANDING_COLORS.items()
    ]
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.1, 0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Prediction visualization saved to {save_path}")
    else:
        plt.show()

def analyze_dataset_statistics(data_path):
    """Analyze and display dataset statistics."""
    
    dataset = DroneDeployDataset(data_path, transform=None)
    
    if len(dataset) == 0:
        print("❌ No samples found in dataset!")
        return
        
    print(f"📊 Dataset Statistics for {data_path}")
    print("=" * 50)
    print(f"Total samples: {len(dataset)}")
    
    # Count pixels per class
    class_counts = {class_id: 0 for class_id in LANDING_CLASSES.keys()}
    total_pixels = 0
    
    print("Analyzing pixel distribution...")
    for i in range(min(len(dataset), 100)):  # Sample first 100 images
        _, label = dataset[i]
        if torch.is_tensor(label):
            label = label.numpy()
            
        unique, counts = np.unique(label, return_counts=True)
        for class_id, count in zip(unique, counts):
            if class_id in class_counts:
                class_counts[class_id] += count
                total_pixels += count
                
    print(f"\nClass distribution (from {min(len(dataset), 100)} samples):")
    for class_id, count in class_counts.items():
        percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
        print(f"  {class_id} ({LANDING_CLASSES[class_id]}): {count:,} pixels ({percentage:.1f}%)")
        
    # Visualize distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart
    classes = [LANDING_CLASSES[i] for i in range(4)]
    counts = [class_counts[i] for i in range(4)]
    colors = [[c/255.0 for c in LANDING_COLORS[i]] for i in range(4)]
    
    ax1.bar(classes, counts, color=colors)
    ax1.set_title('Class Distribution (Pixel Count)')
    ax1.set_ylabel('Number of Pixels')
    plt.setp(ax1.get_xticklabels(), rotation=45)
    
    # Pie chart
    ax2.pie(counts, labels=classes, colors=colors, autopct='%1.1f%%')
    ax2.set_title('Class Distribution (Percentage)')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualization tools for UAV landing detection")
    parser.add_argument("--action", type=str, choices=['dataset', 'history', 'predictions', 'stats'],
                       default='dataset', help="What to visualize")
    parser.add_argument("--data_path", type=str, 
                       default="../datasets/drone_deploy_dataset_intermediate/dataset-medium",
                       help="Path to dataset")
    parser.add_argument("--model_path", type=str,
                       help="Path to trained model (for predictions)")
    parser.add_argument("--history_path", type=str,
                       help="Path to training history JSON file")
    parser.add_argument("--num_samples", type=int, default=6,
                       help="Number of samples to visualize")
    parser.add_argument("--save_path", type=str,
                       help="Path to save visualization")
    
    args = parser.parse_args()
    
    if args.action == 'dataset':
        print("🖼️  Visualizing dataset samples...")
        visualize_dataset_samples(args.data_path, args.num_samples, args.save_path)
        
    elif args.action == 'history':
        if not args.history_path:
            args.history_path = "./fine_tuned_models/training_history.json"
        print(f"📈 Plotting training history from {args.history_path}...")
        plot_training_history(args.history_path, args.save_path)
        
    elif args.action == 'predictions':
        if not args.model_path:
            args.model_path = "./fine_tuned_models/best_model.pth"
        print(f"🔍 Visualizing model predictions from {args.model_path}...")
        visualize_model_predictions(args.model_path, args.data_path, args.num_samples, args.save_path)
        
    elif args.action == 'stats':
        print("📊 Analyzing dataset statistics...")
        analyze_dataset_statistics(args.data_path)

if __name__ == "__main__":
    main()
