#!/usr/bin/env python3
"""
Debug RGB Label Values
======================

Analyze actual RGB values in label images to fix class mapping issues.
"""

import cv2
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

def analyze_label_rgb_values(dataset_path, num_samples=10):
    """Analyze actual RGB values in label images."""
    
    dataset_path = Path(dataset_path)
    label_dir = dataset_path / "label_images_semantic"
    
    print(f"Analyzing RGB values in {label_dir}")
    
    # Get label files
    label_files = list(label_dir.glob("*.png"))
    print(f"Found {len(label_files)} label files")
    
    # Sample a few files
    sample_files = label_files[:num_samples]
    
    all_rgb_values = []
    
    for label_file in sample_files:
        print(f"\nAnalyzing: {label_file.name}")
        
        # Load label image
        label = cv2.imread(str(label_file))
        if label is None:
            print(f"  Cannot read {label_file}")
            continue
            
        # Convert BGR to RGB
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        
        # Get unique RGB values
        h, w, c = label.shape
        rgb_pixels = label.reshape(-1, 3)
        unique_rgb = np.unique(rgb_pixels, axis=0)
        
        print(f"  Image shape: {label.shape}")
        print(f"  Unique RGB values found: {len(unique_rgb)}")
        
        # Count occurrences
        rgb_tuples = [tuple(rgb) for rgb in rgb_pixels]
        rgb_counts = Counter(rgb_tuples)
        
        print("  Top 10 RGB values by frequency:")
        for rgb, count in rgb_counts.most_common(10):
            percentage = (count / len(rgb_tuples)) * 100
            print(f"    RGB{rgb}: {count:,} pixels ({percentage:.2f}%)")
        
        all_rgb_values.extend(unique_rgb)
    
    # Overall unique RGB values
    all_unique = np.unique(np.array(all_rgb_values), axis=0)
    print(f"\n=== OVERALL ANALYSIS ===")
    print(f"Total unique RGB values across {len(sample_files)} files: {len(all_unique)}")
    
    print(f"\nAll unique RGB values:")
    for rgb in all_unique:
        print(f"  {tuple(rgb)}")
    
    return all_unique

def compare_with_expected_mapping():
    """Compare found RGB values with expected mapping."""
    
    # Expected RGB to class mapping
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
    
    print(f"\n=== EXPECTED RGB VALUES ===")
    for rgb, class_idx in RGB_TO_CLASS.items():
        print(f"  Class {class_idx:2d}: RGB{rgb}")

def create_rgb_visualization(dataset_path, sample_file_idx=0):
    """Create a visualization of RGB values in a sample label image."""
    
    dataset_path = Path(dataset_path)
    label_dir = dataset_path / "label_images_semantic"
    label_files = list(label_dir.glob("*.png"))
    
    if sample_file_idx >= len(label_files):
        print(f"Sample index {sample_file_idx} out of range. Max: {len(label_files)-1}")
        return
    
    label_file = label_files[sample_file_idx]
    print(f"\nCreating visualization for: {label_file.name}")
    
    # Load and convert
    label = cv2.imread(str(label_file))
    label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original label image
    axes[0].imshow(label)
    axes[0].set_title(f"Label Image: {label_file.name}")
    axes[0].axis('off')
    
    # RGB value histogram
    h, w, c = label.shape
    rgb_pixels = label.reshape(-1, 3)
    rgb_tuples = [tuple(rgb) for rgb in rgb_pixels]
    rgb_counts = Counter(rgb_tuples)
    
    # Plot top 10 RGB values
    top_rgb = rgb_counts.most_common(10)
    colors = [f"#{r:02x}{g:02x}{b:02x}" for (r, g, b), _ in top_rgb]
    counts = [count for _, count in top_rgb]
    labels = [f"RGB{rgb}" for rgb, _ in top_rgb]
    
    axes[1].bar(range(len(top_rgb)), counts, color=colors)
    axes[1].set_title("Top 10 RGB Values")
    axes[1].set_xlabel("RGB Values")
    axes[1].set_ylabel("Pixel Count")
    axes[1].set_xticks(range(len(top_rgb)))
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('rgb_label_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved as: rgb_label_analysis.png")

def main():
    # Dataset path
    dataset_path = "datasets/Aerial_Semantic_Segmentation_Drone_Dataset/dataset/semantic_drone_dataset"
    
    print("🔍 RGB LABEL DEBUG ANALYSIS")
    print("=" * 50)
    
    # Analyze actual RGB values
    found_rgb = analyze_label_rgb_values(dataset_path, num_samples=5)
    
    # Compare with expected
    compare_with_expected_mapping()
    
    # Create visualization
    create_rgb_visualization(dataset_path, sample_file_idx=0)
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("1. Check if RGB values match exactly")
    print("2. Consider using nearest-neighbor RGB matching")
    print("3. Verify label image format (PNG vs JPG compression)")
    print("4. Check if labels need BGR->RGB conversion")

if __name__ == "__main__":
    main() 