#!/usr/bin/env python3
"""
Debug Original Labels - Semantic Drone Dataset Investigation
============================================================

Investigates the original Semantic Drone Dataset labels to understand:
1. What pixel values actually exist in the original labels
2. How our class mapping is affecting the distribution
3. Why Class 0 (background) is failing completely

This will help identify if the issue is:
- Wrong class mapping
- Missing classes in original dataset
- Preprocessing issues
- Model architecture problems
"""

import cv2
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

def investigate_original_dataset():
    """Investigate the original Semantic Drone Dataset labels."""
    
    # Dataset paths
    dataset_root = Path("H:/landing-system/datasets/Aerial_Semantic_Segmentation_Drone_Dataset/dataset/semantic_drone_dataset")
    images_dir = dataset_root / "original_images"
    labels_dir = dataset_root / "label_images_semantic"
    
    if not dataset_root.exists():
        print(f"❌ Dataset not found at {dataset_root}")
        return
    
    print("🔍 Investigating Original Semantic Drone Dataset")
    print("=" * 60)
    
    # Get all label files
    label_files = sorted(list(labels_dir.glob("*.png")))
    print(f"📊 Found {len(label_files)} label files")
    
    # Analyze first 20 files for detailed investigation
    sample_files = label_files[:20]
    
    # Track all unique pixel values across the dataset
    all_unique_values = set()
    class_distributions = []
    
    print("\n🔍 Analyzing sample label files...")
    
    for i, label_file in enumerate(tqdm(sample_files, desc="Processing")):
        # Load the label image
        label = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)
        
        if label is None:
            print(f"❌ Could not load {label_file}")
            continue
        
        # Get unique values and their counts
        unique_values, counts = np.unique(label, return_counts=True)
        
        # Add to global set
        all_unique_values.update(unique_values)
        
        # Store distribution for this image
        class_dist = dict(zip(unique_values, counts))
        class_distributions.append({
            'file': label_file.name,
            'shape': label.shape,
            'distribution': class_dist,
            'total_pixels': label.shape[0] * label.shape[1]
        })
        
        # Print detailed info for first few files
        if i < 5:
            print(f"\n📋 File: {label_file.name}")
            print(f"   Shape: {label.shape}")
            print(f"   Unique values: {sorted(unique_values)}")
            
            # Show percentage distribution
            total_pixels = label.shape[0] * label.shape[1]
            print("   Value distribution:")
            for val, count in zip(unique_values, counts):
                percentage = (count / total_pixels) * 100
                print(f"     Value {val:3d}: {count:8,} pixels ({percentage:6.2f}%)")
    
    print(f"\n📊 GLOBAL ANALYSIS")
    print(f"=" * 40)
    print(f"🎯 All unique pixel values found: {sorted(all_unique_values)}")
    print(f"🔢 Total number of classes: {len(all_unique_values)}")
    
    # Calculate global distribution
    global_distribution = Counter()
    total_global_pixels = 0
    
    for dist_info in class_distributions:
        for value, count in dist_info['distribution'].items():
            global_distribution[value] += count
            total_global_pixels += count
    
    print(f"\n🌐 Global Class Distribution:")
    sorted_classes = sorted(global_distribution.items(), key=lambda x: x[1], reverse=True)
    
    for value, count in sorted_classes:
        percentage = (count / total_global_pixels) * 100
        print(f"   Class {value:3d}: {count:9,} pixels ({percentage:6.2f}%)")
    
    # Analyze our class mapping
    print(f"\n🗺️  Our Class Mapping Analysis:")
    print(f"=" * 40)
    
    # Our mapping from ultra_fast_dataset.py
    class_mapping = {
        0: 0, 23: 0,  # background
        1: 1, 2: 1, 3: 1, 4: 1,  # safe_landing
        6: 2, 8: 2, 9: 2, 21: 2,  # caution
        5: 3, 7: 3, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3,
        15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3, 22: 3  # danger
    }
    
    # Check which original classes are mapped to each landing class
    mapped_distribution = {0: 0, 1: 0, 2: 0, 3: 0}  # landing classes
    unmapped_pixels = 0
    
    for original_class, pixel_count in sorted_classes:
        if original_class in class_mapping:
            landing_class = class_mapping[original_class]
            mapped_distribution[landing_class] += pixel_count
            print(f"   Original {original_class:3d} → Landing {landing_class} ({pixel_count:8,} pixels)")
        else:
            unmapped_pixels += pixel_count
            print(f"   Original {original_class:3d} → UNMAPPED ({pixel_count:8,} pixels)")
    
    print(f"\n🎯 Final Landing Class Distribution:")
    landing_class_names = {0: "Background", 1: "Safe Landing", 2: "Caution", 3: "Danger"}
    
    total_mapped_pixels = sum(mapped_distribution.values())
    for landing_class in range(4):
        count = mapped_distribution[landing_class]
        percentage = (count / total_mapped_pixels) * 100 if total_mapped_pixels > 0 else 0
        print(f"   Class {landing_class} ({landing_class_names[landing_class]:12}): {count:9,} pixels ({percentage:6.2f}%)")
    
    if unmapped_pixels > 0:
        unmapped_percentage = (unmapped_pixels / total_global_pixels) * 100
        print(f"   🚨 UNMAPPED: {unmapped_pixels:9,} pixels ({unmapped_percentage:6.2f}%)")
    
    # Check if class 0 (background) has any representation
    print(f"\n🚨 CRITICAL ISSUE DIAGNOSIS:")
    print(f"=" * 40)
    
    original_class_0_pixels = global_distribution.get(0, 0)
    original_class_23_pixels = global_distribution.get(23, 0)
    total_background_pixels = original_class_0_pixels + original_class_23_pixels
    
    print(f"🎯 Background class analysis:")
    print(f"   Original class 0:  {original_class_0_pixels:8,} pixels")
    print(f"   Original class 23: {original_class_23_pixels:8,} pixels")
    print(f"   Total background:  {total_background_pixels:8,} pixels")
    
    if total_background_pixels == 0:
        print(f"   🚨 FOUND THE ISSUE: No background pixels in original dataset!")
        print(f"   🔧 This explains why Class 0 has 0% performance")
    elif total_background_pixels < 1000:
        print(f"   ⚠️  SEVERE IMBALANCE: Background class extremely rare")
    
    # Suggest fixes
    print(f"\n💡 SUGGESTED FIXES:")
    print(f"=" * 20)
    
    if total_background_pixels == 0:
        print("1. 🔧 Remap class 0 to most common 'background-like' class")
        print("2. 🔧 Use 3-class classification (remove background)")
        print("3. 🔧 Create synthetic background data")
    
    # Show which classes are actually dominant
    print(f"\n📈 Most common classes (candidates for background remapping):")
    for i, (value, count) in enumerate(sorted_classes[:5]):
        percentage = (count / total_global_pixels) * 100
        print(f"   {i+1}. Class {value}: {count:8,} pixels ({percentage:5.2f}%)")
    
    # Create a simple visualization
    create_class_distribution_plot(sorted_classes, total_global_pixels)

def create_class_distribution_plot(sorted_classes, total_pixels):
    """Create a visualization of class distribution."""
    
    # Take top 15 classes for visibility
    top_classes = sorted_classes[:15]
    
    classes = [str(x[0]) for x in top_classes]
    counts = [x[1] for x in top_classes]
    percentages = [(count / total_pixels) * 100 for count in counts]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot of pixel counts
    bars1 = ax1.bar(classes, counts)
    ax1.set_title('Original Class Distribution (Pixel Counts)')
    ax1.set_xlabel('Original Class ID')
    ax1.set_ylabel('Pixel Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars1, counts):
        height = bar.get_height()
        ax1.annotate(f'{count:,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # Percentage plot
    bars2 = ax2.bar(classes, percentages)
    ax2.set_title('Original Class Distribution (Percentages)')
    ax2.set_xlabel('Original Class ID')
    ax2.set_ylabel('Percentage (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add percentage labels
    for bar, pct in zip(bars2, percentages):
        height = bar.get_height()
        ax2.annotate(f'{pct:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('original_class_distribution_debug.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Class distribution plot saved as 'original_class_distribution_debug.png'")

if __name__ == "__main__":
    investigate_original_dataset() 