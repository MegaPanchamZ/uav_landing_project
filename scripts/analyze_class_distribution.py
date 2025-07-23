#!/usr/bin/env python3
"""
Class Distribution Analysis for UAV Landing Dataset
=================================================

Analyzes class distribution to understand training issues,
particularly the complete failure of class 0 (background).
"""

import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import seaborn as sns
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets.ultra_fast_dataset import UltraFastSemanticDataset

def analyze_original_labels(data_root):
    """Analyze original label distribution before class mapping."""
    labels_dir = Path(data_root) / "label_images_semantic"
    label_files = sorted(list(labels_dir.glob("*.png")))
    
    print("🔍 Analyzing Original Label Distribution...")
    
    all_classes = set()
    class_counts = Counter()
    
    for label_file in tqdm(label_files[:50], desc="Analyzing labels"):  # Sample first 50
        label = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)
        unique_classes, counts = np.unique(label, return_counts=True)
        
        for cls, count in zip(unique_classes, counts):
            all_classes.add(cls)
            class_counts[cls] += count
    
    print(f"\n📊 Original Classes Found: {sorted(all_classes)}")
    print(f"📊 Total Pixels Analyzed: {sum(class_counts.values()):,}")
    
    # Sort by frequency
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\n🏷️  Class Distribution (Original):")
    total_pixels = sum(class_counts.values())
    for cls, count in sorted_classes:
        percentage = (count / total_pixels) * 100
        print(f"   Class {cls:2d}: {count:8,} pixels ({percentage:5.2f}%)")
    
    return class_counts, all_classes

def analyze_mapped_classes(data_root):
    """Analyze class distribution after mapping."""
    print("\n🔄 Analyzing Mapped Class Distribution...")
    
    # Class mapping from the dataset
    class_mapping = {
        0: 0, 23: 0,  # background
        1: 1, 2: 1, 3: 1, 4: 1,  # safe_landing
        6: 2, 8: 2, 9: 2, 21: 2,  # caution
        5: 3, 7: 3, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3,
        15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3, 22: 3  # danger
    }
    
    labels_dir = Path(data_root) / "label_images_semantic"
    label_files = sorted(list(labels_dir.glob("*.png")))
    
    mapped_counts = Counter()
    
    for label_file in tqdm(label_files[:50], desc="Mapping classes"):
        label = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)
        
        # Apply mapping
        mapped_label = np.zeros_like(label, dtype=np.uint8)
        for original_class, landing_class in class_mapping.items():
            mask = (label == original_class)
            mapped_label[mask] = landing_class
        
        unique_classes, counts = np.unique(mapped_label, return_counts=True)
        for cls, count in zip(unique_classes, counts):
            mapped_counts[cls] += count
    
    print("\n Mapped Class Distribution:")
    class_names = {0: "Background", 1: "Safe Landing", 2: "Caution", 3: "Danger"}
    total_pixels = sum(mapped_counts.values())
    
    for cls in range(4):
        count = mapped_counts.get(cls, 0)
        percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
        print(f"   Class {cls} ({class_names[cls]:12}): {count:8,} pixels ({percentage:5.2f}%)")
    
    return mapped_counts, class_mapping

def create_visualization(original_counts, mapped_counts):
    """Create visualization of class distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original distribution
    classes = list(original_counts.keys())
    counts = list(original_counts.values())
    
    ax1.bar(classes, counts)
    ax1.set_title("Original Class Distribution")
    ax1.set_xlabel("Original Class ID")
    ax1.set_ylabel("Pixel Count")
    ax1.tick_params(axis='x', rotation=45)
    
    # Mapped distribution
    mapped_classes = [0, 1, 2, 3]
    mapped_class_counts = [mapped_counts.get(cls, 0) for cls in mapped_classes]
    class_names = ["Background", "Safe Landing", "Caution", "Danger"]
    
    colors = ['gray', 'green', 'yellow', 'red']
    bars = ax2.bar(class_names, mapped_class_counts, color=colors, alpha=0.7)
    ax2.set_title("Mapped Class Distribution")
    ax2.set_xlabel("Landing Class")
    ax2.set_ylabel("Pixel Count")
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, mapped_class_counts):
        height = bar.get_height()
        ax2.annotate(f'{count:,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('class_distribution_analysis.png', dpi=300, bbox_inches='tight')
    print("\n📊 Visualization saved as 'class_distribution_analysis.png'")

def analyze_class_imbalance_severity(mapped_counts):
    """Analyze severity of class imbalance."""
    print("\n⚖️  Class Imbalance Analysis:")
    
    total_pixels = sum(mapped_counts.values())
    class_proportions = {}
    
    for cls in range(4):
        count = mapped_counts.get(cls, 0)
        proportion = count / total_pixels if total_pixels > 0 else 0
        class_proportions[cls] = proportion
    
    # Calculate imbalance ratio
    max_prop = max(class_proportions.values())
    min_prop = min([p for p in class_proportions.values() if p > 0])
    imbalance_ratio = max_prop / min_prop if min_prop > 0 else float('inf')
    
    print(f"   Imbalance Ratio: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio > 100:
        print("   ⚠️  SEVERE class imbalance detected!")
    elif imbalance_ratio > 10:
        print("   ⚠️  HIGH class imbalance detected!")
    else:
        print("    Moderate class imbalance.")
    
    # Recommend class weights
    print("\n🔧 Recommended Class Weights:")
    total_samples = sum(mapped_counts.values())
    n_classes = 4
    
    for cls in range(4):
        count = mapped_counts.get(cls, 0)
        if count > 0:
            weight = total_samples / (n_classes * count)
            print(f"   Class {cls}: {weight:.3f}")
        else:
            print(f"   Class {cls}: ∞ (no samples!)")
    
    return class_proportions

def main():
    # Get dataset path
    project_root = Path(__file__).parent.parent.parent
    data_root = project_root / 'datasets' / 'Aerial_Semantic_Segmentation_Drone_Dataset' / 'dataset' / 'semantic_drone_dataset'
    
    if not data_root.exists():
        print(f"❌ Dataset not found at {data_root}")
        return
    
    print("🔍 UAV Landing Dataset - Class Distribution Analysis")
    print("=" * 60)
    
    # Analyze original and mapped distributions
    original_counts, all_classes = analyze_original_labels(data_root)
    mapped_counts, class_mapping = analyze_mapped_classes(data_root)
    
    # Analyze imbalance severity
    class_proportions = analyze_class_imbalance_severity(mapped_counts)
    
    # Create visualization
    create_visualization(original_counts, mapped_counts)
    
    # Check for unmapped classes
    print("\n🔍 Unmapped Class Analysis:")
    mapped_original_classes = set(class_mapping.keys())
    unmapped_classes = all_classes - mapped_original_classes
    
    if unmapped_classes:
        print(f"   ⚠️  Unmapped classes found: {sorted(unmapped_classes)}")
        
        # Calculate pixels in unmapped classes
        unmapped_pixels = sum(original_counts[cls] for cls in unmapped_classes)
        total_pixels = sum(original_counts.values())
        unmapped_percentage = (unmapped_pixels / total_pixels) * 100
        print(f"   📊 Unmapped pixels: {unmapped_pixels:,} ({unmapped_percentage:.2f}%)")
    else:
        print("    All classes are mapped.")
    
    # Summary
    print("\n📋 Analysis Summary:")
    print("=" * 30)
    
    total_mapped = sum(mapped_counts.values())
    for cls in range(4):
        count = mapped_counts.get(cls, 0)
        percentage = (count / total_mapped) * 100 if total_mapped > 0 else 0
        
        if percentage == 0:
            print(f"   🚨 Class {cls}: NO SAMPLES - This explains training failure!")
        elif percentage < 1:
            print(f"   ⚠️  Class {cls}: Very rare ({percentage:.3f}%)")
        elif percentage < 5:
            print(f"   ⚠️  Class {cls}: Rare ({percentage:.1f}%)")
        else:
            print(f"    Class {cls}: {percentage:.1f}%")

if __name__ == "__main__":
    main() 