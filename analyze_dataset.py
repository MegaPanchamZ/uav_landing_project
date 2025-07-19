#!/usr/bin/env python3
"""
Dataset Analysis for UAV Landing Zone Detection

Analyze the DroneDeploy dataset to understand:
1. Label distribution and class balance
2. Image characteristics and quality
3. Dataset size and diversity
4. Potential issues causing poor training performance
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict, Counter
import seaborn as sns

def analyze_labels(labels_dir):
    """Analyze the label distribution across all images."""
    
    print("🔍 Analyzing Label Distribution")
    print("=" * 35)
    
    label_files = list(Path(labels_dir).glob("*.png"))
    print(f"Found {len(label_files)} label files")
    
    # RGB to class mapping (from training)
    rgb_to_class = {
        (0, 255, 0): 1,    # Green = suitable
        (255, 0, 0): 2,    # Red = obstacle
        (255, 255, 0): 3,  # Yellow = unsafe
        (0, 0, 0): 0       # Black = background
    }
    
    total_pixels_per_class = defaultdict(int)
    images_with_class = defaultdict(int)
    class_percentages_per_image = []
    
    for i, label_file in enumerate(label_files[:10]):  # Sample first 10 for speed
        print(f"  Analyzing {label_file.name}...")
        
        # Load label image
        label_img = cv2.imread(str(label_file))
        if label_img is None:
            continue
            
        label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
        h, w = label_img.shape[:2]
        total_pixels = h * w
        
        # Count pixels per class
        image_class_counts = defaultdict(int)
        
        # Convert to class map
        for rgb, class_id in rgb_to_class.items():
            mask = np.all(label_img == rgb, axis=2)
            count = np.sum(mask)
            image_class_counts[class_id] = count
            total_pixels_per_class[class_id] += count
            
            if count > 0:
                images_with_class[class_id] += 1
        
        # Calculate percentages for this image
        image_percentages = {}
        for class_id in range(4):
            pct = (image_class_counts[class_id] / total_pixels) * 100
            image_percentages[class_id] = pct
            
        class_percentages_per_image.append(image_percentages)
    
    # Print overall statistics
    total_pixels = sum(total_pixels_per_class.values())
    
    print(f"\n📊 Overall Dataset Statistics (sample of {len(class_percentages_per_image)} images):")
    print(f"Total pixels analyzed: {total_pixels:,}")
    
    class_names = {0: "background", 1: "suitable", 2: "obstacle", 3: "unsafe"}
    
    for class_id in range(4):
        count = total_pixels_per_class[class_id]
        percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
        images_count = images_with_class[class_id]
        
        print(f"  {class_names[class_id]:10}: {count:8,} pixels ({percentage:5.1f}%) in {images_count} images")
    
    return class_percentages_per_image, total_pixels_per_class

def analyze_images(images_dir):
    """Analyze image characteristics."""
    
    print(f"\n📸 Analyzing Image Characteristics")
    print("=" * 35)
    
    image_files = list(Path(images_dir).glob("*.tif"))
    
    sizes = []
    brightness_stats = []
    
    for i, img_file in enumerate(image_files[:10]):  # Sample first 10
        img = cv2.imread(str(img_file))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        sizes.append((w, h))
        
        # Convert to grayscale for brightness analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness_stats.append({
            'mean': np.mean(gray),
            'std': np.std(gray),
            'min': np.min(gray),
            'max': np.max(gray)
        })
    
    # Print image statistics
    if sizes:
        unique_sizes = list(set(sizes))
        print(f"Image count: {len(image_files)}")
        print(f"Image sizes: {unique_sizes}")
        
        if brightness_stats:
            mean_brightness = np.mean([s['mean'] for s in brightness_stats])
            mean_contrast = np.mean([s['std'] for s in brightness_stats])
            print(f"Average brightness: {mean_brightness:.1f}")
            print(f"Average contrast (std): {mean_contrast:.1f}")
    
    return sizes, brightness_stats

def identify_issues(class_percentages, total_pixels_per_class):
    """Identify potential training issues."""
    
    print(f"\n⚠️  Potential Training Issues")
    print("=" * 30)
    
    issues = []
    
    # Check class imbalance
    total_pixels = sum(total_pixels_per_class.values())
    class_names = {0: "background", 1: "suitable", 2: "obstacle", 3: "unsafe"}
    
    for class_id in range(4):
        percentage = (total_pixels_per_class[class_id] / total_pixels) * 100 if total_pixels > 0 else 0
        
        if percentage < 1:
            issues.append(f"Class '{class_names[class_id]}' is severely underrepresented ({percentage:.2f}%)")
        elif percentage > 80:
            issues.append(f"Class '{class_names[class_id]}' dominates the dataset ({percentage:.1f}%)")
    
    # Check dataset size
    if len(class_percentages) < 100:
        issues.append(f"Very small dataset ({len(class_percentages)} images analyzed)")
    
    # Check class presence
    for class_id in range(1, 4):  # Skip background
        images_with_class = sum(1 for img in class_percentages if img[class_id] > 0)
        if images_with_class < len(class_percentages) * 0.3:
            issues.append(f"Class '{class_names[class_id]}' appears in very few images ({images_with_class}/{len(class_percentages)})")
    
    if issues:
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ No obvious issues detected")
    
    return issues

def suggest_improvements(issues):
    """Suggest improvements based on identified issues."""
    
    print(f"\n💡 Suggested Improvements")
    print("=" * 25)
    
    suggestions = []
    
    if any("underrepresented" in issue for issue in issues):
        suggestions.append("Use weighted loss function to handle class imbalance")
        suggestions.append("Implement oversampling for minority classes")
        suggestions.append("Consider focal loss for imbalanced classes")
    
    if any("small dataset" in issue for issue in issues):
        suggestions.append("Use heavy data augmentation (rotation, flip, color jitter)")
        suggestions.append("Try transfer learning with frozen backbone")
        suggestions.append("Consider few-shot learning approaches")
        suggestions.append("Use classical computer vision as baseline")
    
    if any("dominates" in issue for issue in issues):
        suggestions.append("Consider binary classification first (safe vs unsafe)")
        suggestions.append("Use stratified sampling during training")
    
    if any("appears in very few" in issue for issue in issues):
        suggestions.append("Focus on simpler model architecture")
        suggestions.append("Use patch-based classification instead of full segmentation")
    
    # General suggestions
    suggestions.extend([
        "Start with classical CV baseline (color thresholding, morphology)",
        "Use smaller learning rate and longer training",
        "Implement early stopping with validation monitoring",
        "Try ensemble of simple models"
    ])
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i:2d}. {suggestion}")

def main():
    """Main analysis function."""
    
    print("🚁 UAV Dataset Analysis")
    print("=" * 25)
    
    # Paths
    dataset_path = Path("../datasets/drone_deploy_dataset_intermediate/dataset-medium")
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    # Analyze labels
    class_percentages, total_pixels_per_class = analyze_labels(labels_dir)
    
    # Analyze images  
    sizes, brightness_stats = analyze_images(images_dir)
    
    # Identify issues
    issues = identify_issues(class_percentages, total_pixels_per_class)
    
    # Suggest improvements
    suggest_improvements(issues)
    
    print(f"\n🎯 Quick Fix Recommendation:")
    print("=" * 30)
    print("1. Start with color-based classical CV")
    print("2. Use it as baseline and for data validation") 
    print("3. If ML needed, try patch classification with heavy augmentation")
    print("4. Focus on binary safe/unsafe classification first")

if __name__ == "__main__":
    main()
