#!/usr/bin/env python3
"""
Analyze Semantic Drone Dataset Classes
=====================================

This script analyzes the actual classes present in the dataset and validates
our class mapping for UAV landing detection.
"""

import cv2
import numpy as np
from pathlib import Path
from collections import Counter
import sys

def analyze_dataset():
    """Analyze the actual dataset structure and classes."""
    
    # Dataset paths
    dataset_root = Path(__file__).parent.parent.parent / 'datasets' / 'Aerial_Semantic_Segmentation_Drone_Dataset'
    images_dir = dataset_root / 'dataset' / 'semantic_drone_dataset' / 'original_images'
    labels_dir = dataset_root / 'dataset' / 'semantic_drone_dataset' / 'label_images_semantic'
    
    # Official class definitions from CSV
    csv_classes = {
        0: 'unlabeled', 1: 'paved-area', 2: 'dirt', 3: 'grass', 4: 'gravel',
        5: 'water', 6: 'rocks', 7: 'pool', 8: 'vegetation', 9: 'roof',
        10: 'wall', 11: 'window', 12: 'door', 13: 'fence', 14: 'fence-pole',
        15: 'person', 16: 'dog', 17: 'car', 18: 'bicycle', 19: 'tree',
        20: 'bald-tree', 21: 'ar-marker', 22: 'obstacle', 23: 'conflicting'
    }
    
    print("🔍 Analyzing Semantic Drone Dataset Structure")
    print("=" * 60)
    
    # Check basic structure
    print(f"Dataset root: {dataset_root}")
    print(f"Images dir exists: {images_dir.exists()}")
    print(f"Labels dir exists: {labels_dir.exists()}")
    
    if not images_dir.exists() or not labels_dir.exists():
        print("❌ Dataset directories not found!")
        return
    
    # Get file lists
    image_files = sorted(list(images_dir.glob("*.jpg")))
    label_files = sorted(list(labels_dir.glob("*.png")))
    
    print(f"Image files: {len(image_files)}")
    print(f"Label files: {len(label_files)}")
    
    # Analyze class distribution across multiple images
    all_classes = Counter()
    sample_count = min(20, len(label_files))  # Analyze first 20 images
    
    print(f"\n📊 Analyzing classes in {sample_count} sample images...")
    
    for i, label_file in enumerate(label_files[:sample_count]):
        if i % 5 == 0:
            print(f"Processing image {i+1}/{sample_count}...")
            
        label = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)
        unique_values, counts = np.unique(label, return_counts=True)
        
        for class_id, count in zip(unique_values, counts):
            all_classes[class_id] += count
    
    print(f"\n Classes found across {sample_count} images:")
    print("-" * 60)
    
    total_pixels = sum(all_classes.values())
    for class_id in sorted(all_classes.keys()):
        class_name = csv_classes.get(class_id, f"unknown_{class_id}")
        count = all_classes[class_id]
        percentage = (count / total_pixels) * 100
        print(f"  {class_id:2d}: {class_name:15s} - {percentage:6.2f}% ({count:,} pixels)")
    
    # Our current mapping for UAV landing
    our_mapping = {
        # Background/Unknown
        0: 0, 23: 0,  # unlabeled, conflicting
        
        # Safe Landing - flat, stable surfaces  
        1: 1, 2: 1, 3: 1, 4: 1,  # paved-area, dirt, grass, gravel
        
        # Caution - potentially suitable, needs assessment
        6: 2, 8: 2, 9: 2, 21: 2,  # rocks, vegetation, roof, ar-marker
        
        # Danger - obstacles, hazards, unsuitable
        5: 3, 7: 3, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3,  # water, pool, wall, window, door, fence, fence-pole
        15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3, 22: 3   # person, dog, car, bicycle, tree, bald-tree, obstacle
    }
    
    landing_classes = {0: "background", 1: "safe_landing", 2: "caution", 3: "danger"}
    
    print(f"\n Our UAV Landing Class Mapping:")
    print("-" * 60)
    
    for landing_class in range(4):
        class_name = landing_classes[landing_class]
        original_classes = [k for k, v in our_mapping.items() if v == landing_class]
        
        print(f"\n{landing_class}: {class_name.upper()}")
        total_pixels_this_class = 0
        
        for orig_class in original_classes:
            if orig_class in all_classes:
                count = all_classes[orig_class]
                total_pixels_this_class += count
                orig_name = csv_classes.get(orig_class, f"unknown_{orig_class}")
                percentage = (count / total_pixels) * 100
                print(f"    {orig_class:2d}: {orig_name:15s} - {percentage:5.2f}%")
        
        class_percentage = (total_pixels_this_class / total_pixels) * 100
        print(f"    --> Total for {class_name}: {class_percentage:.2f}%")
    
    # Check for unmapped classes
    unmapped = set(all_classes.keys()) - set(our_mapping.keys())
    if unmapped:
        print(f"\n⚠️  Unmapped classes found: {unmapped}")
        for class_id in unmapped:
            class_name = csv_classes.get(class_id, f"unknown_{class_id}")
            count = all_classes[class_id]
            percentage = (count / total_pixels) * 100
            print(f"    {class_id}: {class_name} - {percentage:.2f}%")
    
    print(f"\n Analysis complete!")

if __name__ == "__main__":
    analyze_dataset() 