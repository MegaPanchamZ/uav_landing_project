#!/usr/bin/env python3
"""
Quick Dataset Testing Script
===========================

Test script to manually examine dataset samples and verify class mapping.
"""

import torch
import numpy as np
from datasets.dronedeploy_1024_dataset import DroneDeploy1024Dataset, create_dronedeploy_datasets
import matplotlib.pyplot as plt
from collections import Counter

def analyze_raw_patches():
    """Analyze raw DroneDeploy dataset patches."""
    
    print("🔍 Analyzing DroneDeploy Dataset...")
    
    # Create dataset
    datasets = create_dronedeploy_datasets(
        data_root="../datasets/drone_deploy_dataset_intermediate/dataset-medium",
        patch_size=256,
        augmentation=False  # No augmentation for analysis
    )
    
    train_dataset = datasets['train']
    
    print(f"📊 Dataset size: {len(train_dataset)} patches")
    
    # Analyze first 100 samples
    class_counts = Counter()
    unique_values_seen = set()
    
    for i in range(min(100, len(train_dataset))):
        try:
            sample = train_dataset[i]
            mask = sample['mask'].numpy() if hasattr(sample['mask'], 'numpy') else sample['mask']
            
            # Count unique values in this mask
            unique_values = np.unique(mask)
            unique_values_seen.update(unique_values)
            
            # Count pixels per class
            for class_id in range(6):
                count = np.sum(mask == class_id)
                if count > 0:
                    class_counts[class_id] += count
                    
            # Print details for first few samples
            if i < 5:
                print(f"\nSample {i}:")
                print(f"  Mask shape: {mask.shape}")
                print(f"  Unique values in mask: {unique_values}")
                print(f"  Class distribution:")
                for class_id in range(6):
                    count = np.sum(mask == class_id)
                    if count > 0:
                        percentage = count / mask.size * 100
                        print(f"    Class {class_id}: {count} pixels ({percentage:.1f}%)")
                        
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    print(f"\n📊 Overall Analysis (100 samples):")
    print(f"All unique values seen in masks: {sorted(unique_values_seen)}")
    
    total_pixels = sum(class_counts.values())
    print(f"Total pixels analyzed: {total_pixels:,}")
    
    print(f"\nClass distribution:")
    class_names = {
        0: "ground", 1: "vegetation", 2: "building", 
        3: "water", 4: "car", 5: "clutter"
    }
    
    for class_id in range(6):
        count = class_counts[class_id]
        percentage = count / total_pixels * 100 if total_pixels > 0 else 0
        print(f"  {class_id} ({class_names[class_id]}): {count:,} pixels ({percentage:.2f}%)")


def test_dronedeploy_mapping():
    """Test the DroneDeploy class mapping directly."""
    
    print("\n🧪 Testing DroneDeploy Class Mapping...")
    
    # Check the mapping
    from datasets.dronedeploy_1024_dataset import DroneDeploy1024Dataset
    
    mapping = DroneDeploy1024Dataset.DRONEDEPLOY_TO_LANDING
    print("DroneDeploy → Landing mapping:")
    for dd_class, landing_class in mapping.items():
        print(f"  {dd_class} → {landing_class}")
    
    # Create a simple test to see if we can load raw data
    try:
        dataset = DroneDeploy1024Dataset(
            data_root="../datasets/drone_deploy_dataset_intermediate/dataset-medium",
            split="train",
            patch_size=256,
            augmentation=False
        )
        
        print(f"\nDataset loaded successfully: {len(dataset)} patches")
        
        # Try loading first sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            
            if 'mask' in sample:
                mask = sample['mask'].numpy() if hasattr(sample['mask'], 'numpy') else sample['mask']
                unique_values = np.unique(mask)
                print(f"Unique values in first mask: {unique_values}")
                
                # Check if values are being mapped correctly
                print(f"Expected landing classes: 0-5")
                print(f"Values outside expected range: {[v for v in unique_values if v < 0 or v > 5]}")
                
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    analyze_raw_patches()
    test_dronedeploy_mapping() 