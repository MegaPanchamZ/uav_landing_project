#!/usr/bin/env python3
"""
Test Dataset Loading Speed
==========================

Quick test to compare loading speeds between different dataset configurations.
"""

import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets.semantic_drone_dataset import SemanticDroneDataset, create_semantic_drone_transforms

def test_dataset_speed():
    """Test loading speed of different configurations."""
    
    dataset_path = Path(__file__).parent.parent.parent / 'datasets' / 'Aerial_Semantic_Segmentation_Drone_Dataset' / 'dataset' / 'semantic_drone_dataset'
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        return
    
    print("🚀 Testing Dataset Loading Speeds\n")
    
    # Test 1: Random crops (NEW - optimized)
    print("1️⃣ Testing with random crops (4 crops per image)...")
    start_time = time.time()
    
    transforms = create_semantic_drone_transforms(
        input_size=(512, 512),
        is_training=True,
        advanced_augmentation=False,
        use_resize=False
    )
    
    dataset_crops = SemanticDroneDataset(
        data_root=str(dataset_path),
        split="train",
        transform=transforms,
        use_random_crops=True,
        crops_per_image=4,
        cache_images=True
    )
    
    loader_crops = DataLoader(dataset_crops, batch_size=8, shuffle=True, num_workers=4)
    
    # Load 3 batches
    batch_count = 0
    for batch in loader_crops:
        batch_count += 1
        if batch_count >= 3:
            break
    
    crop_time = time.time() - start_time
    print(f"    Random crops: {crop_time:.2f}s for 3 batches ({len(dataset_crops)} total samples)")
    
    # Test 2: Full image resize (OLD - slow)
    print("\n2️⃣ Testing with full image resize...")
    start_time = time.time()
    
    transforms_resize = create_semantic_drone_transforms(
        input_size=(512, 512),
        is_training=True,
        advanced_augmentation=False,
        use_resize=True
    )
    
    dataset_resize = SemanticDroneDataset(
        data_root=str(dataset_path),
        split="train",
        transform=transforms_resize,
        use_random_crops=False,
        crops_per_image=1,
        cache_images=True
    )
    
    loader_resize = DataLoader(dataset_resize, batch_size=8, shuffle=True, num_workers=4)
    
    # Load 3 batches
    batch_count = 0
    for batch in loader_resize:
        batch_count += 1
        if batch_count >= 3:
            break
    
    resize_time = time.time() - start_time
    print(f"    Full resize: {resize_time:.2f}s for 3 batches ({len(dataset_resize)} total samples)")
    
    # Summary
    print(f"\n📊 Speed Comparison:")
    print(f"   Random crops: {crop_time:.2f}s")
    print(f"   Full resize:  {resize_time:.2f}s")
    speedup = resize_time / crop_time if crop_time > 0 else 0
    print(f"   Speedup: {speedup:.1f}x faster with random crops!")
    print(f"\n💡 Effective dataset size with crops: {len(dataset_crops)} vs {len(dataset_resize)}")

if __name__ == "__main__":
    test_dataset_speed() 