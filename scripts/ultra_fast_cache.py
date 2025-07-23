#!/usr/bin/env python3
"""
Ultra-Fast Cache Generation
===========================

Quick cache generation with minimal factors for immediate testing.
Perfect for when you want to test the training pipeline quickly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from datasets.drone_deploy_dataset import DroneDeployDataset
from datasets.udd_dataset import UDDDataset
from datasets.semantic_drone_dataset import SemanticDroneDataset
from datasets.cached_augmentation import create_cached_datasets

def main():
    print("🚀 Ultra-Fast Cache Generation")
    print("=" * 50)
    print("⚡ Using minimal augmentation factors for speed:")
    print("   - Semantic Drone: 3x")
    print("   - DroneDeploy: 2x")
    print("   - UDD: 2x")
    print("   - Expected time: ~2-5 minutes")
    
    start_time = time.time()
    
    # Load base datasets
    print(f"\n📂 Loading base datasets...")
    
    train_drone_deploy = DroneDeployDataset(
        data_root='../datasets/drone_deploy_dataset_intermediate/dataset-medium',
        split='train',
        transform=None,
        use_height=True
    )
    
    train_udd = UDDDataset(
        data_root='../datasets/UDD/UDD/UDD5',
        split='train',
        transform=None
    )
    
    train_semantic = SemanticDroneDataset(
        data_root='../datasets/Aerial_Semantic_Segmentation_Drone_Dataset/dataset/semantic_drone_dataset',
        split='train',
        transform=None,
        class_mapping='enhanced_4_class'
    )
    
    print(f" Base datasets loaded:")
    print(f"   - DroneDeploy: {len(train_drone_deploy)} images")
    print(f"   - UDD: {len(train_udd)} images")
    print(f"   - Semantic Drone: {len(train_semantic)} images")
    
    # Ultra-minimal augmentation factors
    augmentation_factors = {
        'semantic_drone': 3,  # Only 3x for speed
        'drone_deploy': 2,    # Only 2x for speed
        'udd': 2             # Only 2x for speed
    }
    
    expected_patches = {
        'semantic_drone': len(train_semantic) * 3,
        'drone_deploy': len(train_drone_deploy) * 2,
        'udd': len(train_udd) * 2
    }
    
    total_expected = sum(expected_patches.values())
    print(f"\n📈 Expected patch counts (minimal):")
    for name, count in expected_patches.items():
        print(f"   - {name}: ~{count} patches")
    print(f"   - Total: ~{total_expected} patches")
    
    # Create cached datasets with ultra-fast settings
    print(f"\n🚀 Creating cached datasets with factors: {augmentation_factors}")
    cached_datasets = create_cached_datasets(
        drone_deploy_dataset=train_drone_deploy,
        udd_dataset=train_udd,
        semantic_drone_dataset=train_semantic,
        cache_root="cache/ultra_fast_datasets",
        augmentation_factors=augmentation_factors,
        force_rebuild=True  # Always rebuild for ultra-fast mode
    )
    
    # Summary
    total_time = time.time() - start_time
    total_base = len(train_drone_deploy) + len(train_udd) + len(train_semantic)
    total_cached = sum(len(dataset) for dataset in cached_datasets.values())
    
    print(f"\n🎉 ULTRA-FAST CACHE COMPLETE!")
    print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"📊 Statistics:")
    print(f"   - Base images: {total_base}")
    print(f"   - Cached patches: {total_cached}")
    print(f"   - Augmentation factor: {total_cached/total_base:.1f}x")
    
    print(f"\n🚀 Ready for fast training!")
    print(f"Use: python scripts/progressive_training.py --use-cached-augmentation --cache-dir cache/ultra_fast_datasets")

if __name__ == "__main__":
    main() 