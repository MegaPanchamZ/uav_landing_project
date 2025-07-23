#!/usr/bin/env python3
"""
Create Augmented Dataset Cache
=============================

One-time script to generate and cache augmented datasets for fast training.
This creates comprehensive augmented datasets that can be reused instantly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
import time

from datasets.drone_deploy_dataset import DroneDeployDataset
from datasets.udd_dataset import UDDDataset
from datasets.semantic_drone_dataset import SemanticDroneDataset
from datasets.cached_augmentation import create_cached_datasets


def main():
    parser = argparse.ArgumentParser(description='Create cached augmented datasets')
    
    # Dataset paths
    parser.add_argument('--drone-deploy-path', type=str,
                        default='../datasets/drone_deploy_dataset_intermediate/dataset-medium',
                        help='Path to DroneDeploy dataset')
    parser.add_argument('--udd-path', type=str,
                        default='../datasets/UDD/UDD/UDD5',
                        help='Path to UDD dataset')
    parser.add_argument('--semantic-drone-path', type=str,
                        default='../datasets/Aerial_Semantic_Segmentation_Drone_Dataset/dataset/semantic_drone_dataset',
                        help='Path to Semantic Drone dataset')
    
    # Cache configuration
    parser.add_argument('--cache-dir', type=str, default='cache/augmented_datasets',
                        help='Directory to store cached datasets')
    parser.add_argument('--force-rebuild', action='store_true',
                        help='Force rebuilding existing cache')
    
    # Augmentation factors
    parser.add_argument('--semantic-factor', type=int, default=25,
                        help='Augmentation factor for Semantic Drone dataset')
    parser.add_argument('--drone-deploy-factor', type=int, default=20,
                        help='Augmentation factor for DroneDeploy dataset')
    parser.add_argument('--udd-factor', type=int, default=15,
                        help='Augmentation factor for UDD dataset')
    parser.add_argument('--fast-mode', action='store_true',
                        help='Use reduced factors for faster cache generation (10x, 8x, 6x)')
    parser.add_argument('--ultra-fast-mode', action='store_true',
                        help='Use minimal factors for ultra-fast cache generation (5x, 4x, 3x)')
    
    # Processing
    parser.add_argument('--num-workers', type=int, default=16,
                        help='Number of worker threads (default: 16 for high-core systems)')
    
    args = parser.parse_args()
    
    print("🚀 Creating Augmented Dataset Cache")
    print("=" * 60)
    print(f"📁 Cache directory: {args.cache_dir}")
    print(f"🔧 Force rebuild: {args.force_rebuild}")
    print(f"👥 Workers: {args.num_workers}")
    print(f"📊 Augmentation factors:")
    print(f"   - Semantic Drone: {args.semantic_factor}x")
    print(f"   - DroneDeploy: {args.drone_deploy_factor}x")
    print(f"   - UDD: {args.udd_factor}x")
    
    start_time = time.time()
    
    # Load base datasets
    print(f"\n📂 Loading base datasets...")
    
    # DroneDeploy with height maps (4-channel)
    train_drone_deploy = DroneDeployDataset(
        data_root=args.drone_deploy_path,
        split='train',
        transform=None,  # Raw data for augmentation
        use_height=True
    )
    
    # UDD dataset (3-channel)
    train_udd = UDDDataset(
        data_root=args.udd_path,
        split='train',
        transform=None  # Raw data for augmentation
    )
    
    # Semantic Drone dataset (3-channel)
    train_semantic = SemanticDroneDataset(
        data_root=args.semantic_drone_path,
        split='train',
        transform=None,  # Raw data for augmentation
        class_mapping='enhanced_4_class'
    )
    
    print(f" Base datasets loaded:")
    print(f"   - DroneDeploy: {len(train_drone_deploy)} images (4-channel)")
    print(f"   - UDD: {len(train_udd)} images (3-channel)")
    print(f"   - Semantic Drone: {len(train_semantic)} images (3-channel)")
    
    # Create augmentation factors with speed mode overrides
    if args.ultra_fast_mode:
        augmentation_factors = {
            'semantic_drone': 5,
            'drone_deploy': 4,
            'udd': 3
        }
        print("🚀 ULTRA-FAST MODE: Using minimal augmentation factors")
    elif args.fast_mode:
        augmentation_factors = {
            'semantic_drone': 10,
            'drone_deploy': 8,
            'udd': 6
        }
        print("⚡ FAST MODE: Using reduced augmentation factors")
    else:
        augmentation_factors = {
            'semantic_drone': args.semantic_factor,
            'drone_deploy': args.drone_deploy_factor,
            'udd': args.udd_factor
        }
    
    # Expected patch counts
    expected_patches = {
        'semantic_drone': len(train_semantic) * args.semantic_factor,
        'drone_deploy': len(train_drone_deploy) * args.drone_deploy_factor,
        'udd': len(train_udd) * args.udd_factor
    }
    
    total_expected = sum(expected_patches.values())
    print(f"\n📈 Expected patch counts:")
    for name, count in expected_patches.items():
        print(f"   - {name}: ~{count:,} patches")
    print(f"   - Total: ~{total_expected:,} patches")
    
    # Create cached datasets
    try:
        cached_datasets = create_cached_datasets(
            drone_deploy_dataset=train_drone_deploy,
            udd_dataset=train_udd,
            semantic_drone_dataset=train_semantic,
            cache_root=args.cache_dir,
            augmentation_factors=augmentation_factors,
            force_rebuild=args.force_rebuild
        )
        
        # Final summary
        total_time = time.time() - start_time
        total_base = len(train_drone_deploy) + len(train_udd) + len(train_semantic)
        total_cached = sum(len(dataset) for dataset in cached_datasets.values())
        
        print(f"\n🎉 SUCCESS! Augmented datasets cached successfully!")
        print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"📊 Final statistics:")
        print(f"   - Base images: {total_base}")
        print(f"   - Cached patches: {total_cached:,}")
        print(f"   - Augmentation factor: {total_cached/total_base:.1f}x")
        
        # Individual dataset stats
        print(f"\n📈 Individual dataset statistics:")
        for name, dataset in cached_datasets.items():
            print(f"   - {name}: {len(dataset):,} patches")
        
        # Cache directory info
        cache_path = Path(args.cache_dir)
        if cache_path.exists():
            total_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
            print(f"💾 Cache size: {total_size / (1024**3):.2f} GB")
        
        print(f"\n✨ Ready for fast training! Use these cached datasets in your training script.")
        
    except Exception as e:
        print(f"❌ Error creating cached datasets: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 