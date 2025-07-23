#!/usr/bin/env python3
"""
Test Optimized Dataset Speed vs Original
========================================

Compare loading speeds between:
1. Original dataset (load 24MP → crop/resize)
2. Optimized dataset (pre-processed cache)
"""

import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets.semantic_drone_dataset import SemanticDroneDataset, create_semantic_drone_transforms
from datasets.optimized_semantic_dataset import OptimizedSemanticDroneDataset, create_optimized_transforms

def benchmark_dataset(dataset, name, num_batches=5):
    """Benchmark dataset loading speed."""
    
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\n🔥 Benchmarking {name}")
    print(f"   Dataset size: {len(dataset)} samples")
    
    # Warmup
    warmup_batch = next(iter(loader))
    
    # Actual timing
    start_time = time.time()
    total_samples = 0
    
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        total_samples += batch['image'].shape[0]
        
        if i == 0:
            print(f"   Sample shape: {batch['image'].shape}")
            if 'context_image' in batch:
                print(f"   Context shape: {batch['context_image'].shape}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"   ⏱️  {num_batches} batches ({total_samples} samples): {total_time:.3f}s")
    print(f"   📊 {total_time/total_samples*1000:.1f}ms per sample")
    print(f"   🚀 {total_samples/total_time:.1f} samples/sec")
    
    return total_time, total_samples

def main():
    """Compare dataset loading speeds."""
    
    dataset_path = Path(__file__).parent.parent.parent / 'datasets' / 'Aerial_Semantic_Segmentation_Drone_Dataset' / 'dataset' / 'semantic_drone_dataset'
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        return
    
    print("🏁 DATASET LOADING SPEED COMPARISON")
    print("=" * 60)
    
    # 1. Test Original Dataset (Random Crops)
    print("\n1️⃣ Original Dataset with Random Crops")
    original_transforms = create_semantic_drone_transforms(
        input_size=(512, 512),
        is_training=True,
        advanced_augmentation=False,
        use_resize=False
    )
    
    original_dataset = SemanticDroneDataset(
        data_root=str(dataset_path),
        split="train",
        transform=original_transforms,
        use_random_crops=True,
        crops_per_image=4,
        cache_images=True
    )
    
    orig_time, orig_samples = benchmark_dataset(original_dataset, "Original (Random Crops)", 5)
    
    # 2. Test Original Dataset (Full Resize)
    print("\n2️⃣ Original Dataset with Full Resize")
    resize_transforms = create_semantic_drone_transforms(
        input_size=(512, 512),
        is_training=True,
        advanced_augmentation=False,
        use_resize=True
    )
    
    resize_dataset = SemanticDroneDataset(
        data_root=str(dataset_path),
        split="train",
        transform=resize_transforms,
        use_random_crops=False,
        crops_per_image=1,
        cache_images=True
    )
    
    resize_time, resize_samples = benchmark_dataset(resize_dataset, "Original (Full Resize)", 5)
    
    # 3. Test Optimized Dataset
    print("\n3️⃣ Optimized Dataset (Pre-processed Cache)")
    optimized_transforms = create_optimized_transforms(
        input_size=(512, 512),
        is_training=True
    )
    
    optimized_dataset = OptimizedSemanticDroneDataset(
        data_root=str(dataset_path),
        split="train",
        transform=optimized_transforms,
        use_multi_scale=True,
        crops_per_image=4,
        preprocess_on_init=True
    )
    
    opt_time, opt_samples = benchmark_dataset(optimized_dataset, "Optimized (Pre-processed)", 5)
    
    # 4. Summary Comparison
    print("\n📊 SPEED COMPARISON SUMMARY")
    print("=" * 60)
    
    orig_speed = orig_samples / orig_time
    resize_speed = resize_samples / resize_time  
    opt_speed = opt_samples / opt_time
    
    print(f"Original (Crops):    {orig_speed:.1f} samples/sec")
    print(f"Original (Resize):   {resize_speed:.1f} samples/sec") 
    print(f"Optimized (Cache):   {opt_speed:.1f} samples/sec")
    
    print(f"\n🚀 Speedup vs Crops:  {opt_speed/orig_speed:.1f}x faster")
    print(f"🚀 Speedup vs Resize: {opt_speed/resize_speed:.1f}x faster")
    
    print(f"\n💡 Key Benefits:")
    print(f"   • Pre-processed caching eliminates CPU bottleneck")
    print(f"   • Multi-scale support (detail + context)")
    print(f"   • GPU-optimized pipeline with prefetching")
    print(f"   • First epoch: one-time preprocessing cost")
    print(f"   • Subsequent epochs: instant loading from cache")

if __name__ == "__main__":
    main() 