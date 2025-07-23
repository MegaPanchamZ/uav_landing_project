#!/usr/bin/env python3
"""
Profile Data Loading Bottlenecks
================================

Identify exactly where time is being spent in data loading.
"""

import sys
import time
import cProfile
import pstats
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))

from datasets.optimized_semantic_dataset import OptimizedSemanticDroneDataset, create_optimized_transforms

def profile_single_sample():
    """Profile a single sample loading to identify bottlenecks."""
    
    dataset_path = '../datasets/Aerial_Semantic_Segmentation_Drone_Dataset/dataset/semantic_drone_dataset'
    transforms = create_optimized_transforms(input_size=(512, 512), is_training=True)
    
    print("🔍 Profiling single sample loading...")
    
    dataset = OptimizedSemanticDroneDataset(
        data_root=dataset_path,
        split='train',
        transform=transforms,
        use_multi_scale=True,
        crops_per_image=4,
        preprocess_on_init=True
    )
    
    print(f"Dataset loaded, testing sample access...")
    
    # Profile single sample access
    start_time = time.time()
    sample = dataset[0]
    single_time = time.time() - start_time
    
    print(f"⏱️ Single sample: {single_time*1000:.1f}ms")
    print(f"   Image shape: {sample['image'].shape}")
    print(f"   Image dtype: {sample['image'].dtype}")
    if 'context_image' in sample:
        print(f"   Context shape: {sample['context_image'].shape}")
    
    # Time components separately
    print("\n🔬 Component breakdown:")
    
    # Test cache access
    start_time = time.time()
    cached_item = dataset.cached_data[dataset.file_indices[0]]
    cache_time = time.time() - start_time
    print(f"   Cache lookup: {cache_time*1000:.1f}ms")
    
    # Test transform application
    crop_data = cached_item['crops'][0]
    image = crop_data['image'].astype(np.float32)
    label = crop_data['label']
    
    start_time = time.time()
    if dataset.transform:
        transformed = dataset.transform(image=image, mask=label)
        image_t = transformed['image']
        label_t = transformed['mask']
    transform_time = time.time() - start_time
    print(f"   Transform: {transform_time*1000:.1f}ms")
    
    # Test tensor conversion
    start_time = time.time()
    if not isinstance(image_t, torch.Tensor):
        image_tensor = torch.from_numpy(image_t).permute(2, 0, 1).float() / 255.0
    if not isinstance(label_t, torch.Tensor):
        label_tensor = torch.from_numpy(label_t).long()
    tensor_time = time.time() - start_time
    print(f"   Tensor conversion: {tensor_time*1000:.1f}ms")
    
    return single_time, cache_time, transform_time, tensor_time

def profile_batch_loading():
    """Profile batch loading."""
    
    dataset_path = '../datasets/Aerial_Semantic_Segmentation_Drone_Dataset/dataset/semantic_drone_dataset'
    transforms = create_optimized_transforms(input_size=(512, 512), is_training=True)
    
    dataset = OptimizedSemanticDroneDataset(
        data_root=dataset_path,
        split='train',
        transform=transforms,
        use_multi_scale=True,
        crops_per_image=4,
        preprocess_on_init=True
    )
    
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,  # Don't shuffle for consistent timing
        num_workers=0,  # Single threaded for easier profiling
        pin_memory=False
    )
    
    print(f"\n🔍 Profiling batch loading (single worker)...")
    
    start_time = time.time()
    batch = next(iter(loader))
    batch_time = time.time() - start_time
    
    print(f"⏱️ Batch (16 samples): {batch_time*1000:.1f}ms")
    print(f"   Per sample: {batch_time/16*1000:.1f}ms")
    
    return batch_time

def main():
    """Profile data loading and identify bottlenecks."""
    
    print("🚀 DATA LOADING PROFILER")
    print("=" * 50)
    
    import numpy as np
    
    # Profile components
    single_time, cache_time, transform_time, tensor_time = profile_single_sample()
    
    batch_time = profile_batch_loading()
    
    print(f"\n📊 BOTTLENECK ANALYSIS")
    print("=" * 50)
    print(f"Single sample total: {single_time*1000:.1f}ms")
    print(f"  - Cache lookup:    {cache_time*1000:.1f}ms ({cache_time/single_time*100:.1f}%)")
    print(f"  - Transforms:      {transform_time*1000:.1f}ms ({transform_time/single_time*100:.1f}%)")
    print(f"  - Tensor convert:  {tensor_time*1000:.1f}ms ({tensor_time/single_time*100:.1f}%)")
    print(f"  - Other:           {(single_time-cache_time-transform_time-tensor_time)*1000:.1f}ms")
    
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
    if transform_time > 0.01:  # > 10ms
        print(f"    MAJOR: Pre-compute transforms (saves {transform_time*1000:.1f}ms)")
    if tensor_time > 0.005:  # > 5ms
        print(f"    Store as tensors directly (saves {tensor_time*1000:.1f}ms)")
    if cache_time > 0.01:  # > 10ms
        print(f"    Optimize cache format (saves {cache_time*1000:.1f}ms)")
    
    target_speed = 1000 / single_time  # samples per second
    print(f"\n🚀 Current speed: {target_speed:.1f} samples/sec")
    print(f"   Target for training: >50 samples/sec")

if __name__ == "__main__":
    main() 