#!/usr/bin/env python3
"""
Memory-Efficient Semantic Drone Dataset
=======================================

Optimized for systems with limited GPU memory (8GB).
Uses smart caching and on-demand processing to balance speed vs memory.

Key optimizations:
- Smaller cache with strategic pre-processing
- On-demand transforms for memory efficiency
- Progressive loading to avoid memory spikes
- Memory monitoring and cleanup
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import pickle
from typing import Tuple, Dict, Optional, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm
import psutil
import gc

class MemoryEfficientSemanticDataset(Dataset):
    """
    Memory-efficient dataset optimized for 8GB GPU systems.
    
    Features:
    - Pre-computed crops (no full image loading)
    - Lightweight cache (numpy arrays, not tensors)
    - On-demand transforms (to save memory)
    - Progressive loading
    - Memory monitoring
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        target_resolution: Tuple[int, int] = (512, 512),
        crops_per_image: int = 4,
        cache_dir: Optional[str] = None,
        preprocess_on_init: bool = True,
        max_memory_gb: float = 6.0  # Conservative limit
    ):
        """
        Initialize memory-efficient dataset.
        
        Args:
            data_root: Path to dataset root
            split: Dataset split ('train', 'val', 'test')
            target_resolution: Target resolution
            crops_per_image: Number of crops per image
            cache_dir: Directory for preprocessing cache
            preprocess_on_init: Pre-process data on initialization
            max_memory_gb: Maximum memory to use for caching
        """
        self.data_root = Path(data_root)
        self.split = split
        self.target_resolution = target_resolution
        self.crops_per_image = crops_per_image if split == "train" else 1
        self.max_memory_gb = max_memory_gb
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = self.data_root.parent / f"efficient_cache_{target_resolution[0]}x{target_resolution[1]}"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load dataset files
        self.images_dir = self.data_root / "original_images"
        self.labels_dir = self.data_root / "label_images_semantic"
        
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        self.label_files = sorted(list(self.labels_dir.glob("*.png")))
        
        # Create splits
        self._create_splits()
        
        # Class mapping
        self.class_mapping = {
            0: 0, 23: 0,  # background
            1: 1, 2: 1, 3: 1, 4: 1,  # safe_landing
            6: 2, 8: 2, 9: 2, 21: 2,  # caution
            5: 3, 7: 3, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3,
            15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3, 22: 3  # danger
        }
        
        # Create transforms
        self._create_transforms()
        
        # Pre-process data efficiently
        if preprocess_on_init:
            self._preprocess_dataset()
        
        print(f"MemoryEfficientSemanticDataset initialized:")
        print(f"   Split: {split} ({len(self)} samples)")
        print(f"   Memory limit: {max_memory_gb:.1f}GB")
        print(f"   Cache dir: {self.cache_dir}")
        
    def _create_splits(self):
        """Create train/val/test splits."""
        total_files = len(self.image_files)
        indices = np.arange(total_files)
        
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_end = int(0.7 * total_files)
        val_end = int(0.85 * total_files)
        
        if self.split == "train":
            self.file_indices = indices[:train_end]
        elif self.split == "val":
            self.file_indices = indices[train_end:val_end]
        elif self.split == "test":
            self.file_indices = indices[val_end:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def _create_transforms(self):
        """Create efficient transform pipeline."""
        if self.split == "train":
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def _get_memory_usage(self):
        """Get current memory usage in GB."""
        process = psutil.Process()
        return process.memory_info().rss / 1e9
    
    def _preprocess_dataset(self):
        """Pre-process and cache crops efficiently."""
        cache_file = self.cache_dir / f"{self.split}_efficient_cache.pkl"
        
        if cache_file.exists():
            print(f"Loading efficient cache from {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.cached_crops = cache_data['crops']
            self.crop_indices = cache_data['indices']
            
            memory_usage = self._get_memory_usage()
            print(f"   Cache loaded, memory usage: {memory_usage:.1f}GB")
            return
        
        print(f"Pre-processing {len(self.file_indices)} images (memory-efficient)...")
        
        self.cached_crops = []
        self.crop_indices = []
        
        initial_memory = self._get_memory_usage()
        print(f"   Initial memory: {initial_memory:.1f}GB")
        
        def process_image_batch(file_indices_batch):
            """Process a batch of images to control memory usage."""
            batch_crops = []
            batch_indices = []
            
            for file_idx in file_indices_batch:
                image_path = self.image_files[file_idx]
                label_path = self.label_files[file_idx]
                
                # Load full resolution
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
                
                # Map classes
                mapped_label = self._map_classes(label)
                
                # Generate crops
                for crop_idx in range(self.crops_per_image):
                    if self.split == "train":
                        # Random crop
                        crop_h, crop_w = self.target_resolution
                        img_h, img_w = image.shape[:2]
                        
                        if img_h > crop_h:
                            start_h = np.random.randint(0, img_h - crop_h + 1)
                        else:
                            start_h = 0
                        
                        if img_w > crop_w:
                            start_w = np.random.randint(0, img_w - crop_w + 1)
                        else:
                            start_w = 0
                        
                        crop_img = image[start_h:start_h + crop_h, start_w:start_w + crop_w]
                        crop_label = mapped_label[start_h:start_h + crop_h, start_w:start_w + crop_w]
                    else:
                        # Resize for validation
                        crop_img = cv2.resize(image, self.target_resolution[::-1], interpolation=cv2.INTER_LINEAR)
                        crop_label = cv2.resize(mapped_label, self.target_resolution[::-1], interpolation=cv2.INTER_NEAREST)
                    
                    # Store as compact numpy arrays (uint8)
                    crop_data = {
                        'image': crop_img.astype(np.uint8),
                        'label': crop_label.astype(np.uint8)
                    }
                    
                    batch_crops.append(crop_data)
                    batch_indices.append(len(batch_crops) - 1)
            
            return batch_crops, batch_indices
        
        # Process in small batches to control memory
        batch_size = 20  # Process 20 images at a time
        
        for i in tqdm(range(0, len(self.file_indices), batch_size), desc="Processing batches"):
            batch_files = self.file_indices[i:i + batch_size]
            
            # Process batch
            batch_crops, batch_indices = process_image_batch(batch_files)
            
            # Add to main cache
            start_idx = len(self.cached_crops)
            self.cached_crops.extend(batch_crops)
            self.crop_indices.extend([start_idx + idx for idx in batch_indices])
            
            # Memory monitoring
            current_memory = self._get_memory_usage()
            if current_memory > self.max_memory_gb:
                print(f"   Warning: Memory usage {current_memory:.1f}GB > {self.max_memory_gb:.1f}GB limit")
                gc.collect()  # Force garbage collection
            
            # Periodic cleanup
            if i % (batch_size * 5) == 0:
                gc.collect()
        
        # Save cache
        cache_data = {
            'crops': self.cached_crops,
            'indices': self.crop_indices
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        final_memory = self._get_memory_usage()
        print(f"✅ Efficient preprocessing complete!")
        print(f"   Samples cached: {len(self.cached_crops)}")
        print(f"   Final memory: {final_memory:.1f}GB")
        print(f"   Memory increase: {final_memory - initial_memory:.1f}GB")
    
    def _map_classes(self, label: np.ndarray) -> np.ndarray:
        """Map original classes to landing classes."""
        mapped_label = np.zeros_like(label, dtype=np.uint8)
        
        for original_class, landing_class in self.class_mapping.items():
            mask = (label == original_class)
            mapped_label[mask] = landing_class
        
        return mapped_label
    
    def __len__(self):
        return len(self.cached_crops)
    
    def __getitem__(self, idx):
        """Get a sample with on-demand transforms."""
        # Get pre-computed crop
        crop_data = self.cached_crops[idx]
        
        # Convert to float32 for transforms
        image = crop_data['image'].astype(np.float32)
        label = crop_data['label']
        
        # Apply transforms on-demand (memory efficient)
        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
        
        # Convert to tensors if not already done
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label).long()
        
        return {
            'image': image,
            'mask': label,
            'idx': idx
        }


def test_memory_efficient_dataset():
    """Test the memory-efficient dataset."""
    data_root = "../datasets/Aerial_Semantic_Segmentation_Drone_Dataset/dataset/semantic_drone_dataset"
    
    print("🧠 Testing Memory-Efficient Dataset...")
    
    initial_memory = psutil.Process().memory_info().rss / 1e9
    print(f"Initial memory: {initial_memory:.1f}GB")
    
    start_time = time.time()
    
    dataset = MemoryEfficientSemanticDataset(
        data_root=data_root,
        split="train",
        crops_per_image=4,
        max_memory_gb=6.0,  # Conservative for 8GB GPU
        preprocess_on_init=True
    )
    
    init_time = time.time() - start_time
    final_memory = psutil.Process().memory_info().rss / 1e9
    
    print(f"✅ Dataset initialization: {init_time:.2f}s")
    print(f"✅ Memory usage: {final_memory:.1f}GB (+{final_memory-initial_memory:.1f}GB)")
    
    # Test loading speed
    print(f"\n⚡ Testing loading speed...")
    start_time = time.time()
    for i in range(50):
        sample = dataset[i]
        if i == 0:
            print(f"Sample keys: {sample.keys()}")
            print(f"Image shape: {sample['image'].shape}")
            print(f"Image dtype: {sample['image'].dtype}")
    
    loading_time = time.time() - start_time
    print(f"✅ Loading 50 samples: {loading_time:.3f}s ({loading_time/50*1000:.1f}ms per sample)")
    print(f"🚀 Speed: {50/loading_time:.1f} samples/sec")


if __name__ == "__main__":
    test_memory_efficient_dataset() 