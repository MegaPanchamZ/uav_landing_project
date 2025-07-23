#!/usr/bin/env python3
"""
Optimized Semantic Drone Dataset with Multi-Scale Support
=========================================================

This optimized dataset addresses loading bottlenecks and provides both:
- Detail preservation via high-res crops
- Holistic context via resized full images
- Pre-processed caching for faster loading
- GPU-optimized data pipeline
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

class OptimizedSemanticDroneDataset(Dataset):
    """
    High-performance semantic drone dataset with multi-scale support.
    
    Features:
    - Pre-processed crop caching for instant loading
    - Multi-scale: both crops (detail) and resized (context)
    - GPU-optimized pipeline
    - Background pre-processing
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        target_resolution: Tuple[int, int] = (512, 512),
        use_multi_scale: bool = True,
        crops_per_image: int = 4,
        cache_dir: Optional[str] = None,
        preprocess_on_init: bool = True
    ):
        """
        Initialize optimized dataset.
        
        Args:
            data_root: Path to dataset root
            split: Dataset split ('train', 'val', 'test')
            transform: Albumentations transform
            target_resolution: Target resolution
            use_multi_scale: Use both crops and resized images
            crops_per_image: Number of pre-computed crops per image
            cache_dir: Directory for preprocessing cache
            preprocess_on_init: Pre-process all data on initialization
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.target_resolution = target_resolution
        self.use_multi_scale = use_multi_scale
        self.crops_per_image = crops_per_image if split == "train" else 1
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = self.data_root.parent / f"cache_{target_resolution[0]}x{target_resolution[1]}"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load dataset files
        self.images_dir = self.data_root / "original_images"
        self.labels_dir = self.data_root / "label_images_semantic"
        
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        self.label_files = sorted(list(self.labels_dir.glob("*.png")))
        
        # Create splits
        self._create_splits()
        
        # Class mapping (same as before)
        self.class_mapping = {
            0: 0, 23: 0,  # background
            1: 1, 2: 1, 3: 1, 4: 1,  # safe_landing
            6: 2, 8: 2, 9: 2, 21: 2,  # caution
            5: 3, 7: 3, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3,
            15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3, 22: 3  # danger
        }
        
        # Pre-process data for fast loading
        if preprocess_on_init:
            self._preprocess_dataset()
        
        print(f"OptimizedSemanticDroneDataset initialized:")
        print(f"   Split: {split} ({len(self)} samples)")
        print(f"   Multi-scale: {use_multi_scale}")
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
    
    def _preprocess_dataset(self):
        """Pre-process and cache all crops and resized images."""
        cache_file = self.cache_dir / f"{self.split}_cache.pkl"
        
        if cache_file.exists():
            print(f"Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.cached_data = pickle.load(f)
            return
        
        print(f"Pre-processing {len(self.file_indices)} images for {self.split} split...")
        self.cached_data = {}
        
        def process_image(file_idx):
            """Process a single image and return cached data."""
            image_path = self.image_files[file_idx]
            label_path = self.label_files[file_idx]
            
            # Load full resolution
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            
            # Map classes
            mapped_label = self._map_classes(label)
            
            cached_item = {
                'crops': [],
                'resized_image': None,
                'resized_label': None,
                'original_shape': image.shape[:2]
            }
            
            # Generate crops (for detail)
            if self.split == "train":
                crop_h, crop_w = self.target_resolution
                img_h, img_w = image.shape[:2]
                
                for _ in range(self.crops_per_image):
                    # Random crop coordinates
                    if img_h > crop_h:
                        start_h = np.random.randint(0, img_h - crop_h + 1)
                    else:
                        start_h = 0
                    
                    if img_w > crop_w:
                        start_w = np.random.randint(0, img_w - crop_w + 1)
                    else:
                        start_w = 0
                    
                    # Extract crop
                    crop_img = image[start_h:start_h + crop_h, start_w:start_w + crop_w]
                    crop_label = mapped_label[start_h:start_h + crop_h, start_w:start_w + crop_w]
                    
                    cached_item['crops'].append({
                        'image': crop_img.astype(np.uint8),
                        'label': crop_label.astype(np.uint8)
                    })
            
            # Generate resized full image (for context)
            if self.use_multi_scale or self.split != "train":
                resized_img = cv2.resize(image, self.target_resolution[::-1], interpolation=cv2.INTER_LINEAR)
                resized_label = cv2.resize(mapped_label, self.target_resolution[::-1], interpolation=cv2.INTER_NEAREST)
                
                cached_item['resized_image'] = resized_img.astype(np.uint8)
                cached_item['resized_label'] = resized_label.astype(np.uint8)
            
            return file_idx, cached_item
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(tqdm(
                executor.map(process_image, self.file_indices),
                total=len(self.file_indices),
                desc="Pre-processing"
            ))
        
        # Store results
        for file_idx, cached_item in results:
            self.cached_data[file_idx] = cached_item
        
        # Save cache
        with open(cache_file, 'wb') as f:
            pickle.dump(self.cached_data, f)
        
        print(f"✅ Pre-processing complete! Cache saved to {cache_file}")
    
    def _map_classes(self, label: np.ndarray) -> np.ndarray:
        """Map original classes to landing classes."""
        mapped_label = np.zeros_like(label, dtype=np.uint8)
        
        for original_class, landing_class in self.class_mapping.items():
            mask = (label == original_class)
            mapped_label[mask] = landing_class
        
        return mapped_label
    
    def __len__(self):
        if self.split == "train":
            return len(self.file_indices) * self.crops_per_image
        else:
            return len(self.file_indices)
    
    def __getitem__(self, idx):
        """Get a pre-processed sample (very fast!)."""
        if self.split == "train":
            file_idx = self.file_indices[idx // self.crops_per_image]
            crop_idx = idx % self.crops_per_image
            
            # Get pre-processed crop
            cached_item = self.cached_data[file_idx]
            crop_data = cached_item['crops'][crop_idx]
            
            image = crop_data['image'].astype(np.float32)
            label = crop_data['label']
            
            # For multi-scale, also include resized context
            if self.use_multi_scale and cached_item['resized_image'] is not None:
                context_image = cached_item['resized_image'].astype(np.float32)
                context_label = cached_item['resized_label']
            else:
                context_image = None
                context_label = None
                
        else:
            # Validation: use resized full image
            file_idx = self.file_indices[idx]
            cached_item = self.cached_data[file_idx]
            
            image = cached_item['resized_image'].astype(np.float32)
            label = cached_item['resized_label']
            context_image = None
            context_label = None
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
            
            # Transform context if available
            if context_image is not None:
                context_transformed = self.transform(image=context_image, mask=context_label)
                context_image = context_transformed['image']
                context_label = context_transformed['mask']
        
        # Convert to tensors if needed
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label).long()
        
        sample = {
            'image': image,
            'mask': label,
            'idx': idx
        }
        
        # Add context for multi-scale
        if context_image is not None:
            if not isinstance(context_image, torch.Tensor):
                context_image = torch.from_numpy(context_image).permute(2, 0, 1).float() / 255.0
            if not isinstance(context_label, torch.Tensor):
                context_label = torch.from_numpy(context_label).long()
                
            sample['context_image'] = context_image
            sample['context_mask'] = context_label
        
        return sample


def create_optimized_transforms(
    input_size: Tuple[int, int] = (512, 512),
    is_training: bool = True,
    gpu_augmentation: bool = False
) -> A.Compose:
    """
    Create optimized transform pipeline.
    
    Args:
        input_size: Target size
        is_training: Whether training
        gpu_augmentation: Use GPU-based augmentation (if available)
    """
    transforms_list = []
    
    if is_training:
        # Light, fast augmentations
        transforms_list.extend([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        ])
    
    # Normalization and tensor conversion
    transforms_list.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return A.Compose(transforms_list)


if __name__ == "__main__":
    # Test the optimized dataset
    data_root = "../datasets/Aerial_Semantic_Segmentation_Drone_Dataset/dataset/semantic_drone_dataset"
    
    print("🚀 Testing Optimized Dataset...")
    
    start_time = time.time()
    
    transforms = create_optimized_transforms(is_training=True)
    
    dataset = OptimizedSemanticDroneDataset(
        data_root=data_root,
        split="train",
        transform=transforms,
        use_multi_scale=True,
        crops_per_image=4,
        preprocess_on_init=True
    )
    
    init_time = time.time() - start_time
    print(f"✅ Dataset initialization: {init_time:.2f}s")
    
    # Test loading speed
    start_time = time.time()
    for i in range(10):
        sample = dataset[i]
        if i == 0:
            print(f"Sample keys: {sample.keys()}")
            print(f"Image shape: {sample['image'].shape}")
            print(f"Mask shape: {sample['mask'].shape}")
            if 'context_image' in sample:
                print(f"Context image shape: {sample['context_image'].shape}")
    
    loading_time = time.time() - start_time
    print(f"✅ Loading 10 samples: {loading_time:.3f}s ({loading_time/10*1000:.1f}ms per sample)") 