#!/usr/bin/env python3
"""
Cached Augmentation System for UAV Landing Detection
==================================================

Generates and caches augmented datasets to disk for fast reuse.
Creates a one-time comprehensive augmentation with optimal factors.

Features:
- Disk-based caching for instant loading
- Comprehensive augmentation factors
- Progress tracking and resume capability
- Memory-efficient processing
- Metadata storage for reproducibility
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import albumentations as A
import random
import pickle
import json
import hashlib
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class CachedAugmentedDataset(Dataset):
    """Cached augmented dataset with disk storage for fast reuse."""
    
    def __init__(
        self,
        base_dataset: Dataset,
        cache_dir: str,
        dataset_name: str,
        patch_scales: List[Tuple[int, int]] = [(512, 512), (768, 768)],
        augmentation_factor: int = 20,
        min_object_ratio: float = 0.05,
        use_overlapping: bool = True,
        overlap_ratio: float = 0.25,
        uav_augmentations: bool = True,
        force_rebuild: bool = False,
        num_workers: int = 4,
        fast_mode: bool = False
    ):
        """
        Args:
            base_dataset: Original dataset to augment
            cache_dir: Directory to store cached patches
            dataset_name: Unique name for this dataset cache
            patch_scales: List of (width, height) scales to extract
            augmentation_factor: Number of patches per base image
            min_object_ratio: Minimum ratio of non-background pixels
            use_overlapping: Whether to use overlapping patches
            overlap_ratio: Overlap ratio for patches (0.0-0.5)
            uav_augmentations: Whether to apply UAV-specific augmentations
            force_rebuild: Whether to force rebuilding the cache
            num_workers: Number of worker threads for parallel processing
        """
        self.base_dataset = base_dataset
        self.cache_dir = Path(cache_dir)
        self.dataset_name = dataset_name
        self.patch_scales = patch_scales
        self.augmentation_factor = augmentation_factor
        self.min_object_ratio = min_object_ratio
        self.use_overlapping = use_overlapping
        self.overlap_ratio = overlap_ratio
        self.uav_augmentations = uav_augmentations
        self.num_workers = num_workers
        self.fast_mode = fast_mode
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.patches_dir = self.cache_dir / f"{dataset_name}_patches"
        self.patches_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file paths
        self.metadata_file = self.cache_dir / f"{dataset_name}_metadata.json"
        self.index_file = self.cache_dir / f"{dataset_name}_index.pkl"
        
        # Generate unique cache hash based on parameters
        self.cache_hash = self._generate_cache_hash()
        
        # Check if cache exists and is valid
        if self._is_cache_valid() and not force_rebuild:
            print(f"📂 Loading cached dataset: {dataset_name}")
            self._load_cache()
        else:
            print(f"🔧 Building augmented dataset: {dataset_name}")
            print(f"   Expected patches: ~{len(base_dataset) * augmentation_factor}")
            self._build_cache()
        
        print(f"📊 CachedAugmentedDataset loaded:")
        print(f"   Dataset: {dataset_name}")
        print(f"   Base size: {len(base_dataset)}")
        print(f"   Cached patches: {len(self.patch_index)}")
        print(f"   Augmentation factor: {len(self.patch_index)/len(base_dataset):.1f}x")
        print(f"   Cache size: {self._get_cache_size():.1f} MB")
    
    def _generate_cache_hash(self) -> str:
        """Generate a unique hash for the current configuration."""
        config_str = f"{len(self.base_dataset)}_{self.patch_scales}_{self.augmentation_factor}_{self.min_object_ratio}_{self.use_overlapping}_{self.overlap_ratio}_{self.uav_augmentations}"
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def _is_cache_valid(self) -> bool:
        """Check if the existing cache is valid."""
        if not self.metadata_file.exists() or not self.index_file.exists():
            return False
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            return (
                metadata.get('cache_hash') == self.cache_hash and
                metadata.get('base_dataset_size') == len(self.base_dataset) and
                Path(metadata.get('patches_dir', '')).exists()
            )
        except:
            return False
    
    def _load_cache(self):
        """Load existing cache."""
        with open(self.index_file, 'rb') as f:
            self.patch_index = pickle.load(f)
    
    def _build_cache(self):
        """Build and cache the augmented dataset."""
        start_time = time.time()
        
        # Create UAV augmentation pipeline
        if self.uav_augmentations:
            self.uav_transforms = self._create_uav_augmentations()
        
        # Generate patches in parallel
        self.patch_index = []
        
        print("🚀 Extracting patches from base images...")
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all base images for processing
            future_to_idx = {
                executor.submit(self._process_base_image, base_idx): base_idx
                for base_idx in range(len(self.base_dataset))
            }
            
            # Process completed futures with progress bar
            with tqdm(total=len(self.base_dataset), desc="Processing base images") as pbar:
                for future in as_completed(future_to_idx):
                    base_idx = future_to_idx[future]
                    try:
                        patches = future.result()
                        self.patch_index.extend(patches)
                        pbar.set_postfix({
                            'patches': len(self.patch_index),
                            'avg_per_image': f"{len(self.patch_index)/(pbar.n+1):.1f}"
                        })
                    except Exception as e:
                        print(f"❌ Error processing image {base_idx}: {e}")
                    finally:
                        pbar.update(1)
        
        # Save augmented patches to disk with progress bar (batch processing for speed)
        print("💾 Saving augmented patches to disk...")
        
        # Process patches in larger batches for better efficiency
        batch_size = min(200, max(50, len(self.patch_index) // (self.num_workers * 2)))
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            # Submit batch saving tasks
            for i in range(0, len(self.patch_index), batch_size):
                batch = self.patch_index[i:i + batch_size]
                future = executor.submit(self._save_patch_batch, i, batch)
                futures.append(future)
            
            # Wait for completion with progress bar
            with tqdm(total=len(futures), desc="Saving patch batches") as pbar:
                for future in as_completed(futures):
                    try:
                        batch_size_actual = future.result()
                        pbar.set_postfix({'patches_saved': f'{(pbar.n + 1) * batch_size}'})
                    except Exception as e:
                        print(f"❌ Error saving batch: {e}")
                    finally:
                        pbar.update(1)
        
        # Save metadata and index
        self._save_metadata()
        
        build_time = time.time() - start_time
        print(f" Dataset cached successfully!")
        print(f"   Time taken: {build_time:.1f} seconds")
        print(f"   Patches created: {len(self.patch_index)}")
        print(f"   Cache size: {self._get_cache_size():.1f} MB")
    
    def _process_base_image(self, base_idx: int) -> List[Dict]:
        """Process a single base image to extract patches."""
        patches = []
        
        try:
            # Get base sample
            base_sample = self.base_dataset[base_idx]
            
            # Convert to numpy if needed
            if isinstance(base_sample['image'], torch.Tensor):
                if base_sample['image'].dim() == 3:  # CHW format
                    image = base_sample['image'].permute(1, 2, 0).numpy()
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                else:
                    image = base_sample['image'].numpy()
            else:
                image = base_sample['image']
            
            if isinstance(base_sample['mask'], torch.Tensor):
                mask = base_sample['mask'].numpy()
            else:
                mask = base_sample['mask']
            
            # Generate patches for each scale
            for scale_w, scale_h in self.patch_scales:
                scale_patches = self._extract_patches_from_image(
                    image, mask, base_idx, scale_w, scale_h
                )
                patches.extend(scale_patches)
        
        except Exception as e:
            print(f"❌ Error processing base image {base_idx}: {e}")
        
        return patches
    
    def _extract_patches_from_image(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        base_idx: int,
        patch_w: int, 
        patch_h: int
    ) -> List[Dict]:
        """Extract patches from a single high-resolution image."""
        patches = []
        h, w = image.shape[:2]
        
        if h < patch_h or w < patch_w:
            # Image too small, resize and use as single patch
            patches.append({
                'base_idx': base_idx,
                'patch_coords': (0, 0, w, h),
                'scale': (patch_w, patch_h),
                'needs_resize': True,
                'altitude_sim': 1.0,
                'quality_score': 1.0,
                'patch_type': 'resized'
            })
            return patches
        
        # Calculate step size for overlapping patches
        if self.use_overlapping:
            step_w = int(patch_w * (1 - self.overlap_ratio))
            step_h = int(patch_h * (1 - self.overlap_ratio))
        else:
            step_w, step_h = patch_w, patch_h
        
        # Generate systematic patches
        patch_positions = []
        for y in range(0, h - patch_h + 1, step_h):
            for x in range(0, w - patch_w + 1, step_w):
                patch_positions.append((x, y))
        
        # Add random patches for diversity
        num_random = max(1, self.augmentation_factor // 3)
        for _ in range(num_random):
            x = random.randint(0, max(0, w - patch_w))
            y = random.randint(0, max(0, h - patch_h))
            patch_positions.append((x, y))
        
        # Filter patches by quality and select best ones
        valid_patches = []
        for x, y in patch_positions:
            patch_mask = mask[y:y+patch_h, x:x+patch_w]
            
            # Quality metrics
            non_bg_ratio = (patch_mask > 0).sum() / patch_mask.size
            class_diversity = len(np.unique(patch_mask))
            
            if non_bg_ratio >= self.min_object_ratio:
                quality_score = non_bg_ratio + (class_diversity - 1) * 0.1
                valid_patches.append({
                    'coords': (x, y),
                    'quality': quality_score,
                    'non_bg_ratio': non_bg_ratio,
                    'class_diversity': class_diversity
                })
        
        # Sort by quality and select top patches
        valid_patches.sort(key=lambda p: p['quality'], reverse=True)
        selected_patches = valid_patches[:self.augmentation_factor]
        
        # Create patch metadata
        for i, patch_info in enumerate(selected_patches):
            x, y = patch_info['coords']
            
            # Simulate different altitudes
            altitude_factor = random.uniform(0.7, 1.3)
            
            patches.append({
                'base_idx': base_idx,
                'patch_coords': (x, y, x + patch_w, y + patch_h),
                'scale': (patch_w, patch_h),
                'needs_resize': False,
                'altitude_sim': altitude_factor,
                'quality_score': patch_info['quality'],
                'non_bg_ratio': patch_info['non_bg_ratio'],
                'patch_type': 'extracted'
            })
        
        return patches
    
    def _save_patch_batch(self, start_idx: int, patch_batch: List[Dict]) -> int:
        """Save a batch of patches to disk for better efficiency."""
        saved_count = 0
        
        for i, patch_info in enumerate(patch_batch):
            patch_idx = start_idx + i
            try:
                # Generate the patch
                patch_image, patch_mask = self._generate_patch(patch_info)
                
                # Save image and mask
                image_path = self.patches_dir / f"patch_{patch_idx:06d}_image.npz"
                mask_path = self.patches_dir / f"patch_{patch_idx:06d}_mask.npy"
                
                # Save as uncompressed numpy arrays for speed (compression is slow)
                np.save(image_path.with_suffix('.npy'), patch_image)
                np.save(mask_path, patch_mask)
                
                # Update patch info with file paths
                patch_info['image_path'] = str(image_path.with_suffix('.npy'))
                patch_info['mask_path'] = str(mask_path)
                patch_info['patch_idx'] = patch_idx
                
                saved_count += 1
                
            except Exception as e:
                print(f"❌ Error saving patch {patch_idx}: {e}")
        
        return saved_count

    def _save_patch(self, patch_idx: int, patch_info: Dict):
        """Save a single patch to disk (legacy method)."""
        try:
            # Generate the patch
            patch_image, patch_mask = self._generate_patch(patch_info)
            
            # Save image and mask
            image_path = self.patches_dir / f"patch_{patch_idx:06d}_image.npz"
            mask_path = self.patches_dir / f"patch_{patch_idx:06d}_mask.npy"
            
            # Save as uncompressed numpy arrays for speed
            np.save(image_path.with_suffix('.npy'), patch_image)
            np.save(mask_path, patch_mask)
            
            # Update patch info with file paths
            patch_info['image_path'] = str(image_path.with_suffix('.npy'))
            patch_info['mask_path'] = str(mask_path)
            patch_info['patch_idx'] = patch_idx
            
        except Exception as e:
            print(f"❌ Error saving patch {patch_idx}: {e}")
    
    def _generate_patch(self, patch_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a patch from the base image."""
        base_idx = patch_info['base_idx']
        base_sample = self.base_dataset[base_idx]
        
        # Convert to numpy if needed
        if isinstance(base_sample['image'], torch.Tensor):
            if base_sample['image'].dim() == 3:  # CHW format
                image = base_sample['image'].permute(1, 2, 0).numpy()
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
            else:
                image = base_sample['image'].numpy()
        else:
            image = base_sample['image']
        
        if isinstance(base_sample['mask'], torch.Tensor):
            mask = base_sample['mask'].numpy()
        else:
            mask = base_sample['mask']
        
        # Extract patch
        if patch_info['needs_resize']:
            # Small image, resize to target scale
            patch_image = cv2.resize(image, patch_info['scale'])
            patch_mask = cv2.resize(mask, patch_info['scale'], interpolation=cv2.INTER_NEAREST)
        else:
            # Extract patch from high-resolution image
            x1, y1, x2, y2 = patch_info['patch_coords']
            patch_image = image[y1:y2, x1:x2].copy()
            patch_mask = mask[y1:y2, x1:x2].copy()
            
            # Resize to target scale if needed
            if patch_image.shape[:2] != patch_info['scale'][::-1]:
                patch_image = cv2.resize(patch_image, patch_info['scale'])
                patch_mask = cv2.resize(patch_mask, patch_info['scale'], interpolation=cv2.INTER_NEAREST)
        
        # Apply altitude simulation
        altitude_factor = patch_info['altitude_sim']
        if altitude_factor != 1.0:
            h, w = patch_image.shape[:2]
            new_size = (int(w * altitude_factor), int(h * altitude_factor))
            patch_image = cv2.resize(patch_image, new_size)
            patch_mask = cv2.resize(patch_mask, new_size, interpolation=cv2.INTER_NEAREST)
            
            # Crop or pad to target size
            target_h, target_w = patch_info['scale'][::-1]
            patch_image = self._crop_or_pad(patch_image, (target_h, target_w))
            patch_mask = self._crop_or_pad(patch_mask, (target_h, target_w))
        
        return patch_image, patch_mask
    
    def _crop_or_pad(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Crop or pad image to target size."""
        target_h, target_w = target_size
        h, w = image.shape[:2]
        
        if h == target_h and w == target_w:
            return image
        
        # Calculate padding/cropping
        if h < target_h or w < target_w:
            # Pad
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            
            if len(image.shape) == 3:
                padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
            else:
                padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='edge')
            
            return padded[:target_h, :target_w]
        else:
            # Crop from center
            start_h = (h - target_h) // 2
            start_w = (w - target_w) // 2
            return image[start_h:start_h + target_h, start_w:start_w + target_w]
    
    def _create_uav_augmentations(self) -> A.Compose:
        """Create UAV-specific augmentations."""
        return A.Compose([
            # Motion blur (UAV movement, wind)
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            
            # Lighting conditions
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.3, 0.3), 
                    contrast_limit=(-0.2, 0.2), 
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            ], p=0.6),
            
            # Weather effects (simplified for compatibility)
            A.OneOf([
                A.RandomFog(p=1.0),
                A.RandomRain(p=1.0),
                A.RandomSunFlare(p=1.0),
            ], p=0.2),
            
            # Shadows
            A.RandomShadow(p=0.15),
            
            # Color variations
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=10, 
                    sat_shift_limit=20, 
                    val_shift_limit=15, 
                    p=1.0
                ),
                A.ColorJitter(
                    brightness=0.2, 
                    contrast=0.2, 
                    saturation=0.2, 
                    hue=0.1, 
                    p=1.0
                ),
            ], p=0.4),
            
            # Noise
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.ISONoise(p=1.0),
            ], p=0.2),
            
            # Perspective variations
            A.OneOf([
                A.Perspective(scale=(0.02, 0.05), p=1.0),
                A.ShiftScaleRotate(
                    shift_limit=0.05, 
                    scale_limit=0.1, 
                    rotate_limit=5, 
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1.0
                ),
            ], p=0.3),
            
            # Final normalization and tensor conversion
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.pytorch.ToTensorV2()
        ])
    
    def _save_metadata(self):
        """Save metadata and index to disk."""
        # Metadata
        metadata = {
            'dataset_name': self.dataset_name,
            'cache_hash': self.cache_hash,
            'base_dataset_size': len(self.base_dataset),
            'patch_count': len(self.patch_index),
            'patch_scales': self.patch_scales,
            'augmentation_factor': self.augmentation_factor,
            'min_object_ratio': self.min_object_ratio,
            'use_overlapping': self.use_overlapping,
            'overlap_ratio': self.overlap_ratio,
            'uav_augmentations': self.uav_augmentations,
            'patches_dir': str(self.patches_dir),
            'created_timestamp': time.time()
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Index
        with open(self.index_file, 'wb') as f:
            pickle.dump(self.patch_index, f)
    
    def _get_cache_size(self) -> float:
        """Get cache size in MB."""
        total_size = 0
        for file_path in self.patches_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)  # Convert to MB
    
    def __len__(self) -> int:
        return len(self.patch_index)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a cached patch."""
        patch_info = self.patch_index[idx]
        
        # Load image and mask from disk (uncompressed for speed)
        patch_image = np.load(patch_info['image_path'])
        patch_mask = np.load(patch_info['mask_path'])
        
        # Apply UAV augmentations to loaded patch
        if self.uav_augmentations:
            try:
                augmented = self.uav_transforms(image=patch_image, mask=patch_mask)
                patch_image = augmented['image']
                patch_mask = augmented['mask']
            except Exception as e:
                # Fallback: basic conversion
                if len(patch_image.shape) == 3:
                    patch_image = torch.from_numpy(patch_image).permute(2, 0, 1).float() / 255.0
                else:
                    patch_image = torch.from_numpy(patch_image).float() / 255.0
                patch_mask = torch.from_numpy(patch_mask).long()
        else:
            # Basic conversion
            if len(patch_image.shape) == 3:
                patch_image = torch.from_numpy(patch_image).permute(2, 0, 1).float() / 255.0
            else:
                patch_image = torch.from_numpy(patch_image).float() / 255.0
            patch_mask = torch.from_numpy(patch_mask).long()
        
        # Create sample
        sample = {
            'image': patch_image,
            'mask': patch_mask,
            'base_idx': patch_info['base_idx'],
            'patch_idx': patch_info['patch_idx'],
            'quality_score': patch_info['quality_score'],
            'patch_type': patch_info['patch_type'],
            'scale': patch_info['scale']
        }
        
        return sample


def create_cached_datasets(
    drone_deploy_dataset,
    udd_dataset, 
    semantic_drone_dataset,
    cache_root: str = "cache/augmented_datasets",
    augmentation_factors: Dict[str, int] = None,
    force_rebuild: bool = False
) -> Dict[str, CachedAugmentedDataset]:
    """Create cached augmented versions of all three datasets."""
    
    if augmentation_factors is None:
        augmentation_factors = {
            'semantic_drone': 25,    # Maximum for best dataset
            'drone_deploy': 20,      # High for 4-channel dataset
            'udd': 15               # Good for urban scenes
        }
    
    print(f"🏗️ Creating Cached Augmented Datasets")
    print(f"Cache directory: {cache_root}")
    print(f"Augmentation factors: {augmentation_factors}")
    print("=" * 60)
    
    cached_datasets = {}
    
    # Dataset configurations
    dataset_configs = [
        {
            'name': 'semantic_drone',
            'dataset': semantic_drone_dataset,
            'patch_scales': [(512, 512), (768, 768), (1024, 1024)],
            'min_object_ratio': 0.05,
            'overlap_ratio': 0.25,
        },
        {
            'name': 'drone_deploy',
            'dataset': drone_deploy_dataset,
            'patch_scales': [(512, 512), (768, 768)],
            'min_object_ratio': 0.1,
            'overlap_ratio': 0.3,
        },
        {
            'name': 'udd',
            'dataset': udd_dataset,
            'patch_scales': [(512, 512)],
            'min_object_ratio': 0.15,
            'overlap_ratio': 0.2,
        }
    ]
    
    # Create cached datasets
    for config in dataset_configs:
        name = config['name']
        print(f"\n🚀 Processing {name} dataset...")
        
        cached_datasets[name] = CachedAugmentedDataset(
            base_dataset=config['dataset'],
            cache_dir=cache_root,
            dataset_name=name,
            patch_scales=config['patch_scales'],
            augmentation_factor=augmentation_factors[name],
            min_object_ratio=config['min_object_ratio'],
            overlap_ratio=config['overlap_ratio'],
            use_overlapping=True,
            uav_augmentations=True,
            force_rebuild=force_rebuild,
            num_workers=4
        )
    
    # Summary
    total_base = sum(len(config['dataset']) for config in dataset_configs)
    total_augmented = sum(len(dataset) for dataset in cached_datasets.values())
    
    print(f"\n All datasets cached successfully!")
    print(f"📊 Summary:")
    print(f"   Total base images: {total_base}")
    print(f"   Total augmented patches: {total_augmented}")
    print(f"   Overall augmentation factor: {total_augmented/total_base:.1f}x")
    
    return cached_datasets


if __name__ == "__main__":
    print("🚀 Cached Augmentation System")
    print("=" * 50)
    print("This system creates and caches augmented datasets for fast reuse.")
    print("Run this from your training script to generate cached datasets.")
    
    # Example usage:
    print("\n📖 Example usage:")
    print("""
    from datasets.cached_augmentation import create_cached_datasets
    
    # Create cached datasets (one-time operation)
    cached_datasets = create_cached_datasets(
        drone_deploy_dataset, 
        udd_dataset, 
        semantic_drone_dataset,
        cache_root="cache/augmented_datasets",
        augmentation_factors={'semantic_drone': 25, 'drone_deploy': 20, 'udd': 15}
    )
    
    # Use in DataLoader (instant loading)
    train_loader = DataLoader(cached_datasets['semantic_drone'], batch_size=8, shuffle=True)
    """) 