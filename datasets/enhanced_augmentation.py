#!/usr/bin/env python3
"""
Enhanced Augmentation for UAV Landing Detection
==============================================

Multi-scale patch extraction and UAV-specific augmentations for:
1. Semantic Drone Dataset (6000x4000 → 77x more patches)
2. DroneDeploy Dataset (11K×8K+ → 20-30x more patches) 
3. UDD Dataset (4096×2160 → 8-12x more patches)

Features:
- Multi-scale patch extraction
- UAV-specific augmentations (motion blur, lighting, weather)
- Altitude simulation
- Quality filtering
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import albumentations as A
import random
from concurrent.futures import ThreadPoolExecutor
import pickle


class MultiScaleAugmentedDataset(Dataset):
    """Multi-scale augmented dataset that generates patches from high-resolution images."""
    
    def __init__(
        self,
        base_dataset: Dataset,
        patch_scales: List[Tuple[int, int]] = [(512, 512), (768, 768), (1024, 1024)],
        patches_per_image: int = 20,
        min_object_ratio: float = 0.05,
        use_overlapping: bool = True,
        overlap_ratio: float = 0.25,
        cache_patches: bool = True,
        uav_augmentations: bool = True
    ):
        """
        Args:
            base_dataset: Original dataset to augment
            patch_scales: List of (width, height) scales to extract
            patches_per_image: Number of patches per base image
            min_object_ratio: Minimum ratio of non-background pixels
            use_overlapping: Whether to use overlapping patches
            overlap_ratio: Overlap ratio for patches (0.0-0.5)
            cache_patches: Whether to cache extracted patches
            uav_augmentations: Whether to apply UAV-specific augmentations
        """
        self.base_dataset = base_dataset
        self.patch_scales = patch_scales
        self.patches_per_image = patches_per_image
        self.min_object_ratio = min_object_ratio
        self.use_overlapping = use_overlapping
        self.overlap_ratio = overlap_ratio
        self.cache_patches = cache_patches
        self.uav_augmentations = uav_augmentations
        
        # Cache
        self.patch_cache = {}
        self.metadata_cache = {}
        
        # Generate patch metadata
        self._generate_patch_metadata()
        
        # Create UAV-specific augmentations
        if self.uav_augmentations:
            self.uav_transforms = self._create_uav_augmentations()
        
        print(f"📊 MultiScaleAugmentedDataset initialized:")
        print(f"   Base dataset size: {len(self.base_dataset)}")
        print(f"   Augmented size: {len(self.patches)} patches")
        print(f"   Scales: {patch_scales}")
        print(f"   Augmentation factor: {len(self.patches)/len(self.base_dataset):.1f}x")
    
    def _generate_patch_metadata(self):
        """Pre-generate metadata for all patches."""
        self.patches = []
        
        for base_idx in range(len(self.base_dataset)):
            # Get base sample
            base_sample = self.base_dataset[base_idx]
            
            if isinstance(base_sample['image'], torch.Tensor):
                # Convert tensor to numpy for processing
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
                patches = self._extract_patches_from_image(
                    image, mask, base_idx, scale_w, scale_h
                )
                self.patches.extend(patches)
    
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
                'altitude_sim': 1.0  # Normal altitude
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
        
        # Add some random patches for diversity
        num_random = max(1, self.patches_per_image // 3)
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
        selected_patches = valid_patches[:self.patches_per_image]
        
        # Create patch metadata with altitude simulation
        for i, patch_info in enumerate(selected_patches):
            x, y = patch_info['coords']
            
            # Simulate different altitudes (affects resolution and perspective)
            altitude_factor = random.uniform(0.7, 1.3)  # 70%-130% of base altitude
            
            patches.append({
                'base_idx': base_idx,
                'patch_coords': (x, y, x + patch_w, y + patch_h),
                'scale': (patch_w, patch_h),
                'needs_resize': False,
                'altitude_sim': altitude_factor,
                'quality_score': patch_info['quality'],
                'non_bg_ratio': patch_info['non_bg_ratio']
            })
        
        return patches
    
    def _create_uav_augmentations(self) -> A.Compose:
        """Create UAV-specific augmentations."""
        
        # UAV-specific augmentation pipeline
        uav_transforms = A.Compose([
            # Motion blur (UAV movement, wind)
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            
            # Lighting conditions (altitude, weather, time of day)
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.3, 0.3), 
                    contrast_limit=(-0.2, 0.2), 
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            ], p=0.6),
            
            # Weather effects
            A.OneOf([
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, alpha_coef=0.1, p=1.0),
                A.RandomRain(
                    slant_lower=-10, slant_upper=10,
                    drop_length=10, drop_width=1,
                    drop_color=(200, 200, 200),
                    blur_value=1, p=1.0
                ),
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.5),
                    angle_lower=0, angle_upper=1,
                    num_flare_circles_lower=1, num_flare_circles_upper=2,
                    p=1.0
                ),
            ], p=0.2),
            
            # Shadow variations (clouds, objects)
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1, num_shadows_upper=2,
                shadow_dimension=5, p=0.15
            ),
            
            # Color variations (atmospheric effects, camera settings)
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
            
            # Noise (sensor noise, compression artifacts)
            A.OneOf([
                A.GaussNoise(var_limit=(5, 25), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.2),
            
            # Perspective variations (slight camera tilt)
            A.OneOf([
                A.Perspective(scale=(0.02, 0.05), p=1.0),
                A.ShiftScaleRotate(
                    shift_limit=0.05, 
                    scale_limit=0.1, 
                    rotate_limit=5, 
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0, p=1.0
                ),
            ], p=0.3),
            
            # Final normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.pytorch.ToTensorV2()
        ])
        
        return uav_transforms
    
    def __len__(self) -> int:
        return len(self.patches)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get an augmented patch."""
        patch_meta = self.patches[idx]
        base_idx = patch_meta['base_idx']
        
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
        
        # Extract patch
        if patch_meta['needs_resize']:
            # Small image, resize to target scale
            patch_image = cv2.resize(image, patch_meta['scale'])
            patch_mask = cv2.resize(mask, patch_meta['scale'], interpolation=cv2.INTER_NEAREST)
        else:
            # Extract patch from high-resolution image
            x1, y1, x2, y2 = patch_meta['patch_coords']
            patch_image = image[y1:y2, x1:x2].copy()
            patch_mask = mask[y1:y2, x1:x2].copy()
            
            # Resize to target scale if needed
            if patch_image.shape[:2] != patch_meta['scale'][::-1]:
                patch_image = cv2.resize(patch_image, patch_meta['scale'])
                patch_mask = cv2.resize(patch_mask, patch_meta['scale'], interpolation=cv2.INTER_NEAREST)
        
        # Apply altitude simulation (slight scaling)
        altitude_factor = patch_meta['altitude_sim']
        if altitude_factor != 1.0:
            h, w = patch_image.shape[:2]
            new_size = (int(w * altitude_factor), int(h * altitude_factor))
            patch_image = cv2.resize(patch_image, new_size)
            patch_mask = cv2.resize(patch_mask, new_size, interpolation=cv2.INTER_NEAREST)
            
            # Crop or pad to target size
            target_h, target_w = patch_meta['scale'][::-1]
            patch_image = self._crop_or_pad(patch_image, (target_h, target_w))
            patch_mask = self._crop_or_pad(patch_mask, (target_h, target_w))
        
        # Apply UAV-specific augmentations
        if self.uav_augmentations:
            try:
                augmented = self.uav_transforms(image=patch_image, mask=patch_mask)
                patch_image = augmented['image']
                patch_mask = augmented['mask']
            except Exception as e:
                # Fallback: basic conversion
                patch_image = torch.from_numpy(patch_image).permute(2, 0, 1).float() / 255.0
                patch_mask = torch.from_numpy(patch_mask).long()
        else:
            # Basic conversion
            patch_image = torch.from_numpy(patch_image).permute(2, 0, 1).float() / 255.0
            patch_mask = torch.from_numpy(patch_mask).long()
        
        # Create sample
        sample = {
            'image': patch_image,
            'mask': patch_mask,
            'base_idx': base_idx,
            'patch_coords': patch_meta['patch_coords'],
            'scale': patch_meta['scale'],
            'altitude_sim': patch_meta['altitude_sim'],
            'quality_score': patch_meta.get('quality_score', 1.0),
            'image_path': base_sample.get('image_path', ''),
        }
        
        return sample
    
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


def create_augmented_datasets(
    drone_deploy_dataset,
    udd_dataset, 
    semantic_drone_dataset,
    augmentation_config: Dict[str, Any] = None
) -> Dict[str, MultiScaleAugmentedDataset]:
    """Create augmented versions of all three datasets."""
    
    if augmentation_config is None:
        augmentation_config = {
            'semantic_drone': {
                'patch_scales': [(512, 512), (768, 768), (1024, 1024)],
                'patches_per_image': 25,  # High for best dataset
                'min_object_ratio': 0.05,
                'use_overlapping': True,
                'overlap_ratio': 0.25,
            },
            'drone_deploy': {
                'patch_scales': [(512, 512), (768, 768)],
                'patches_per_image': 15,  # Medium
                'min_object_ratio': 0.1,
                'use_overlapping': True,
                'overlap_ratio': 0.3,
            },
            'udd': {
                'patch_scales': [(512, 512)],
                'patches_per_image': 8,   # Lower for smaller base images
                'min_object_ratio': 0.15,
                'use_overlapping': True,
                'overlap_ratio': 0.2,
            }
        }
    
    augmented_datasets = {}
    
    # Create augmented datasets
    for name, dataset, config_key in [
        ('semantic_drone', semantic_drone_dataset, 'semantic_drone'),
        ('drone_deploy', drone_deploy_dataset, 'drone_deploy'), 
        ('udd', udd_dataset, 'udd')
    ]:
        print(f"\n🚀 Creating augmented {name} dataset...")
        
        augmented_datasets[name] = MultiScaleAugmentedDataset(
            base_dataset=dataset,
            uav_augmentations=True,
            **augmentation_config[config_key]
        )
    
    return augmented_datasets


if __name__ == "__main__":
    print("🚀 Enhanced Augmentation System")
    print("=" * 50)
    
    # This would be used in the training pipeline
    print("Ready for integration with progressive training!")
    print("Expected augmentation factors:")
    print("  - Semantic Drone: 25-77x (depending on scale)")
    print("  - DroneDeploy: 15-30x")
    print("  - UDD: 8-12x")
    print("Total training samples: ~50,000+ patches") 