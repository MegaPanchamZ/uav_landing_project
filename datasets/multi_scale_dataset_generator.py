#!/usr/bin/env python3
"""
Multi-Scale Dataset Generator for UAV Landing Detection
======================================================

Leverages the high resolution (6000×4000) of Semantic Drone Dataset to generate
multiple training samples at different scales and zoom levels, dramatically
increasing effective dataset size from 400 to 10,000+ samples.

Key Features:
- Multi-scale cropping: 512×512, 768×768, 1024×1024 patches
- Altitude simulation: Different zoom levels simulate various flight altitudes
- Overlap sampling: Sliding window with configurable overlap
- Quality filtering: Remove low-information patches
- Class balancing: Ensure diverse landing scenarios
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import json
from typing import Tuple, Dict, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


class MultiScaleDatasetGenerator:
    """
    Generate multiple training samples from high-resolution drone images.
    
    Transforms 400 base images → 10,000+ training samples through:
    - Multi-scale patches (3 sizes × 4-9 patches per size)
    - Altitude simulation (zoom levels representing 5m-50m flight height)
    - Quality-filtered sampling (remove sky-only, blur, low-detail patches)
    """
    
    def __init__(
        self,
        base_dataset_path: str,
        output_path: str,
        patch_sizes: List[Tuple[int, int]] = [(512, 512), (768, 768), (1024, 1024)],
        altitude_ranges: Dict[str, Tuple[float, float]] = None,
        overlap_ratio: float = 0.25,
        min_landing_pixels: int = 1000,
        quality_threshold: float = 0.3
    ):
        """
        Initialize multi-scale dataset generator.
        
        Args:
            base_dataset_path: Path to Semantic Drone Dataset
            output_path: Output directory for generated dataset
            patch_sizes: List of patch sizes to extract
            altitude_ranges: Altitude simulation ranges for each patch size
            overlap_ratio: Overlap between adjacent patches (0.0-0.8)
            min_landing_pixels: Minimum landing-relevant pixels per patch
            quality_threshold: Quality threshold for patch selection
        """
        self.base_path = Path(base_dataset_path)
        self.output_path = Path(output_path)
        self.patch_sizes = patch_sizes
        self.overlap_ratio = overlap_ratio
        self.min_landing_pixels = min_landing_pixels
        self.quality_threshold = quality_threshold
        
        # Altitude simulation: different patch sizes represent different flight heights
        if altitude_ranges is None:
            altitude_ranges = {
                (512, 512): (5.0, 15.0),    # Low altitude: detailed view
                (768, 768): (15.0, 30.0),   # Medium altitude: balanced view
                (1024, 1024): (30.0, 50.0) # High altitude: wide area view
            }
        self.altitude_ranges = altitude_ranges
        
        # Class mapping from Semantic Drone Dataset
        self.class_mapping = {
            0: 0, 23: 0,  # background, conflicting
            1: 1, 2: 1, 3: 1, 4: 1,  # safe: paved-area, dirt, grass, gravel
            6: 2, 8: 2, 9: 2, 21: 2,  # caution: rocks, vegetation, roof, ar-marker
            5: 3, 7: 3, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3,  # danger: water, pool, walls, etc.
            15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3, 22: 3   # danger: people, vehicles, trees, etc.
        }
        
        # Landing-relevant classes (for quality filtering)
        self.landing_classes = {1, 2, 3}  # safe, caution, danger
        
        # Initialize metadata
        self.generated_samples = []
        self.class_distribution = Counter()
        self.quality_stats = {}
        
        print(f"🔍 Multi-Scale Dataset Generator initialized:")
        print(f"   Base dataset: {self.base_path}")
        print(f"   Output path: {self.output_path}")
        print(f"   Patch sizes: {self.patch_sizes}")
        print(f"   Expected samples: {self.estimate_sample_count()}")
    
    def estimate_sample_count(self) -> int:
        """Estimate total number of samples that will be generated."""
        base_image_count = 400  # Semantic Drone Dataset size
        
        total_samples = 0
        for patch_size in self.patch_sizes:
            # Calculate patches per image for this size
            patches_per_row = int((6000 - patch_size[0]) / (patch_size[0] * (1 - self.overlap_ratio))) + 1
            patches_per_col = int((4000 - patch_size[1]) / (patch_size[1] * (1 - self.overlap_ratio))) + 1
            patches_per_image = patches_per_row * patches_per_col
            
            # Estimate after quality filtering (assume 60% pass)
            quality_filtered = int(patches_per_image * 0.6)
            total_samples += base_image_count * quality_filtered
        
        return total_samples
    
    def generate_dataset(self, train_split: float = 0.7, val_split: float = 0.15):
        """
        Generate multi-scale dataset from base images.
        
        Args:
            train_split: Fraction for training set
            val_split: Fraction for validation set (remainder goes to test)
        """
        print("🚀 Starting multi-scale dataset generation...")
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'val', 'test']:
            (self.output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Load base dataset file lists
        images_dir = self.base_path / "original_images"
        labels_dir = self.base_path / "label_images_semantic"
        
        image_files = sorted(list(images_dir.glob("*.jpg")))
        label_files = sorted(list(labels_dir.glob("*.png")))
        
        if len(image_files) == 0:
            raise ValueError(f"No images found in {images_dir}")
        
        print(f"📊 Found {len(image_files)} base images")
        
        # Process each base image
        total_generated = 0
        quality_rejected = 0
        
        for img_idx, (img_path, label_path) in enumerate(tqdm(
            zip(image_files, label_files), 
            desc="Processing base images",
            total=len(image_files)
        )):
            # Load base image and label
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            
            if image is None or label is None:
                print(f"⚠️ Failed to load {img_path.name}")
                continue
            
            # Generate patches for each scale
            for patch_size in self.patch_sizes:
                patches = self._extract_patches(image, label, patch_size, img_idx)
                
                for patch_data in patches:
                    # Quality filtering
                    if self._assess_patch_quality(patch_data):
                        # Determine split
                        split = self._determine_split(total_generated, train_split, val_split)
                        
                        # Save patch
                        self._save_patch(patch_data, split, total_generated)
                        total_generated += 1
                    else:
                        quality_rejected += 1
        
        print(f" Dataset generation completed!")
        print(f"   Total samples generated: {total_generated}")
        print(f"   Quality rejected: {quality_rejected}")
        print(f"   Quality acceptance rate: {total_generated/(total_generated+quality_rejected)*100:.1f}%")
        
        # Save metadata
        self._save_metadata(total_generated, train_split, val_split)
        
        # Generate analysis report
        self._generate_analysis_report()
    
    def _extract_patches(
        self, 
        image: np.ndarray, 
        label: np.ndarray, 
        patch_size: Tuple[int, int],
        img_idx: int
    ) -> List[Dict]:
        """Extract overlapping patches from base image."""
        
        patches = []
        img_h, img_w = image.shape[:2]
        patch_h, patch_w = patch_size
        
        # Calculate stride based on overlap
        stride_h = int(patch_h * (1 - self.overlap_ratio))
        stride_w = int(patch_w * (1 - self.overlap_ratio))
        
        # Simulate altitude for this patch size
        altitude_range = self.altitude_ranges.get(patch_size, (10.0, 30.0))
        
        y = 0
        while y + patch_h <= img_h:
            x = 0
            while x + patch_w <= img_w:
                # Extract patch
                patch_img = image[y:y+patch_h, x:x+patch_w]
                patch_label = label[y:y+patch_h, x:x+patch_w]
                
                # Map to landing classes
                mapped_label = self._map_classes(patch_label)
                
                # Simulate altitude (random within range for this patch size)
                simulated_altitude = random.uniform(*altitude_range)
                
                # Create patch metadata
                patch_data = {
                    'image': patch_img,
                    'label': mapped_label,
                    'original_label': patch_label,
                    'patch_size': patch_size,
                    'position': (x, y),
                    'base_image_idx': img_idx,
                    'simulated_altitude': simulated_altitude,
                    'metadata': {
                        'source_image_size': (img_w, img_h),
                        'patch_coord': (x, y, x+patch_w, y+patch_h),
                        'coverage_area_m2': self._calculate_coverage_area(patch_size, simulated_altitude)
                    }
                }
                
                patches.append(patch_data)
                x += stride_w
            y += stride_h
        
        return patches
    
    def _map_classes(self, label: np.ndarray) -> np.ndarray:
        """Map original 24 classes to 4 landing classes."""
        mapped = np.zeros_like(label, dtype=np.uint8)
        for orig_class, landing_class in self.class_mapping.items():
            mapped[label == orig_class] = landing_class
        return mapped
    
    def _assess_patch_quality(self, patch_data: Dict) -> bool:
        """
        Assess if patch is suitable for training.
        
        Quality criteria:
        - Sufficient landing-relevant content
        - Good contrast and detail
        - Not predominantly sky or uniform areas
        - Balanced class distribution
        """
        image = patch_data['image']
        label = patch_data['label']
        
        # 1. Check landing-relevant pixel count
        landing_pixels = np.isin(label, list(self.landing_classes)).sum()
        if landing_pixels < self.min_landing_pixels:
            return False
        
        # 2. Check image quality metrics
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Contrast (standard deviation)
        contrast = np.std(gray)
        if contrast < 20:  # Too uniform
            return False
        
        # Detail level (Laplacian variance)
        detail = cv2.Laplacian(gray, cv2.CV_64F).var()
        if detail < 100:  # Too blurry or low detail
            return False
        
        # 3. Sky detection (avoid sky-dominant patches)
        # Simple heuristic: top 1/3 with low variance = likely sky
        top_third = gray[:gray.shape[0]//3]
        if np.std(top_third) < 15 and np.mean(top_third) > 180:
            return False
        
        # 4. Class diversity (avoid single-class patches)
        unique_classes = np.unique(label)
        if len(unique_classes) < 2:
            return False
        
        return True
    
    def _calculate_coverage_area(self, patch_size: Tuple[int, int], altitude: float) -> float:
        """Calculate real-world coverage area in square meters."""
        # Simplified calculation based on typical drone camera specs
        # Assume 84° FOV horizontal, 16:9 aspect ratio
        
        patch_h, patch_w = patch_size
        
        # Ground sampling distance (GSD) in meters per pixel
        # Formula: GSD = (altitude × sensor_size) / (focal_length × image_size)
        # Simplified approximation for typical drone camera
        gsd = altitude * 0.0024  # Approximate for DJI-style cameras
        
        # Real-world patch dimensions
        real_width = patch_w * gsd
        real_height = patch_h * gsd
        
        return real_width * real_height
    
    def _determine_split(self, sample_idx: int, train_split: float, val_split: float) -> str:
        """Determine which split this sample belongs to."""
        # Use deterministic splitting based on sample index
        rand_val = (sample_idx * 2654435761) % 1000 / 1000.0  # Hash-based random
        
        if rand_val < train_split:
            return 'train'
        elif rand_val < train_split + val_split:
            return 'val'
        else:
            return 'test'
    
    def _save_patch(self, patch_data: Dict, split: str, sample_idx: int):
        """Save individual patch to appropriate split directory."""
        
        # Generate filename
        base_idx = patch_data['base_image_idx']
        x, y = patch_data['position']
        patch_h, patch_w = patch_data['patch_size']
        
        filename = f"patch_{base_idx:04d}_{x:04d}_{y:04d}_{patch_w}x{patch_h}_{sample_idx:06d}"
        
        # Save image
        img_path = self.output_path / split / 'images' / f"{filename}.jpg"
        image_bgr = cv2.cvtColor(patch_data['image'], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(img_path), image_bgr)
        
        # Save label
        label_path = self.output_path / split / 'labels' / f"{filename}.png"
        cv2.imwrite(str(label_path), patch_data['label'])
        
        # Update statistics
        unique, counts = np.unique(patch_data['label'], return_counts=True)
        for class_id, count in zip(unique, counts):
            self.class_distribution[class_id] += count
        
        # Store sample metadata
        self.generated_samples.append({
            'filename': filename,
            'split': split,
            'patch_size': patch_data['patch_size'],
            'simulated_altitude': patch_data['simulated_altitude'],
            'coverage_area_m2': patch_data['metadata']['coverage_area_m2'],
            'base_image_idx': patch_data['base_image_idx'],
            'position': patch_data['position']
        })
    
    def _save_metadata(self, total_samples: int, train_split: float, val_split: float):
        """Save dataset metadata and statistics."""
        
        metadata = {
            'generation_info': {
                'total_base_images': 400,
                'total_generated_samples': total_samples,
                'patch_sizes': self.patch_sizes,
                'overlap_ratio': self.overlap_ratio,
                'altitude_ranges': {str(k): v for k, v in self.altitude_ranges.items()},
                'splits': {
                    'train': train_split,
                    'val': val_split,
                    'test': 1.0 - train_split - val_split
                }
            },
            'class_distribution': dict(self.class_distribution),
            'class_names': {
                0: 'background',
                1: 'safe_landing',
                2: 'caution',
                3: 'danger'
            },
            'quality_criteria': {
                'min_landing_pixels': self.min_landing_pixels,
                'quality_threshold': self.quality_threshold,
                'contrast_threshold': 20,
                'detail_threshold': 100
            },
            'sample_metadata': self.generated_samples
        }
        
        # Save as JSON
        metadata_path = self.output_path / 'dataset_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Metadata saved to {metadata_path}")
    
    def _generate_analysis_report(self):
        """Generate detailed analysis report of the generated dataset."""
        
        print("\n📊 MULTI-SCALE DATASET ANALYSIS REPORT")
        print("=" * 60)
        
        # Split distribution
        split_counts = Counter(sample['split'] for sample in self.generated_samples)
        print(f"\n📈 Sample Distribution:")
        for split, count in split_counts.items():
            percentage = count / len(self.generated_samples) * 100
            print(f"   {split.capitalize()}: {count:,} samples ({percentage:.1f}%)")
        
        # Patch size distribution
        size_counts = Counter(str(sample['patch_size']) for sample in self.generated_samples)
        print(f"\n🔍 Patch Size Distribution:")
        for size, count in size_counts.items():
            percentage = count / len(self.generated_samples) * 100
            print(f"   {size}: {count:,} samples ({percentage:.1f}%)")
        
        # Altitude simulation analysis
        altitudes = [sample['simulated_altitude'] for sample in self.generated_samples]
        print(f"\n✈️ Altitude Simulation:")
        print(f"   Range: {min(altitudes):.1f}m - {max(altitudes):.1f}m")
        print(f"   Mean: {np.mean(altitudes):.1f}m")
        print(f"   Std: {np.std(altitudes):.1f}m")
        
        # Coverage area analysis
        areas = [sample['coverage_area_m2'] for sample in self.generated_samples]
        print(f"\n🌍 Coverage Area:")
        print(f"   Range: {min(areas):.1f} - {max(areas):.1f} m²")
        print(f"   Mean: {np.mean(areas):.1f} m²")
        print(f"   Total coverage: {sum(areas)/1000000:.1f} km²")
        
        # Class distribution
        total_pixels = sum(self.class_distribution.values())
        print(f"\n Class Distribution (pixels):")
        class_names = {0: 'background', 1: 'safe_landing', 2: 'caution', 3: 'danger'}
        for class_id in sorted(self.class_distribution.keys()):
            count = self.class_distribution[class_id]
            percentage = count / total_pixels * 100
            print(f"   {class_names.get(class_id, f'class_{class_id}')}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n Dataset generation multiplier: {len(self.generated_samples) / 400:.1f}x")
        print(f"   400 base images → {len(self.generated_samples):,} training samples")


class MultiScaleDataset(Dataset):
    """Dataset class for loading generated multi-scale patches."""
    
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        target_size: Tuple[int, int] = (512, 512)
    ):
        """
        Initialize multi-scale dataset.
        
        Args:
            dataset_path: Path to generated multi-scale dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Albumentations transform pipeline
            target_size: Target size for all patches (will resize if needed)
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        
        # Load metadata
        metadata_path = self.dataset_path / 'dataset_metadata.json'
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Get file lists for this split
        images_dir = self.dataset_path / split / 'images'
        labels_dir = self.dataset_path / split / 'labels'
        
        self.image_files = sorted(list(images_dir.glob("*.jpg")))
        self.label_files = sorted(list(labels_dir.glob("*.png")))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found for split '{split}' in {images_dir}")
        
        print(f"📊 MultiScaleDataset loaded: {len(self.image_files)} {split} samples")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        
        # Load image and label
        image = cv2.imread(str(self.image_files[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(str(self.label_files[idx]), cv2.IMREAD_GRAYSCALE)
        
        # Resize to target size if needed
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
        
        # Get sample metadata if available
        filename = self.image_files[idx].stem
        sample_metadata = next(
            (s for s in self.metadata['sample_metadata'] if s['filename'] == filename),
            {}
        )
        
        return {
            'image': image,
            'mask': label,
            'filename': filename,
            'altitude': sample_metadata.get('simulated_altitude', 20.0),
            'coverage_area': sample_metadata.get('coverage_area_m2', 10000.0),
            'patch_size': sample_metadata.get('patch_size', self.target_size)
        }


def create_multi_scale_transforms(target_size: Tuple[int, int] = (512, 512)) -> Tuple[A.Compose, A.Compose]:
    """Create train and validation transforms for multi-scale dataset."""
    
    train_transform = A.Compose([
        A.Resize(target_size[0], target_size[1], interpolation=cv2.INTER_LINEAR),
        
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        
        # Color augmentations (lighter since we have altitude variation)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
        
        # Weather simulation
        A.OneOf([
            A.RandomShadow(p=1.0),
            A.RandomSunFlare(p=1.0),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.2, p=1.0),
        ], p=0.2),
        
        # Normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(target_size[0], target_size[1], interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transform, val_transform


if __name__ == "__main__":
    # Example usage
    base_dataset_path = "../datasets/semantic_drone_dataset"
    output_path = "../datasets/multi_scale_semantic_drone"
    
    # Generate multi-scale dataset
    generator = MultiScaleDatasetGenerator(
        base_dataset_path=base_dataset_path,
        output_path=output_path,
        patch_sizes=[(512, 512), (768, 768), (1024, 1024)],
        overlap_ratio=0.25,
        min_landing_pixels=1000
    )
    
    print("🚀 Starting dataset generation...")
    generator.generate_dataset()
    
    # Test loading the generated dataset
    train_transform, val_transform = create_multi_scale_transforms()
    
    train_dataset = MultiScaleDataset(
        output_path, 
        split="train", 
        transform=train_transform
    )
    
    print(f"\n Generated dataset ready!")
    print(f"   Training samples: {len(train_dataset)}")
    
    # Test sample
    sample = train_dataset[0]
    print(f"   Sample shape: {sample['image'].shape}")
    print(f"   Simulated altitude: {sample['altitude']:.1f}m")
    print(f"   Coverage area: {sample['coverage_area']:.1f}m²") 