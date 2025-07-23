#!/usr/bin/env python3
"""
Aerial Semantic Dataset - Full 24-Class Preservation
===================================================

This dataset loader preserves all 24 semantic classes from the Semantic Drone Dataset
instead of mapping them to crude 4-class landing categories.

The neural network will output rich 24-class semantic segmentations, and Scallop
will handle the landing logic reasoning separately.
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Tuple, Dict, Optional, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from collections import Counter


class AerialSemantic24Dataset(Dataset):
    """
    Semantic Drone Dataset with full 24-class preservation for neuro-symbolic UAV landing.
    
    This dataset maintains all semantic richness instead of reducing to 4 crude categories.
    The neural network will learn rich aerial semantics, and Scallop will handle landing logic.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        target_resolution: Tuple[int, int] = (512, 512),
        return_confidence: bool = False,
        cache_images: bool = True,
        use_random_crops: bool = True,
        crops_per_image: int = 6
    ):
        """
        Initialize Aerial Semantic 24-class Dataset.
        
        Args:
            data_root: Path to semantic_drone_dataset directory
            split: Dataset split ('train', 'val', 'test')
            transform: Albumentations transform pipeline
            target_resolution: Target image resolution for training
            return_confidence: Whether to return confidence maps
            cache_images: Cache images in memory for faster training
            use_random_crops: Use random crops for data augmentation
            crops_per_image: Number of random crops per image
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.target_resolution = target_resolution
        self.return_confidence = return_confidence
        self.cache_images = cache_images
        self.use_random_crops = use_random_crops
        self.crops_per_image = crops_per_image if split == "train" else 1
        
        # Full 24 semantic classes (NO REDUCTION)
        self.class_info = {
            0: {"name": "unlabeled", "rgb": (0, 0, 0), "landing_relevance": "unknown"},
            1: {"name": "paved-area", "rgb": (128, 64, 128), "landing_relevance": "safe"},
            2: {"name": "dirt", "rgb": (130, 76, 0), "landing_relevance": "safe"},
            3: {"name": "grass", "rgb": (0, 102, 0), "landing_relevance": "safe"},
            4: {"name": "gravel", "rgb": (112, 103, 87), "landing_relevance": "safe"},
            5: {"name": "water", "rgb": (28, 42, 168), "landing_relevance": "danger"},
            6: {"name": "rocks", "rgb": (48, 41, 30), "landing_relevance": "caution"},
            7: {"name": "pool", "rgb": (0, 50, 89), "landing_relevance": "danger"},
            8: {"name": "vegetation", "rgb": (107, 142, 35), "landing_relevance": "caution"},
            9: {"name": "roof", "rgb": (70, 70, 70), "landing_relevance": "caution"},
            10: {"name": "wall", "rgb": (102, 102, 156), "landing_relevance": "obstacle"},
            11: {"name": "window", "rgb": (254, 228, 12), "landing_relevance": "landmark"},
            12: {"name": "door", "rgb": (254, 148, 12), "landing_relevance": "landmark"},
            13: {"name": "fence", "rgb": (190, 153, 153), "landing_relevance": "obstacle"},
            14: {"name": "fence-pole", "rgb": (153, 153, 153), "landing_relevance": "obstacle"},
            15: {"name": "person", "rgb": (255, 22, 96), "landing_relevance": "danger"},
            16: {"name": "dog", "rgb": (102, 51, 0), "landing_relevance": "danger"},
            17: {"name": "car", "rgb": (9, 143, 150), "landing_relevance": "danger"},
            18: {"name": "bicycle", "rgb": (119, 11, 32), "landing_relevance": "danger"},
            19: {"name": "tree", "rgb": (51, 51, 0), "landing_relevance": "danger"},
            20: {"name": "bald-tree", "rgb": (190, 250, 190), "landing_relevance": "caution"},
            21: {"name": "ar-marker", "rgb": (112, 150, 146), "landing_relevance": "landmark"},
            22: {"name": "obstacle", "rgb": (2, 135, 115), "landing_relevance": "danger"},
            23: {"name": "conflicting", "rgb": (255, 0, 0), "landing_relevance": "unknown"}
        }
        
        # Create RGB to class ID mapping
        self.rgb_to_class = {}
        for class_id, info in self.class_info.items():
            self.rgb_to_class[info["rgb"]] = class_id
        
        # Load dataset files
        self.images_dir = self.data_root / "original_images"
        self.labels_dir = self.data_root / "label_images_semantic"
        
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise ValueError(f"Labels directory not found: {self.labels_dir}")
        
        # Load image and label files
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        self.label_files = sorted(list(self.labels_dir.glob("*.png")))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")
        if len(self.label_files) == 0:
            raise ValueError(f"No labels found in {self.labels_dir}")
        
        # Create train/val/test splits
        self._create_splits()
        
        # Image cache for faster training
        self.image_cache = {} if cache_images else None
        self.label_cache = {} if cache_images else None
        
        print(f"🛩️  AerialSemantic24Dataset initialized:")
        print(f"   Split: {split} ({len(self.file_indices)} images)")
        print(f"   Effective samples: {len(self)} (with crops)")
        print(f"   Classes: 24 semantic classes (FULL RICHNESS)")
        print(f"   Resolution: {target_resolution}")
        print(f"   Crops per image: {crops_per_image}")
        
    def _create_splits(self):
        """Create train/val/test splits with stratification."""
        total_files = len(self.image_files)
        indices = np.arange(total_files)
        
        # Stratified splits for reproducibility
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
    
    def __len__(self):
        return len(self.file_indices) * self.crops_per_image
    
    def __getitem__(self, idx):
        # Determine which image and which crop
        image_idx = idx // self.crops_per_image
        crop_idx = idx % self.crops_per_image
        
        file_idx = self.file_indices[image_idx]
        image_path = self.image_files[file_idx]
        label_path = self.label_files[file_idx]
        
        # Load image
        if self.image_cache is not None and file_idx in self.image_cache:
            image = self.image_cache[file_idx]
        else:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.image_cache is not None:
                self.image_cache[file_idx] = image
        
        # Load label
        if self.label_cache is not None and file_idx in self.label_cache:
            label_rgb = self.label_cache[file_idx]
        else:
            label_rgb = cv2.imread(str(label_path))
            if label_rgb is None:
                raise ValueError(f"Could not load label: {label_path}")
            label_rgb = cv2.cvtColor(label_rgb, cv2.COLOR_BGR2RGB)
            if self.label_cache is not None:
                self.label_cache[file_idx] = label_rgb
        
        # Random cropping for data augmentation and memory efficiency
        if self.use_random_crops and self.split == "train":
            image, label_rgb = self._random_crop(image, label_rgb, crop_idx)
        else:
            # For validation/test, resize to target resolution
            image = cv2.resize(image, self.target_resolution, interpolation=cv2.INTER_LINEAR)
            label_rgb = cv2.resize(label_rgb, self.target_resolution, interpolation=cv2.INTER_NEAREST)
        
        # Convert RGB label to class IDs (24 classes)
        label = self._rgb_to_class_ids(label_rgb)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
        
        # Convert to tensors if not already done by transforms
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label).long()
        
        sample = {
            'image': image,
            'mask': label,
            'image_path': str(image_path),
            'class_names': [self.class_info[i.item()]["name"] for i in torch.unique(label)],
            'original_shape': label_rgb.shape[:2]
        }
        
        # Add confidence map if requested
        if self.return_confidence:
            confidence = self._compute_confidence_map(label_rgb, label)
            sample['confidence'] = torch.from_numpy(confidence).float()
        
        return sample
    
    def _random_crop(self, image: np.ndarray, label_rgb: np.ndarray, crop_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract random crop for data augmentation."""
        crop_h, crop_w = self.target_resolution
        img_h, img_w = image.shape[:2]
        
        # Ensure crop size doesn't exceed image size
        crop_h = min(crop_h, img_h)
        crop_w = min(crop_w, img_w)
        
        # Deterministic crop based on crop_idx for reproducibility during validation
        if self.split != "train":
            np.random.seed(crop_idx)
        
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
        image_crop = image[start_h:start_h + crop_h, start_w:start_w + crop_w]
        label_crop = label_rgb[start_h:start_h + crop_h, start_w:start_w + crop_w]
        
        return image_crop, label_crop
    
    def _rgb_to_class_ids(self, label_rgb: np.ndarray) -> np.ndarray:
        """Convert RGB label to class IDs (24 classes)."""
        h, w = label_rgb.shape[:2]
        label = np.zeros((h, w), dtype=np.uint8)
        
        # Map each RGB color to its class ID
        for rgb_color, class_id in self.rgb_to_class.items():
            # Find pixels matching this RGB color
            mask = np.all(label_rgb == np.array(rgb_color), axis=2)
            label[mask] = class_id
        
        return label
    
    def _compute_confidence_map(self, label_rgb: np.ndarray, label: np.ndarray) -> np.ndarray:
        """Compute confidence map based on class reliability."""
        confidence = np.ones_like(label, dtype=np.float32)
        
        # Lower confidence for ambiguous/conflicting classes
        conflicting_mask = (label == 23)  # conflicting class
        unlabeled_mask = (label == 0)     # unlabeled class
        confidence[conflicting_mask] = 0.3
        confidence[unlabeled_mask] = 0.5
        
        # High confidence for clear semantic classes
        clear_classes = [1, 2, 3, 4, 5, 15, 16, 17, 18, 19, 22]  # paved, dirt, grass, water, people, vehicles, etc.
        for class_id in clear_classes:
            mask = (label == class_id)
            confidence[mask] = 1.0
        
        # Medium confidence for contextual classes
        contextual_classes = [6, 8, 9, 20, 21]  # rocks, vegetation, roof, bald-tree, ar-marker
        for class_id in contextual_classes:
            mask = (label == class_id)
            confidence[mask] = 0.8
        
        return confidence
    
    def get_class_distribution(self) -> Dict[str, float]:
        """Analyze class distribution in the dataset."""
        class_counts = Counter()
        total_pixels = 0
        
        print("📊 Analyzing 24-class distribution...")
        sample_size = min(len(self.file_indices), 20)  # Sample subset for analysis
        
        for i in range(sample_size):
            file_idx = self.file_indices[i]
            label_path = self.label_files[file_idx]
            
            # Load and process label
            label_rgb = cv2.imread(str(label_path))
            label_rgb = cv2.cvtColor(label_rgb, cv2.COLOR_BGR2RGB)
            label = self._rgb_to_class_ids(label_rgb)
            
            unique, counts = np.unique(label, return_counts=True)
            for class_id, count in zip(unique, counts):
                class_counts[class_id] += count
                total_pixels += count
        
        # Convert to percentages and class names
        distribution = {}
        for class_id, count in class_counts.items():
            class_name = self.class_info[class_id]["name"]
            percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
            distribution[class_name] = {
                "percentage": percentage,
                "count": count,
                "landing_relevance": self.class_info[class_id]["landing_relevance"]
            }
        
        return distribution
    
    def get_landing_relevant_stats(self) -> Dict[str, float]:
        """Get statistics grouped by landing relevance."""
        distribution = self.get_class_distribution()
        
        relevance_stats = {
            "safe": 0.0,
            "caution": 0.0, 
            "danger": 0.0,
            "obstacle": 0.0,
            "landmark": 0.0,
            "unknown": 0.0
        }
        
        for class_name, stats in distribution.items():
            relevance = stats["landing_relevance"]
            relevance_stats[relevance] += stats["percentage"]
        
        return relevance_stats


def create_aerial_semantic_transforms(
    input_size: Tuple[int, int] = (512, 512),
    is_training: bool = True,
    advanced_augmentation: bool = True
) -> A.Compose:
    """Create transforms optimized for aerial semantic segmentation."""
    
    transforms = []
    
    if is_training and advanced_augmentation:
        # Aerial-specific augmentations
        transforms.extend([
            A.RandomRotate90(p=0.8),  # Critical for aerial views - all rotations valid
            A.Flip(p=0.7),           # Horizontal and vertical flips valid
            A.RandomBrightnessContrast(p=0.8, brightness_limit=0.3, contrast_limit=0.3),
            A.HueSaturationValue(p=0.6, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
            A.RandomGamma(p=0.4, gamma_limit=(80, 120)),
            A.CLAHE(p=0.3, clip_limit=4.0),  # Improve contrast
            A.RandomShadow(p=0.3),  # Simulate cloud shadows
            A.RandomFog(p=0.2, fog_coef_lower=0.1, fog_coef_upper=0.3),  # Weather simulation
        ])
    elif is_training:
        # Basic augmentations
        transforms.extend([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ])
    
    # Always apply normalization and tensor conversion
    transforms.extend([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # ImageNet normalization
        ToTensorV2(),
    ])
    
    return A.Compose(transforms)


def test_dataset():
    """Test the dataset loading and processing."""
    try:
        # Test dataset loading
        data_root = "../datasets/Aerial_Semantic_Segmentation_Drone_Dataset/dataset/semantic_drone_dataset"
        
        transforms = create_aerial_semantic_transforms(
            input_size=(512, 512),
            is_training=True,
            advanced_augmentation=True
        )
        
        dataset = AerialSemantic24Dataset(
            data_root=data_root,
            split="train",
            transform=transforms,
            use_random_crops=True,
            crops_per_image=4
        )
        
        print(f"\n Dataset test successful!")
        print(f"   Dataset size: {len(dataset)} samples")
        
        # Test sample loading
        sample = dataset[0]
        print(f"   Sample image shape: {sample['image'].shape}")
        print(f"   Sample mask shape: {sample['mask'].shape}")
        print(f"   Unique classes in sample: {torch.unique(sample['mask'])}")
        print(f"   Class names: {sample['class_names']}")
        
        # Analyze class distribution
        distribution = dataset.get_class_distribution()
        print(f"\n📊 Class Distribution (top 10):")
        sorted_classes = sorted(distribution.items(), key=lambda x: x[1]["percentage"], reverse=True)
        for class_name, stats in sorted_classes[:10]:
            print(f"   {class_name}: {stats['percentage']:.2f}% ({stats['landing_relevance']})")
        
        # Landing relevance stats
        relevance_stats = dataset.get_landing_relevant_stats()
        print(f"\n Landing Relevance Distribution:")
        for relevance, percentage in relevance_stats.items():
            print(f"   {relevance}: {percentage:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False


if __name__ == "__main__":
    test_dataset() 