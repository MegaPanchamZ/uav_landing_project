#!/usr/bin/env python3
"""
UDD Dataset (Urban Drone Dataset)
================================

Dataset loader for UDD5/UDD6 urban drone dataset.
Supports UAV landing detection with proper class mappings.

Original UDD Classes:
- 0: Other
- 1: Facade (Buildings)
- 2: Road
- 3: Vegetation
- 4: Vehicle
- 5: Roof

Landing Mapping:
- 0: Background (Other)
- 1: Safe (Road)
- 2: Caution (Vegetation, Roof)
- 3: Danger (Facade, Vehicle)
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import albumentations as A


class UDDDataset(Dataset):
    """UDD dataset for UAV landing detection."""
    
    # Original 6 classes to 4 landing classes mapping
    CLASS_MAPPING = {
        0: 0,  # Other → Background
        1: 3,  # Facade → Danger (buildings)
        2: 1,  # Road → Safe
        3: 2,  # Vegetation → Caution
        4: 3,  # Vehicle → Danger
        5: 2,  # Roof → Caution (could be landing zone)
    }
    
    LANDING_CLASSES = ['Background', 'Safe', 'Caution', 'Danger']
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[A.Compose] = None,
        target_resolution: Tuple[int, int] = (512, 512),
        return_confidence: bool = False
    ):
        """
        Initialize UDD dataset.
        
        Args:
            data_root: Path to UDD5 or UDD6 directory
            split: 'train' or 'val'
            transform: Albumentations transforms
            target_resolution: Target image size
            return_confidence: Whether to return confidence maps
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.target_resolution = target_resolution
        self.return_confidence = return_confidence
        
        # Paths
        self.images_dir = self.data_root / split / "src"
        self.labels_dir = self.data_root / split / "gt"
        
        # Validate paths
        if not all(p.exists() for p in [self.images_dir, self.labels_dir]):
            raise ValueError(f"UDD directories not found in {data_root}")
            
        # Load image and label pairs
        self.samples = self._load_samples()
        
        print(f"📊 UDDDataset initialized:")
        print(f"   Split: {split} ({len(self.samples)} samples)")
        print(f"   Classes: 4 landing classes")
        print(f"   Resolution: {target_resolution}")
        print(f"   Dataset: {self.data_root.name}")
        
    def _load_samples(self) -> list:
        """Load image and label file pairs."""
        samples = []
        
        # Get all image files
        image_files = list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))
        
        for image_path in sorted(image_files):
            # Find corresponding label file
            label_name = image_path.stem + ".png"
            label_path = self.labels_dir / label_name
            
            if label_path.exists():
                samples.append({
                    'image_path': image_path,
                    'label_path': label_path,
                    'image_id': image_path.stem
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample."""
        sample_info = self.samples[idx]
        
        # Load RGB image
        image_path = sample_info['image_path']
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load label
        label_path = sample_info['label_path']
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise ValueError(f"Could not load label: {label_path}")
        
        # Map classes to landing classes
        mapped_label = self._map_classes(label)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mapped_label)
            image, mapped_label = transformed['image'], transformed['mask']
        
        # Convert to tensors if not already done by transforms
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if not isinstance(mapped_label, torch.Tensor):
            mapped_label = torch.from_numpy(mapped_label).long()
        
        sample = {
            'image': image,
            'mask': mapped_label,
            'image_path': str(image_path),
            'original_shape': label.shape,
            'image_id': sample_info['image_id']
        }
        
        # Add confidence map if requested
        if self.return_confidence:
            confidence = self._compute_confidence_map(label, mapped_label)
            sample['confidence'] = torch.from_numpy(confidence).float()
        
        return sample
    
    def _map_classes(self, label: np.ndarray) -> np.ndarray:
        """Map original 6 classes to 4 landing classes."""
        mapped_label = np.zeros_like(label, dtype=np.uint8)
        
        for original_class, landing_class in self.CLASS_MAPPING.items():
            mask = (label == original_class)
            mapped_label[mask] = landing_class
        
        return mapped_label
    
    def _compute_confidence_map(self, original_label: np.ndarray, mapped_label: np.ndarray) -> np.ndarray:
        """Compute confidence map based on class certainty."""
        confidence = np.ones_like(mapped_label, dtype=np.float32)
        
        # Lower confidence for ambiguous classes
        vegetation_mask = (original_label == 3)  # Vegetation varies in safety
        roof_mask = (original_label == 5)  # Roofs can be safe or dangerous
        confidence[vegetation_mask] = 0.7
        confidence[roof_mask] = 0.6  # Lower confidence for roofs
        
        # High confidence for clear classes
        road_mask = (original_label == 2)  # Roads clearly safe
        facade_mask = (original_label == 1)  # Buildings clearly dangerous
        vehicle_mask = (original_label == 4)  # Vehicles clearly dangerous
        confidence[road_mask] = 1.0
        confidence[facade_mask] = 1.0
        confidence[vehicle_mask] = 1.0
        
        return confidence
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution for the current split."""
        class_counts = {cls: 0 for cls in self.LANDING_CLASSES}
        
        for i in range(len(self)):
            sample = self[i]
            mask = sample['mask'].numpy()
            unique, counts = np.unique(mask, return_counts=True)
            
            for class_id, count in zip(unique, counts):
                if class_id < len(self.LANDING_CLASSES):
                    class_counts[self.LANDING_CLASSES[class_id]] += count
        
        return class_counts


def create_udd_transforms(
    input_resolution: Tuple[int, int] = (512, 512),
    is_training: bool = True
) -> A.Compose:
    """Create transforms for UDD dataset."""
    
    if is_training:
        transforms = [
            A.Resize(height=input_resolution[0], width=input_resolution[1]),
            A.RandomCrop(height=input_resolution[0], width=input_resolution[1], p=0.3),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.pytorch.ToTensorV2()
        ]
    else:
        transforms = [
            A.Resize(height=input_resolution[0], width=input_resolution[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.pytorch.ToTensorV2()
        ]
    
    return A.Compose(transforms)


if __name__ == "__main__":
    # Test the dataset
    dataset_path = "../datasets/UDD/UDD/UDD5"
    
    # Test dataset
    dataset = UDDDataset(
        data_root=dataset_path,
        split='train',
        transform=create_udd_transforms(is_training=True)
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")  # Should be [3, H, W]
        print(f"Mask shape: {sample['mask'].shape}")
        print(f"Classes: {np.unique(sample['mask'].numpy())}")
        
        # Check class distribution
        class_dist = dataset.get_class_distribution()
        print("Class distribution:", class_dist) 