#!/usr/bin/env python3
"""
DroneDeploy Dataset with Height Maps
===================================

Dataset loader for DroneDeploy medium dataset with elevation (height) maps.
Supports 4-channel input (RGB + Height) for enhanced landing zone detection.

Classes:
- 0: Background  
- 1: Building (Danger)
- 2: Road (Safe)
- 3: Trees (Caution) 
- 4: Car (Danger)
- 5: Pool (Danger - water)
- 6: Other (Background)

Landing Mapping:
- 0: Background
- 1: Safe (Road)
- 2: Caution (Trees) 
- 3: Danger (Building, Car, Pool, Other)
"""

import os
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import albumentations as A


class DroneDeployDataset(Dataset):
    """DroneDeploy dataset with height maps for UAV landing detection."""
    
    # DroneDeploy actual class values to 4 landing classes mapping
    # Based on inspection: [81, 91, 99, 105, 132, 155, 255]
    CLASS_MAPPING = {
        81: 3,   # Building → Danger
        91: 1,   # Road → Safe  
        99: 3,   # Car → Danger
        105: 0,  # Background/Other → Background
        132: 2,  # Trees → Caution
        155: 3,  # Pool/Water → Danger
        255: 0,  # Background/Other → Background
    }
    
    LANDING_CLASSES = ['Background', 'Safe', 'Caution', 'Danger']
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[A.Compose] = None,
        use_height: bool = True,
        target_resolution: Tuple[int, int] = (512, 512),
        return_confidence: bool = False
    ):
        """
        Initialize DroneDeploy dataset.
        
        Args:
            data_root: Path to dataset-medium directory
            split: 'train', 'val', or 'test'
            transform: Albumentations transforms
            use_height: Whether to include height maps (4-channel input)
            target_resolution: Target image size
            return_confidence: Whether to return confidence maps
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.use_height = use_height
        self.target_resolution = target_resolution
        self.return_confidence = return_confidence
        
        # Paths
        self.images_dir = self.data_root / "images"
        self.labels_dir = self.data_root / "labels" 
        self.elevations_dir = self.data_root / "elevations"
        self.index_file = self.data_root / "index.csv"
        
        # Validate paths
        if not all(p.exists() for p in [self.images_dir, self.labels_dir, self.index_file]):
            raise ValueError(f"Dataset directories not found in {data_root}")
            
        # Load dataset index
        self.samples = self._load_index()
        
        # Create train/val/test splits
        self.samples = self._create_splits()
        
        print(f"📊 DroneDeployDataset initialized:")
        print(f"   Split: {split} ({len(self.samples)} samples)")
        print(f"   Classes: 4 landing classes")
        print(f"   Resolution: {target_resolution}")
        print(f"   Height Maps: {'✅' if use_height else '❌'}")
        
    def _load_index(self) -> list:
        """Load dataset index from CSV."""
        samples = []
        
        with open(self.index_file, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                if len(row) >= 2:
                    quality, image_id = row[0], row[1]
                    if quality == 'GOOD':  # Only use high-quality samples
                        samples.append({
                            'image_id': image_id,
                            'image_path': self.images_dir / f"{image_id}-ortho.tif",
                            'label_path': self.labels_dir / f"{image_id}-label.png",
                            'elevation_path': self.elevations_dir / f"{image_id}-elevation.tif"
                        })
        
        return samples
    
    def _create_splits(self) -> list:
        """Create train/val/test splits."""
        total = len(self.samples)
        
        if self.split == 'train':
            return self.samples[:int(0.7 * total)]  # 70% train
        elif self.split == 'val':
            return self.samples[int(0.7 * total):int(0.85 * total)]  # 15% val
        else:  # test
            return self.samples[int(0.85 * total):]  # 15% test
    
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
        
        # Load height map if requested
        if self.use_height:
            elevation_path = sample_info['elevation_path']
            if elevation_path.exists():
                height_map = cv2.imread(str(elevation_path), cv2.IMREAD_GRAYSCALE)
                if height_map is not None:
                    # Normalize height map to 0-255 range
                    height_map = cv2.normalize(height_map, None, 0, 255, cv2.NORM_MINMAX)
                    # Add height as 4th channel
                    image = np.dstack([image, height_map])
                else:
                    # Fallback: create dummy height channel
                    height_map = np.zeros(image.shape[:2], dtype=np.uint8)
                    image = np.dstack([image, height_map])
            else:
                # Fallback: create dummy height channel  
                height_map = np.zeros(image.shape[:2], dtype=np.uint8)
                image = np.dstack([image, height_map])
        
        # Map classes to landing classes
        mapped_label = self._map_classes(label)
        
        # Apply transforms
        if self.transform:
            if self.use_height:
                # Special handling for 4-channel images in albumentations
                rgb_channels = image[:, :, :3]
                height_channel = image[:, :, 3]
                
                transformed = self.transform(image=rgb_channels, mask=mapped_label)
                image = transformed['image']
                mapped_label = transformed['mask']
                
                # Resize height channel separately and add back
                height_resized = cv2.resize(height_channel, self.target_resolution, interpolation=cv2.INTER_LINEAR)
                if len(image.shape) == 3:  # If tensor format
                    height_resized = torch.from_numpy(height_resized).unsqueeze(0).float() / 255.0
                    image = torch.cat([image, height_resized], dim=0)
                else:  # If numpy format
                    image = np.dstack([image, height_resized])
            else:
                transformed = self.transform(image=image, mask=mapped_label)
                image, mapped_label = transformed['image'], transformed['mask']
        
        # Convert to tensors if not already done by transforms
        if not isinstance(image, torch.Tensor):
            if self.use_height:
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            else:
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
        """Map original 7 classes to 4 landing classes."""
        mapped_label = np.zeros_like(label, dtype=np.uint8)
        
        for original_class, landing_class in self.CLASS_MAPPING.items():
            mask = (label == original_class)
            mapped_label[mask] = landing_class
        
        return mapped_label
    
    def _compute_confidence_map(self, original_label: np.ndarray, mapped_label: np.ndarray) -> np.ndarray:
        """Compute confidence map based on class certainty."""
        confidence = np.ones_like(mapped_label, dtype=np.float32)
        
        # Lower confidence for ambiguous classes
        trees_mask = (original_label == 3)  # Trees can vary in safety
        confidence[trees_mask] = 0.7
        
        # High confidence for clear danger/safe classes
        road_mask = (original_label == 2)  # Roads are clearly safe
        building_mask = (original_label == 1)  # Buildings clearly dangerous
        confidence[road_mask] = 1.0
        confidence[building_mask] = 1.0
        
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


def create_drone_deploy_transforms(
    input_resolution: Tuple[int, int] = (512, 512),
    is_training: bool = True
) -> A.Compose:
    """Create transforms for DroneDeploy dataset."""
    
    if is_training:
        transforms = [
            A.Resize(height=input_resolution[0], width=input_resolution[1]),
            A.RandomCrop(height=input_resolution[0], width=input_resolution[1], p=0.3),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
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
    dataset_path = "../datasets/drone_deploy_dataset_intermediate/dataset-medium"
    
    # Test with height maps
    dataset = DroneDeployDataset(
        data_root=dataset_path,
        split='train',
        use_height=True,
        transform=create_drone_deploy_transforms(is_training=True)
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")  # Should be [4, H, W] with height
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Classes: {np.unique(sample['mask'].numpy())}")
    
    # Check class distribution
    class_dist = dataset.get_class_distribution()
    print("Class distribution:", class_dist) 