#!/usr/bin/env python3
"""
Edge-Optimized UAV Landing Dataset
==================================

Combines all available datasets with intelligent 6-class mapping for real-time edge inference.
Designed for limited data (~1,666 images) with extreme augmentation for robust training.

Classes optimized for landing decisions:
0: unknown, 1: safe_flat, 2: safe_soft, 3: obstacle, 4: hazard, 5: boundary
"""

import torch
from torch.utils.data import Dataset, ConcatDataset
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class EdgeLandingDataset(Dataset):
    """
    Edge-optimized dataset combining all available data sources.
    Maps complex semantic classes to 6 essential landing categories.
    """
    
    # 6-class landing-focused mapping
    LANDING_CLASSES = {
        0: "unknown",      # Ambiguous/unlabeled areas
        1: "safe_flat",    # Paved areas, short grass, dirt - ideal landing
        2: "safe_soft",    # Taller grass, vegetation - acceptable landing
        3: "obstacle",     # Trees, buildings, vehicles, people - avoid
        4: "hazard",       # Water, steep slopes, dangerous areas - critical avoid
        5: "boundary"      # Edges, fences, boundaries - spatial reference
    }
    
    def __init__(
        self,
        dataset_configs: List[Dict],
        split: str = "train",
        input_size: int = 256,
        extreme_augmentation: bool = True,
        cache_preprocessing: bool = True
    ):
        """
        Initialize edge landing dataset.
        
        Args:
            dataset_configs: List of dataset configuration dictionaries
            split: Dataset split ('train', 'val', 'test')
            input_size: Target input size for edge inference
            extreme_augmentation: Use aggressive augmentation for limited data
            cache_preprocessing: Cache preprocessed data for speed
        """
        self.dataset_configs = dataset_configs
        self.split = split
        self.input_size = input_size
        self.extreme_augmentation = extreme_augmentation
        self.cache_preprocessing = cache_preprocessing
        
        # Load all available datasets
        self.samples = []
        self._load_all_datasets()
        
        # Create augmentation pipeline
        self.transform = self._create_transforms()
        
        # Cache for preprocessed data
        self.cache = {} if cache_preprocessing else None
        
        print(f"🚁 EdgeLandingDataset initialized:")
        print(f"   Split: {split}")
        print(f"   Total samples: {len(self.samples)}")
        print(f"   Input size: {input_size}x{input_size}")
        print(f"   Extreme augmentation: {extreme_augmentation}")
        
        # Analyze class distribution
        self._analyze_class_distribution()
    
    def _load_all_datasets(self):
        """Load samples from all available datasets."""
        
        for config in self.dataset_configs:
            dataset_type = config['type']
            dataset_path = Path(config['path'])
            
            if not dataset_path.exists():
                print(f"⚠️  Dataset path not found: {dataset_path}")
                continue
            
            if dataset_type == 'semantic_drone':
                self._load_semantic_drone(dataset_path, config)
            elif dataset_type == 'udd6':
                self._load_udd6(dataset_path, config)
            elif dataset_type == 'drone_deploy':
                self._load_drone_deploy(dataset_path, config)
            else:
                print(f"⚠️  Unknown dataset type: {dataset_type}")
    
    def _load_semantic_drone(self, dataset_path: Path, config: Dict):
        """Load Semantic Drone Dataset with 24→6 class mapping."""
        
        images_dir = dataset_path / "original_images"
        labels_dir = dataset_path / "label_images_semantic"
        
        if not (images_dir.exists() and labels_dir.exists()):
            print(f"⚠️  Semantic Drone directories not found in {dataset_path}")
            return
        
        # Original 24-class to 6-class mapping
        semantic_to_landing = {
            0: 0,   # unlabeled → unknown
            1: 1,   # paved-area → safe_flat
            2: 1,   # dirt → safe_flat
            3: 2,   # grass → safe_soft
            4: 1,   # gravel → safe_flat
            5: 4,   # water → hazard
            6: 3,   # rocks → obstacle
            7: 4,   # pool → hazard
            8: 2,   # vegetation → safe_soft
            9: 3,   # roof → obstacle
            10: 5,  # wall → boundary
            11: 5,  # window → boundary
            12: 5,  # door → boundary
            13: 5,  # fence → boundary
            14: 5,  # fence-pole → boundary
            15: 3,  # person → obstacle
            16: 3,  # dog → obstacle
            17: 3,  # car → obstacle
            18: 3,  # bicycle → obstacle
            19: 3,  # tree → obstacle
            20: 3,  # bald-tree → obstacle
            21: 5,  # ar-marker → boundary
            22: 3,  # obstacle → obstacle
            23: 0,  # conflicting → unknown
        }
        
        # RGB to class mapping from the dataset
        rgb_to_semantic = {
            (0, 0, 0): 0,        # unlabeled
            (128, 64, 128): 1,   # paved-area
            (130, 76, 0): 2,     # dirt
            (0, 102, 0): 3,      # grass
            (112, 103, 87): 4,   # gravel
            (28, 42, 168): 5,    # water
            (48, 41, 30): 6,     # rocks
            (0, 50, 89): 7,      # pool
            (107, 142, 35): 8,   # vegetation
            (70, 70, 70): 9,     # roof
            (102, 102, 156): 10, # wall
            (254, 228, 12): 11,  # window
            (254, 148, 12): 12,  # door
            (190, 153, 153): 13, # fence
            (153, 153, 153): 14, # fence-pole
            (255, 22, 96): 15,   # person
            (102, 51, 0): 16,    # dog
            (9, 143, 150): 17,   # car
            (119, 11, 32): 18,   # bicycle
            (51, 51, 0): 19,     # tree
            (190, 250, 190): 20, # bald-tree
            (112, 150, 146): 21, # ar-marker
            (2, 135, 115): 22,   # obstacle
            (255, 0, 0): 23,     # conflicting
        }
        
        # Load image files
        image_files = sorted(list(images_dir.glob("*.jpg")))
        
        # Create train/val/test splits
        total_files = len(image_files)
        train_end = int(0.7 * total_files)
        val_end = int(0.85 * total_files)
        
        if self.split == "train":
            selected_files = image_files[:train_end]
        elif self.split == "val":
            selected_files = image_files[train_end:val_end]
        else:  # test
            selected_files = image_files[val_end:]
        
        for img_file in selected_files:
            label_file = labels_dir / (img_file.stem + ".png")
            if label_file.exists():
                self.samples.append({
                    'image_path': str(img_file),
                    'label_path': str(label_file),
                    'dataset': 'semantic_drone',
                    'rgb_to_semantic': rgb_to_semantic,
                    'semantic_to_landing': semantic_to_landing
                })
        
        print(f"   Loaded {len(selected_files)} Semantic Drone samples")
    
    def _load_udd6(self, dataset_path: Path, config: Dict):
        """Load UDD6 dataset with 6→6 class mapping."""
        
        split_dir = dataset_path / self.split
        images_dir = split_dir / "src"
        labels_dir = split_dir / "gt"
        
        if not (images_dir.exists() and labels_dir.exists()):
            print(f"⚠️  UDD6 directories not found in {dataset_path}")
            return
        
        # UDD6 grayscale values to 6-class mapping
        udd_to_landing = {
            0: 0,    # Other → unknown
            16: 3,   # Facade → obstacle
            90: 1,   # Road → safe_flat
            108: 2,  # Vegetation → safe_soft
            119: 3,  # Roof → obstacle
            255: 0,  # Background → unknown
        }
        
        # Find matching image/label pairs
        image_files = list(images_dir.glob("*.jpg"))
        
        for img_file in image_files:
            # Find corresponding label file
            label_candidates = [
                labels_dir / (img_file.stem + ".png"),
                labels_dir / (img_file.stem + ".jpg"),
            ]
            
            label_file = None
            for candidate in label_candidates:
                if candidate.exists():
                    label_file = candidate
                    break
            
            if label_file:
                self.samples.append({
                    'image_path': str(img_file),
                    'label_path': str(label_file),
                    'dataset': 'udd6',
                    'udd_to_landing': udd_to_landing
                })
        
        print(f"   Loaded {len([s for s in self.samples if s['dataset'] == 'udd6'])} UDD6 samples")
    
    def _load_drone_deploy(self, dataset_path: Path, config: Dict):
        """Load DroneDeploy dataset with custom mapping."""
        
        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"
        
        if not (images_dir.exists() and labels_dir.exists()):
            print(f"⚠️  DroneDeploy directories not found in {dataset_path}")
            return
        
        # DroneDeploy grayscale values to 6-class mapping (based on analysis)
        dronedeploy_to_landing = {
            81: 3,   # Building → obstacle
            91: 1,   # Road → safe_flat
            99: 3,   # Car → obstacle
            105: 0,  # Background → unknown
            132: 2,  # Trees → safe_soft (could be landing in emergency)
            155: 4,  # Pool/Water → hazard
            255: 0,  # Background → unknown
        }
        
        # Load a subset for training (DroneDeploy files are very large)
        image_files = sorted(list(images_dir.glob("*.tif")))[:50]  # Limit to manageable size
        
        for img_file in image_files:
            label_file = labels_dir / img_file.name
            if label_file.exists():
                self.samples.append({
                    'image_path': str(img_file),
                    'label_path': str(label_file),
                    'dataset': 'drone_deploy',
                    'dronedeploy_to_landing': dronedeploy_to_landing
                })
        
        print(f"   Loaded {len([s for s in self.samples if s['dataset'] == 'drone_deploy'])} DroneDeploy samples")
    
    def _create_transforms(self) -> A.Compose:
        """Create augmentation pipeline optimized for edge training."""
        
        transforms = []
        
        if self.split == "train" and self.extreme_augmentation:
            # Extreme augmentation for limited data
            transforms.extend([
                # Geometric transforms (critical for aerial views)
                A.RandomRotate90(p=0.8),
                A.Flip(p=0.7),
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.3, 
                    rotate_limit=30, 
                    p=0.8
                ),
                
                # Photometric transforms
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, 
                    contrast_limit=0.3, 
                    p=0.8
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.6
                ),
                A.RandomGamma(gamma_limit=(70, 130), p=0.5),
                
                # Weather simulation
                A.RandomFog(
                    fog_coef_lower=0.1,
                    fog_coef_upper=0.4,
                    alpha_coef=0.1,
                    p=0.3
                ),
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_lower=1,
                    num_shadows_upper=3,
                    p=0.4
                ),
                
                # Noise and blur (simulate motion/camera artifacts)
                A.GaussNoise(var_limit=(10, 80), p=0.5),
                A.MotionBlur(blur_limit=7, p=0.3),
                A.MedianBlur(blur_limit=5, p=0.2),
                
                # Elastic deformation for robustness
                A.ElasticTransform(
                    alpha=1, 
                    sigma=50, 
                    alpha_affine=30, 
                    p=0.3
                ),
            ])
        elif self.split == "train":
            # Basic augmentation for validation/testing
            transforms.extend([
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ])
        
        # Always apply these
        transforms.extend([
            A.Resize(self.input_size, self.input_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        
        return A.Compose(transforms)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample with caching for speed."""
        
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]
        
        sample_info = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample_info['image_path'])
        if image is None:
            raise ValueError(f"Could not load image: {sample_info['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load and process label based on dataset type
        label = self._load_and_process_label(sample_info)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
        
        # Convert to tensors if not already done
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label).long()
        
        result = {
            'image': image,
            'mask': label,
            'dataset': sample_info['dataset'],
            'image_path': sample_info['image_path']
        }
        
        # Cache if enabled
        if self.cache is not None:
            self.cache[idx] = result
        
        return result
    
    def _load_and_process_label(self, sample_info: Dict) -> np.ndarray:
        """Load and process label based on dataset type."""
        
        dataset_type = sample_info['dataset']
        
        if dataset_type == 'semantic_drone':
            return self._process_semantic_drone_label(sample_info)
        elif dataset_type == 'udd6':
            return self._process_udd6_label(sample_info)
        elif dataset_type == 'drone_deploy':
            return self._process_drone_deploy_label(sample_info)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def _process_semantic_drone_label(self, sample_info: Dict) -> np.ndarray:
        """Process Semantic Drone Dataset RGB labels."""
        
        # Load RGB label
        label_rgb = cv2.imread(sample_info['label_path'])
        if label_rgb is None:
            raise ValueError(f"Could not load label: {sample_info['label_path']}")
        label_rgb = cv2.cvtColor(label_rgb, cv2.COLOR_BGR2RGB)
        
        # Convert RGB to semantic classes
        h, w = label_rgb.shape[:2]
        semantic_label = np.zeros((h, w), dtype=np.uint8)
        
        rgb_to_semantic = sample_info['rgb_to_semantic']
        for rgb_color, semantic_class in rgb_to_semantic.items():
            mask = np.all(label_rgb == np.array(rgb_color), axis=2)
            semantic_label[mask] = semantic_class
        
        # Convert semantic to landing classes
        semantic_to_landing = sample_info['semantic_to_landing']
        landing_label = np.zeros((h, w), dtype=np.uint8)
        for semantic_class, landing_class in semantic_to_landing.items():
            mask = (semantic_label == semantic_class)
            landing_label[mask] = landing_class
        
        return landing_label
    
    def _process_udd6_label(self, sample_info: Dict) -> np.ndarray:
        """Process UDD6 grayscale labels."""
        
        # Load grayscale label
        label = cv2.imread(sample_info['label_path'], cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise ValueError(f"Could not load label: {sample_info['label_path']}")
        
        # Convert UDD classes to landing classes
        udd_to_landing = sample_info['udd_to_landing']
        landing_label = np.zeros_like(label, dtype=np.uint8)
        
        for udd_class, landing_class in udd_to_landing.items():
            mask = (label == udd_class)
            landing_label[mask] = landing_class
        
        return landing_label
    
    def _process_drone_deploy_label(self, sample_info: Dict) -> np.ndarray:
        """Process DroneDeploy grayscale labels."""
        
        # Load grayscale label
        label = cv2.imread(sample_info['label_path'], cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise ValueError(f"Could not load label: {sample_info['label_path']}")
        
        # Convert DroneDeploy classes to landing classes
        dronedeploy_to_landing = sample_info['dronedeploy_to_landing']
        landing_label = np.zeros_like(label, dtype=np.uint8)
        
        for dd_class, landing_class in dronedeploy_to_landing.items():
            mask = (label == dd_class)
            landing_label[mask] = landing_class
        
        return landing_label
    
    def _analyze_class_distribution(self):
        """Analyze class distribution across all datasets."""
        
        if len(self.samples) == 0:
            print("⚠️  No samples loaded for class distribution analysis")
            return
        
        print(f"\n📊 Analyzing class distribution ({len(self.samples)} samples)...")
        
        # Sample a subset for analysis (faster)
        sample_indices = np.random.choice(
            len(self.samples), 
            min(50, len(self.samples)), 
            replace=False
        )
        
        class_counts = Counter()
        total_pixels = 0
        
        for idx in sample_indices:
            try:
                sample_info = self.samples[idx]
                label = self._load_and_process_label(sample_info)
                
                unique, counts = np.unique(label, return_counts=True)
                for class_id, count in zip(unique, counts):
                    class_counts[class_id] += count
                    total_pixels += count
                    
            except Exception as e:
                print(f"⚠️  Error processing sample {idx}: {e}")
        
        # Print distribution
        print(f"   Class Distribution (sampled):")
        for class_id in range(6):
            class_name = self.LANDING_CLASSES[class_id]
            count = class_counts.get(class_id, 0)
            percentage = (count / total_pixels * 100) if total_pixels > 0 else 0
            print(f"   {class_id} ({class_name:12}): {percentage:5.1f}% ({count:,} pixels)")
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for balanced training."""
        
        # Sample distribution analysis
        sample_indices = np.random.choice(
            len(self.samples), 
            min(100, len(self.samples)), 
            replace=False
        )
        
        class_counts = Counter()
        
        for idx in sample_indices:
            try:
                sample_info = self.samples[idx]
                label = self._load_and_process_label(sample_info)
                unique, counts = np.unique(label, return_counts=True)
                for class_id, count in zip(unique, counts):
                    class_counts[class_id] += count
            except:
                continue
        
        # Compute inverse frequency weights
        total_pixels = sum(class_counts.values())
        class_weights = torch.ones(6)
        
        for class_id in range(6):
            count = class_counts.get(class_id, 1)  # Avoid division by zero
            class_weights[class_id] = total_pixels / (6 * count)
        
        # Apply safety multipliers
        class_weights[4] *= 2.0  # Emphasize hazard detection
        class_weights[3] *= 1.5  # Emphasize obstacle detection
        
        # Normalize and clip
        class_weights = torch.clamp(class_weights, min=0.1, max=50.0)
        
        return class_weights


def create_edge_datasets(
    data_root: str = "datasets",
    input_size: int = 256,
    extreme_augmentation: bool = True
) -> Dict[str, EdgeLandingDataset]:
    """
    Create edge-optimized datasets from all available data sources.
    
    Args:
        data_root: Root directory containing datasets
        input_size: Target input size for edge inference
        extreme_augmentation: Use aggressive augmentation for limited data
        
    Returns:
        Dictionary with train/val/test datasets
    """
    
    # Configure available datasets
    dataset_configs = [
        {
            'type': 'semantic_drone',
            'path': f"{data_root}/Aerial_Semantic_Segmentation_Drone_Dataset/dataset/semantic_drone_dataset"
        },
        {
            'type': 'udd6',
            'path': f"{data_root}/UDD/UDD/UDD6"
        },
        {
            'type': 'drone_deploy',
            'path': f"{data_root}/drone_deploy_dataset_intermediate/dataset-medium"
        }
    ]
    
    # Create datasets for each split
    datasets = {}
    for split in ['train', 'val', 'test']:
        datasets[split] = EdgeLandingDataset(
            dataset_configs=dataset_configs,
            split=split,
            input_size=input_size,
            extreme_augmentation=extreme_augmentation and split == 'train'
        )
    
    return datasets


if __name__ == "__main__":
    # Test dataset creation
    print("🚁 Testing Edge Landing Dataset...")
    
    try:
        datasets = create_edge_datasets(
            data_root="../datasets",
            input_size=256,
            extreme_augmentation=True
        )
        
        print(f"\n✅ Dataset creation successful!")
        for split, dataset in datasets.items():
            print(f"   {split}: {len(dataset)} samples")
        
        # Test sample loading
        if len(datasets['train']) > 0:
            sample = datasets['train'][0]
            print(f"\n📋 Sample test:")
            print(f"   Image shape: {sample['image'].shape}")
            print(f"   Mask shape: {sample['mask'].shape}")
            print(f"   Unique classes: {torch.unique(sample['mask'])}")
            print(f"   Dataset: {sample['dataset']}")
        
        # Get class weights
        class_weights = datasets['train'].get_class_weights()
        print(f"\n⚖️  Class weights: {class_weights}")
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc() 