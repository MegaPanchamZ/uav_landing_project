#!/usr/bin/env python3
"""
Dataset Preparation for Three-Step Fine-Tuning Pipeline

This script helps prepare the datasets needed for the three-step fine-tuning:
1. DroneDeploy dataset for general aerial view understanding
2. UDD-6 dataset for low-altitude drone view specialization

Handles dataset downloading, organization, and format standardization.
"""

import os
import shutil
import requests
import zipfile
import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetPreparer:
    """Prepare datasets for the three-step fine-tuning pipeline."""
    
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(exist_ok=True)
        
        # Dataset URLs and configurations
        self.dataset_configs = {
            "dronedeploy": {
                "name": "DroneDeploy Segmentation Dataset",
                "url": "https://github.com/dronedeploy/dd-ml-segmentation-benchmark/releases/download/v1.0/dataset.zip",
                "target_classes": {
                    0: "background",
                    1: "building", 
                    2: "clutter",
                    3: "vegetation",
                    4: "water", 
                    5: "ground",
                    6: "car"
                },
                "landing_mapping": {
                    0: 0,  # background -> background
                    1: 3,  # building -> obstacles
                    2: 3,  # clutter -> obstacles
                    3: 2,  # vegetation -> marginal (if flat)
                    4: 4,  # water -> unsafe
                    5: 1,  # ground -> suitable
                    6: 4   # car -> unsafe
                }
            },
            "udd6": {
                "name": "Urban Drone Dataset (UDD-6)",
                "url": "https://github.com/MarcWong/UDD/releases/download/1.0/udd6.zip",
                "target_classes": {
                    0: "other",
                    1: "facade", 
                    2: "road",
                    3: "vegetation",
                    4: "vehicle",
                    5: "roof"
                },
                "landing_mapping": {
                    0: 0,  # other -> background
                    1: 3,  # facade -> obstacles
                    2: 1,  # road -> suitable
                    3: 2,  # vegetation -> marginal
                    4: 4,  # vehicle -> unsafe
                    5: 3   # roof -> obstacles
                }
            }
        }
        
        # Our target 6-class system
        self.target_classes = {
            0: "background",
            1: "suitable",     # Flat, clear areas good for landing
            2: "marginal",     # Areas that might be suitable (vegetation)
            3: "obstacles",    # Buildings, structures to avoid
            4: "unsafe",       # Water, vehicles, dangerous areas
            5: "unknown"       # Uncertain/unclassified areas
        }
        
    def download_dataset(self, dataset_name: str, force_download: bool = False) -> bool:
        """Download and extract a dataset."""
        
        if dataset_name not in self.dataset_configs:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
            
        config = self.dataset_configs[dataset_name]
        dataset_dir = self.data_root / dataset_name
        
        # Check if already downloaded
        if dataset_dir.exists() and not force_download:
            logger.info(f"Dataset {dataset_name} already exists at {dataset_dir}")
            return True
            
        logger.info(f"Downloading {config['name']}...")
        
        # Create temporary download directory
        download_dir = self.data_root / "downloads"
        download_dir.mkdir(exist_ok=True)
        
        zip_path = download_dir / f"{dataset_name}.zip"
        
        try:
            # Download dataset
            response = requests.get(config['url'], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {dataset_name}") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                            
            # Extract dataset
            logger.info(f"Extracting {dataset_name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
                
            # Clean up download
            zip_path.unlink()
            
            logger.info(f"✅ Successfully downloaded and extracted {dataset_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download {dataset_name}: {e}")
            return False
            
    def prepare_dronedeploy_dataset(self) -> bool:
        """Prepare DroneDeploy dataset for training."""
        
        logger.info("📋 Preparing DroneDeploy dataset...")
        
        dataset_dir = self.data_root / "dronedeploy"
        if not dataset_dir.exists():
            logger.error("DroneDeploy dataset not found. Please download first.")
            return False
            
        # Expected DroneDeploy structure after extraction
        original_structure = dataset_dir / "dataset"
        if not original_structure.exists():
            logger.error("Unexpected DroneDeploy dataset structure")
            return False
            
        # Reorganize to our standard structure
        target_structure = {
            "train": {"images": [], "annotations": []},
            "val": {"images": [], "annotations": []},
            "test": {"images": [], "annotations": []}
        }
        
        # Process each split
        for split in ["train", "val", "test"]:
            split_dir = original_structure / split
            if not split_dir.exists():
                continue
                
            # Create target directories
            (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (dataset_dir / "annotations" / split).mkdir(parents=True, exist_ok=True)
            
            # Process images and annotations
            img_dir = split_dir / "images"
            ann_dir = split_dir / "labels"
            
            if img_dir.exists() and ann_dir.exists():
                for img_path in img_dir.glob("*.jpg"):
                    # Copy image
                    target_img = dataset_dir / "images" / split / img_path.name
                    if not target_img.exists():
                        shutil.copy2(img_path, target_img)
                    
                    # Process annotation
                    ann_path = ann_dir / f"{img_path.stem}.png"
                    if ann_path.exists():
                        target_ann = dataset_dir / "annotations" / split / f"{img_path.stem}.png"
                        if not target_ann.exists():
                            self._convert_annotation(
                                str(ann_path), 
                                str(target_ann),
                                self.dataset_configs["dronedeploy"]["landing_mapping"]
                            )
                            
        logger.info("✅ DroneDeploy dataset preparation completed")
        return True
        
    def prepare_udd6_dataset(self) -> bool:
        """Prepare UDD-6 dataset for training."""
        
        logger.info("📋 Preparing UDD-6 dataset...")
        
        dataset_dir = self.data_root / "udd6"
        if not dataset_dir.exists():
            logger.error("UDD-6 dataset not found. Please download first.")
            return False
            
        # UDD-6 has a different structure - find the actual data
        possible_paths = [
            dataset_dir / "udd6",
            dataset_dir / "UDD",
            dataset_dir
        ]
        
        actual_path = None
        for path in possible_paths:
            if (path / "train").exists() or (path / "val").exists():
                actual_path = path
                break
                
        if actual_path is None:
            logger.error("Could not find UDD-6 data structure")
            return False
            
        # Process each split
        for split in ["train", "val"]:
            split_dir = actual_path / split
            if not split_dir.exists():
                continue
                
            # Create target directories  
            (dataset_dir / split / "src").mkdir(parents=True, exist_ok=True)
            (dataset_dir / split / "gt").mkdir(parents=True, exist_ok=True)
            
            # Find source and ground truth directories
            src_dir = split_dir / "src" if (split_dir / "src").exists() else split_dir / "images"
            gt_dir = split_dir / "gt" if (split_dir / "gt").exists() else split_dir / "labels"
            
            if src_dir.exists() and gt_dir.exists():
                for img_path in src_dir.glob("*.jpg"):
                    # Copy image
                    target_img = dataset_dir / split / "src" / img_path.name
                    if not target_img.exists():
                        shutil.copy2(img_path, target_img)
                    
                    # Process annotation
                    ann_path = gt_dir / f"{img_path.stem}.png"
                    if ann_path.exists():
                        target_ann = dataset_dir / split / "gt" / f"{img_path.stem}.png"
                        if not target_ann.exists():
                            self._convert_annotation(
                                str(ann_path),
                                str(target_ann), 
                                self.dataset_configs["udd6"]["landing_mapping"]
                            )
                            
        logger.info("✅ UDD-6 dataset preparation completed")
        return True
        
    def _convert_annotation(self, input_path: str, output_path: str, class_mapping: Dict[int, int]):
        """Convert annotation to our 6-class system."""
        
        # Load original annotation
        annotation = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if annotation is None:
            logger.warning(f"Could not load annotation: {input_path}")
            return
            
        # Apply class mapping
        converted = np.zeros_like(annotation)
        for orig_class, target_class in class_mapping.items():
            converted[annotation == orig_class] = target_class
            
        # Save converted annotation
        cv2.imwrite(output_path, converted)
        
    def create_synthetic_dataset(self, output_dir: str = "data/synthetic", num_samples: int = 1000):
        """Create a synthetic dataset for testing the pipeline."""
        
        logger.info(f"🎨 Creating synthetic dataset with {num_samples} samples...")
        
        output_path = Path(output_dir)
        
        # Create directory structure
        for split in ["train", "val"]:
            (output_path / split / "images").mkdir(parents=True, exist_ok=True)
            (output_path / split / "annotations").mkdir(parents=True, exist_ok=True)
            
        # Generate samples
        train_samples = int(num_samples * 0.8)
        val_samples = num_samples - train_samples
        
        for split, n_samples in [("train", train_samples), ("val", val_samples)]:
            for i in tqdm(range(n_samples), desc=f"Generating {split} samples"):
                # Create synthetic aerial image
                image = self._generate_synthetic_aerial_image()
                
                # Create corresponding segmentation mask
                mask = self._generate_synthetic_mask(image)
                
                # Save files
                img_path = output_path / split / "images" / f"{split}_{i:06d}.jpg"
                ann_path = output_path / split / "annotations" / f"{split}_{i:06d}.png"
                
                cv2.imwrite(str(img_path), image)
                cv2.imwrite(str(ann_path), mask)
                
        logger.info(f"✅ Synthetic dataset created at {output_path}")
        return str(output_path)
        
    def _generate_synthetic_aerial_image(self, size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """Generate a synthetic aerial image."""
        
        height, width = size
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Generate ground base color (grass/dirt)
        base_color = np.random.randint(60, 120, 3)  # Greenish-brown
        image[:] = base_color
        
        # Add some texture/noise
        noise = np.random.randint(-20, 20, (height, width, 3))
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add random rectangular "buildings"
        num_buildings = np.random.randint(2, 8)
        for _ in range(num_buildings):
            x1 = np.random.randint(0, width//2)
            y1 = np.random.randint(0, height//2)
            w = np.random.randint(30, 100)
            h = np.random.randint(30, 100)
            x2 = min(x1 + w, width)
            y2 = min(y1 + h, height)
            
            building_color = np.random.randint(80, 180, 3)
            cv2.rectangle(image, (x1, y1), (x2, y2), building_color.tolist(), -1)
            
        # Add some circular "trees" 
        num_trees = np.random.randint(5, 15)
        for _ in range(num_trees):
            center = (np.random.randint(0, width), np.random.randint(0, height))
            radius = np.random.randint(10, 30)
            tree_color = np.random.randint(40, 100, 3)
            tree_color[1] = np.random.randint(80, 150)  # More green
            cv2.circle(image, center, radius, tree_color.tolist(), -1)
            
        # Add potential landing zones (clear rectangular areas)
        num_landing_zones = np.random.randint(1, 3)
        for _ in range(num_landing_zones):
            x1 = np.random.randint(0, width//2)
            y1 = np.random.randint(0, height//2)
            w = np.random.randint(40, 80)
            h = np.random.randint(40, 80)
            x2 = min(x1 + w, width)
            y2 = min(y1 + h, height)
            
            # Clear, flat color for landing zone
            landing_color = np.random.randint(100, 140, 3)
            cv2.rectangle(image, (x1, y1), (x2, y2), landing_color.tolist(), -1)
            
        return image
        
    def _generate_synthetic_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate segmentation mask for synthetic image."""
        
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Simple color-based segmentation for synthetic data
        # This is a simplified approach - real annotations would be more complex
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect different regions based on color
        # Green areas (vegetation) -> marginal (2)
        green_mask = cv2.inRange(hsv, (35, 30, 30), (85, 255, 255))
        mask[green_mask > 0] = 2
        
        # Darker areas (buildings/obstacles) -> obstacles (3) 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark_mask = gray > 120
        building_mask = cv2.inRange(hsv, (0, 0, 80), (180, 50, 180))
        mask[building_mask > 0] = 3
        
        # Lighter, uniform areas (potential landing zones) -> suitable (1)
        light_mask = gray > 110
        uniform_mask = cv2.inRange(hsv, (0, 0, 100), (180, 30, 160))
        suitable_mask = light_mask & (uniform_mask > 0)
        mask[suitable_mask] = 1
        
        # Everything else remains background (0)
        
        return mask
        
    def validate_dataset(self, dataset_path: str, dataset_type: str = "dronedeploy") -> bool:
        """Validate dataset structure and content."""
        
        logger.info(f"🔍 Validating {dataset_type} dataset at {dataset_path}")
        
        dataset_dir = Path(dataset_path)
        
        if not dataset_dir.exists():
            logger.error(f"Dataset directory does not exist: {dataset_path}")
            return False
            
        # Check structure based on dataset type
        if dataset_type == "dronedeploy":
            required_dirs = [
                "images/train", "images/val", 
                "annotations/train", "annotations/val"
            ]
        elif dataset_type == "udd6":
            required_dirs = [
                "train/src", "train/gt",
                "val/src", "val/gt"
            ]
        else:
            logger.error(f"Unknown dataset type: {dataset_type}")
            return False
            
        # Check directories exist
        for dir_path in required_dirs:
            full_path = dataset_dir / dir_path
            if not full_path.exists():
                logger.error(f"Missing directory: {full_path}")
                return False
                
        # Check that we have matching images and annotations
        validation_errors = []
        
        if dataset_type == "dronedeploy":
            for split in ["train", "val"]:
                img_dir = dataset_dir / "images" / split
                ann_dir = dataset_dir / "annotations" / split
                
                img_files = set(f.stem for f in img_dir.glob("*.jpg"))
                ann_files = set(f.stem for f in ann_dir.glob("*.png"))
                
                if img_files != ann_files:
                    missing_in_ann = img_files - ann_files
                    missing_in_img = ann_files - img_files
                    
                    if missing_in_ann:
                        validation_errors.append(f"Missing annotations in {split}: {missing_in_ann}")
                    if missing_in_img:
                        validation_errors.append(f"Missing images in {split}: {missing_in_img}")
                        
        elif dataset_type == "udd6":
            for split in ["train", "val"]:
                img_dir = dataset_dir / split / "src"
                ann_dir = dataset_dir / split / "gt"
                
                img_files = set(f.stem for f in img_dir.glob("*.jpg"))
                ann_files = set(f.stem for f in ann_dir.glob("*.png"))
                
                if img_files != ann_files:
                    missing_in_ann = img_files - ann_files
                    missing_in_img = ann_files - img_files
                    
                    if missing_in_ann:
                        validation_errors.append(f"Missing annotations in {split}: {missing_in_ann}")
                    if missing_in_img:
                        validation_errors.append(f"Missing images in {split}: {missing_in_img}")
                        
        if validation_errors:
            for error in validation_errors:
                logger.error(error)
            return False
            
        logger.info("✅ Dataset validation passed")
        return True
        
    def get_dataset_stats(self, dataset_path: str, dataset_type: str = "dronedeploy") -> Dict:
        """Get statistics about a dataset."""
        
        logger.info(f"📊 Analyzing dataset statistics...")
        
        dataset_dir = Path(dataset_path)
        stats = {
            "dataset_type": dataset_type,
            "splits": {},
            "class_distribution": {i: 0 for i in range(6)},
            "total_pixels": 0
        }
        
        if dataset_type == "dronedeploy":
            splits = ["train", "val", "test"]
            for split in splits:
                ann_dir = dataset_dir / "annotations" / split
                if ann_dir.exists():
                    split_stats = {"num_images": 0, "class_counts": {i: 0 for i in range(6)}}
                    
                    for ann_path in ann_dir.glob("*.png"):
                        split_stats["num_images"] += 1
                        
                        # Load and analyze annotation
                        annotation = cv2.imread(str(ann_path), cv2.IMREAD_GRAYSCALE)
                        if annotation is not None:
                            unique, counts = np.unique(annotation, return_counts=True)
                            for class_id, count in zip(unique, counts):
                                if 0 <= class_id < 6:
                                    split_stats["class_counts"][int(class_id)] += int(count)
                                    stats["class_distribution"][int(class_id)] += int(count)
                                    stats["total_pixels"] += int(count)
                                    
                    stats["splits"][split] = split_stats
                    
        elif dataset_type == "udd6":
            splits = ["train", "val"]
            for split in splits:
                ann_dir = dataset_dir / split / "gt"
                if ann_dir.exists():
                    split_stats = {"num_images": 0, "class_counts": {i: 0 for i in range(6)}}
                    
                    for ann_path in ann_dir.glob("*.png"):
                        split_stats["num_images"] += 1
                        
                        annotation = cv2.imread(str(ann_path), cv2.IMREAD_GRAYSCALE)
                        if annotation is not None:
                            unique, counts = np.unique(annotation, return_counts=True)
                            for class_id, count in zip(unique, counts):
                                if 0 <= class_id < 6:
                                    split_stats["class_counts"][int(class_id)] += int(count)
                                    stats["class_distribution"][int(class_id)] += int(count)
                                    stats["total_pixels"] += int(count)
                                    
                    stats["splits"][split] = split_stats
                    
        return stats

def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for UAV landing detection")
    parser.add_argument("--data_root", type=str, default="data", 
                       help="Root directory for datasets")
    parser.add_argument("--download", nargs="+", choices=["dronedeploy", "udd6"], 
                       help="Download specific datasets")
    parser.add_argument("--prepare", nargs="+", choices=["dronedeploy", "udd6"],
                       help="Prepare specific datasets")
    parser.add_argument("--validate", nargs="+", 
                       help="Validate datasets (format: dataset_name:dataset_type)")
    parser.add_argument("--stats", nargs="+",
                       help="Get dataset statistics (format: dataset_path:dataset_type)")
    parser.add_argument("--synthetic", action="store_true",
                       help="Create synthetic dataset for testing")
    parser.add_argument("--synthetic_samples", type=int, default=1000,
                       help="Number of synthetic samples to generate")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download/preparation")
    
    args = parser.parse_args()
    
    preparer = DatasetPreparer(data_root=args.data_root)
    
    # Download datasets
    if args.download:
        for dataset in args.download:
            success = preparer.download_dataset(dataset, force_download=args.force)
            if success:
                print(f"✅ Downloaded {dataset}")
            else:
                print(f"❌ Failed to download {dataset}")
                
    # Prepare datasets
    if args.prepare:
        for dataset in args.prepare:
            if dataset == "dronedeploy":
                success = preparer.prepare_dronedeploy_dataset()
            elif dataset == "udd6":
                success = preparer.prepare_udd6_dataset()
            else:
                logger.error(f"Unknown dataset: {dataset}")
                continue
                
            if success:
                print(f"✅ Prepared {dataset}")
            else:
                print(f"❌ Failed to prepare {dataset}")
                
    # Validate datasets
    if args.validate:
        for item in args.validate:
            if ":" in item:
                dataset_path, dataset_type = item.split(":", 1)
            else:
                dataset_path, dataset_type = item, "dronedeploy"
                
            success = preparer.validate_dataset(dataset_path, dataset_type)
            if success:
                print(f"✅ Validated {dataset_path}")
            else:
                print(f"❌ Validation failed for {dataset_path}")
                
    # Get dataset statistics
    if args.stats:
        for item in args.stats:
            if ":" in item:
                dataset_path, dataset_type = item.split(":", 1) 
            else:
                dataset_path, dataset_type = item, "dronedeploy"
                
            stats = preparer.get_dataset_stats(dataset_path, dataset_type)
            print(f"\n📊 Statistics for {dataset_path}:")
            print(f"   Dataset type: {stats['dataset_type']}")
            print(f"   Total pixels: {stats['total_pixels']:,}")
            print(f"   Splits: {list(stats['splits'].keys())}")
            
            for split, split_stats in stats['splits'].items():
                print(f"   {split}: {split_stats['num_images']} images")
                
            print(f"   Class distribution:")
            for class_id, count in stats['class_distribution'].items():
                class_name = preparer.target_classes[class_id]
                percentage = count / stats['total_pixels'] * 100 if stats['total_pixels'] > 0 else 0
                print(f"     {class_id} ({class_name}): {count:,} ({percentage:.1f}%)")
                
    # Create synthetic dataset
    if args.synthetic:
        synthetic_path = preparer.create_synthetic_dataset(
            output_dir=str(preparer.data_root / "synthetic"),
            num_samples=args.synthetic_samples
        )
        print(f"✅ Created synthetic dataset at {synthetic_path}")

if __name__ == "__main__":
    main()
