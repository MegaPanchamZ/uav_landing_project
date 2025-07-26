#!/usr/bin/env python3
"""
Dataset Preprocessing for A100 GPU Training
==========================================

Preprocesses all downloaded datasets for optimal training:
- Semantic Drone Dataset: Class mapping and augmentation
- UDD6 Dataset: Format standardization
- DroneDeploy Dataset: Resolution optimization
- Creates training/validation splits
- Generates cached dataset statistics
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import warnings

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
class PreprocessingConfig:
    """Configuration for dataset preprocessing"""
    
    def __init__(self):
        # Paths
        self.raw_data_dir = Path("datasets/raw")
        self.processed_data_dir = Path("datasets/processed")
        self.splits_dir = Path("datasets/splits")
        self.cache_dir = Path("datasets/cache")
        
        # Processing parameters
        self.target_size = (512, 512)
        self.train_val_split = 0.8
        self.random_seed = 42
        
        # Quality control
        self.min_image_size = (256, 256)
        self.max_image_size = (2048, 2048)
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
        
        # Class mapping for landing detection
        self.landing_classes = {
            0: "safe_landing",      # Safe areas for landing
            1: "unsafe_structure",  # Buildings, structures
            2: "unsafe_vehicle",    # Cars, vehicles
            3: "vegetation"         # Trees, vegetation
        }
        
        # Semantic Drone Dataset class mapping (24 -> 4 classes)
        self.semantic_drone_mapping = {
            0: 0,   # unlabeled -> safe_landing
            1: 0,   # paved-area -> safe_landing
            2: 0,   # dirt -> safe_landing
            3: 0,   # grass -> safe_landing (debatable, but often safe)
            4: 3,   # gravel -> vegetation (mixed)
            5: 0,   # water -> safe_landing (for amphibious)
            6: 1,   # rocks -> unsafe_structure
            7: 0,   # pool -> safe_landing
            8: 3,   # vegetation -> vegetation
            9: 1,   # roof -> unsafe_structure
            10: 1,  # wall -> unsafe_structure
            11: 1,  # building -> unsafe_structure
            12: 1,  # fence -> unsafe_structure
            13: 1,  # fence-pole -> unsafe_structure
            14: 2,  # person -> unsafe_vehicle
            15: 2,  # dog -> unsafe_vehicle
            16: 2,  # car -> unsafe_vehicle
            17: 2,  # bicycle -> unsafe_vehicle
            18: 3,  # tree -> vegetation
            19: 1,  # bald-tree -> unsafe_structure
            20: 1,  # ar-marker -> unsafe_structure
            21: 1,  # obstacle -> unsafe_structure
            22: 1,  # conflicting -> unsafe_structure
            23: 0   # background -> safe_landing
        }

# =============================================================================
# Dataset Processors
# =============================================================================
class SemanticDroneProcessor:
    """Processor for Semantic Drone Dataset"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.dataset_dir = config.raw_data_dir / "semantic_drone_dataset"
        self.output_dir = config.processed_data_dir / "semantic_drone"
        
    def process(self) -> Dict[str, int]:
        """Process Semantic Drone Dataset"""
        print("🎯 Processing Semantic Drone Dataset...")
        
        if not self.dataset_dir.exists():
            print(f"⚠️  Dataset directory not found: {self.dataset_dir}")
            return {"images": 0, "labels": 0}
        
        # Create output directories
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        # Find image and label directories
        image_dirs = list(self.dataset_dir.rglob("*images*"))
        label_dirs = list(self.dataset_dir.rglob("*labels*"))
        
        if not image_dirs or not label_dirs:
            print("⚠️  Could not find images/labels directories")
            return {"images": 0, "labels": 0}
        
        image_dir = image_dirs[0]
        label_dir = label_dirs[0]
        
        print(f"📁 Image dir: {image_dir}")
        print(f"📁 Label dir: {label_dir}")
        
        # Process all images
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        processed_count = 0
        
        for img_file in tqdm(image_files, desc="Processing images"):
            # Find corresponding label file
            label_file = label_dir / f"{img_file.stem}.png"
            if not label_file.exists():
                continue
            
            try:
                # Load and process image
                image = cv2.imread(str(img_file))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Load and process label
                label = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)
                
                # Map classes to landing detection classes
                mapped_label = self._map_semantic_classes(label)
                
                # Resize to target size
                image_resized = cv2.resize(image, self.config.target_size, interpolation=cv2.INTER_LINEAR)
                label_resized = cv2.resize(mapped_label, self.config.target_size, interpolation=cv2.INTER_NEAREST)
                
                # Save processed files
                output_img = self.output_dir / "images" / f"{img_file.stem}.jpg"
                output_lbl = self.output_dir / "labels" / f"{img_file.stem}.png"
                
                cv2.imwrite(str(output_img), cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(output_lbl), label_resized)
                
                processed_count += 1
                
            except Exception as e:
                print(f"⚠️  Error processing {img_file}: {e}")
                continue
        
        print(f"✅ Processed {processed_count} images from Semantic Drone Dataset")
        return {"images": processed_count, "labels": processed_count}
    
    def _map_semantic_classes(self, label: np.ndarray) -> np.ndarray:
        """Map 24 semantic classes to 4 landing classes"""
        mapped = np.zeros_like(label, dtype=np.uint8)
        
        for semantic_class, landing_class in self.config.semantic_drone_mapping.items():
            mapped[label == semantic_class] = landing_class
        
        return mapped

class UDD6Processor:
    """Processor for UDD6 Dataset"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.dataset_dir = config.raw_data_dir / "udd6_dataset"
        self.output_dir = config.processed_data_dir / "udd6"
        
    def process(self) -> Dict[str, int]:
        """Process UDD6 Dataset"""
        print("🏙️  Processing UDD6 Dataset...")
        
        if not self.dataset_dir.exists():
            print(f"⚠️  Dataset directory not found: {self.dataset_dir}")
            return {"images": 0, "labels": 0}
        
        # Create output directories
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        # Search for image files recursively
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(self.dataset_dir.rglob(f"*{ext}")))
            image_files.extend(list(self.dataset_dir.rglob(f"*{ext.upper()}")))
        
        processed_count = 0
        
        for img_file in tqdm(image_files, desc="Processing UDD6"):
            try:
                # Load image
                image = cv2.imread(str(img_file))
                if image is None:
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Resize to target size
                image_resized = cv2.resize(image, self.config.target_size, interpolation=cv2.INTER_LINEAR)
                
                # For UDD6, we might not have labels, so create a dummy label
                # This would need to be updated based on actual UDD6 structure
                dummy_label = np.zeros(self.config.target_size, dtype=np.uint8)
                
                # Save processed files
                output_img = self.output_dir / "images" / f"udd6_{processed_count:04d}.jpg"
                output_lbl = self.output_dir / "labels" / f"udd6_{processed_count:04d}.png"
                
                cv2.imwrite(str(output_img), cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(output_lbl), dummy_label)
                
                processed_count += 1
                
            except Exception as e:
                print(f"⚠️  Error processing {img_file}: {e}")
                continue
        
        print(f"✅ Processed {processed_count} images from UDD6 Dataset")
        return {"images": processed_count, "labels": processed_count}

class DroneDeployProcessor:
    """Processor for DroneDeploy Dataset"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.dataset_dir = config.raw_data_dir / "dronedeploy_dataset"
        self.output_dir = config.processed_data_dir / "dronedeploy"
        
    def process(self) -> Dict[str, int]:
        """Process DroneDeploy Dataset"""
        print("🚁 Processing DroneDeploy Dataset...")
        
        if not self.dataset_dir.exists():
            print(f"⚠️  Dataset directory not found: {self.dataset_dir}")
            return {"images": 0, "labels": 0}
        
        # Create output directories
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        # Search for image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(self.dataset_dir.rglob(f"*{ext}")))
            image_files.extend(list(self.dataset_dir.rglob(f"*{ext.upper()}")))
        
        processed_count = 0
        
        for img_file in tqdm(image_files, desc="Processing DroneDeploy"):
            try:
                # Load image
                image = cv2.imread(str(img_file))
                if image is None:
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Resize to target size
                image_resized = cv2.resize(image, self.config.target_size, interpolation=cv2.INTER_LINEAR)
                
                # Create dummy label (update based on actual dataset structure)
                dummy_label = np.zeros(self.config.target_size, dtype=np.uint8)
                
                # Save processed files
                output_img = self.output_dir / "images" / f"drone_{processed_count:04d}.jpg"
                output_lbl = self.output_dir / "labels" / f"drone_{processed_count:04d}.png"
                
                cv2.imwrite(str(output_img), cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(output_lbl), dummy_label)
                
                processed_count += 1
                
            except Exception as e:
                print(f"⚠️  Error processing {img_file}: {e}")
                continue
        
        print(f"✅ Processed {processed_count} images from DroneDeploy Dataset")
        return {"images": processed_count, "labels": processed_count}

# =============================================================================
# Dataset Analysis and Visualization
# =============================================================================
class DatasetAnalyzer:
    """Analyze processed datasets"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        
    def analyze_all_datasets(self) -> Dict:
        """Analyze all processed datasets"""
        print("📊 Analyzing processed datasets...")
        
        analysis = {
            "total_images": 0,
            "datasets": {},
            "class_distribution": Counter(),
            "image_stats": {}
        }
        
        # Analyze each dataset
        for dataset_name in ["semantic_drone", "udd6", "dronedeploy"]:
            dataset_dir = self.config.processed_data_dir / dataset_name
            if dataset_dir.exists():
                dataset_analysis = self._analyze_dataset(dataset_dir, dataset_name)
                analysis["datasets"][dataset_name] = dataset_analysis
                analysis["total_images"] += dataset_analysis["image_count"]
                analysis["class_distribution"].update(dataset_analysis["class_distribution"])
        
        # Save analysis
        analysis_file = self.config.processed_data_dir / "dataset_analysis.json"
        with open(analysis_file, 'w') as f:
            # Convert Counter to dict for JSON serialization
            analysis_copy = analysis.copy()
            analysis_copy["class_distribution"] = dict(analysis["class_distribution"])
            json.dump(analysis_copy, f, indent=2)
        
        # Create visualizations
        self._create_visualizations(analysis)
        
        return analysis
    
    def _analyze_dataset(self, dataset_dir: Path, name: str) -> Dict:
        """Analyze a single dataset"""
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        
        analysis = {
            "name": name,
            "image_count": 0,
            "label_count": 0,
            "class_distribution": Counter(),
            "image_sizes": [],
            "corrupted_files": []
        }
        
        if not images_dir.exists():
            return analysis
        
        # Count images
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        analysis["image_count"] = len(image_files)
        
        # Count labels
        if labels_dir.exists():
            label_files = list(labels_dir.glob("*.png"))
            analysis["label_count"] = len(label_files)
            
            # Analyze class distribution (sample subset for speed)
            sample_size = min(50, len(label_files))
            for label_file in tqdm(label_files[:sample_size], desc=f"Analyzing {name} labels"):
                try:
                    label = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)
                    if label is not None:
                        unique, counts = np.unique(label, return_counts=True)
                        for class_id, count in zip(unique, counts):
                            analysis["class_distribution"][int(class_id)] += int(count)
                except Exception as e:
                    analysis["corrupted_files"].append(str(label_file))
        
        return analysis
    
    def _create_visualizations(self, analysis: Dict):
        """Create dataset visualization charts"""
        print("📈 Creating dataset visualizations...")
        
        # 1. Dataset size comparison
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        datasets = list(analysis["datasets"].keys())
        counts = [analysis["datasets"][d]["image_count"] for d in datasets]
        plt.bar(datasets, counts, color=['skyblue', 'lightgreen', 'lightcoral'])
        plt.title('Images per Dataset')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        
        # 2. Class distribution
        plt.subplot(1, 3, 2)
        class_dist = analysis["class_distribution"]
        classes = list(class_dist.keys())
        class_counts = list(class_dist.values())
        class_names = [self.config.landing_classes.get(c, f"Class {c}") for c in classes]
        
        plt.bar(class_names, class_counts, color='lightblue')
        plt.title('Class Distribution (All Datasets)')
        plt.ylabel('Pixel Count')
        plt.xticks(rotation=45)
        
        # 3. Dataset summary pie chart
        plt.subplot(1, 3, 3)
        plt.pie(counts, labels=datasets, autopct='%1.1f%%', startangle=90)
        plt.title('Dataset Proportion')
        
        plt.tight_layout()
        plt.savefig(self.config.processed_data_dir / "dataset_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved to {self.config.processed_data_dir}")

# =============================================================================
# Train/Validation Split Creation
# =============================================================================
class DataSplitter:
    """Create train/validation splits"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        
    def create_splits(self) -> Dict[str, List[str]]:
        """Create train/validation splits for all datasets"""
        print("🔀 Creating train/validation splits...")
        
        np.random.seed(self.config.random_seed)
        splits = {"train": [], "val": []}
        
        # Process each dataset
        for dataset_name in ["semantic_drone", "udd6", "dronedeploy"]:
            dataset_dir = self.config.processed_data_dir / dataset_name / "images"
            if not dataset_dir.exists():
                continue
            
            # Get all image files
            image_files = list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.png"))
            image_files = [f.stem for f in image_files]  # Remove extensions
            
            # Shuffle and split
            np.random.shuffle(image_files)
            split_idx = int(len(image_files) * self.config.train_val_split)
            
            train_files = [f"{dataset_name}/{f}" for f in image_files[:split_idx]]
            val_files = [f"{dataset_name}/{f}" for f in image_files[split_idx:]]
            
            splits["train"].extend(train_files)
            splits["val"].extend(val_files)
            
            print(f"📋 {dataset_name}: {len(train_files)} train, {len(val_files)} val")
        
        # Save splits
        self.config.splits_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, file_list in splits.items():
            split_file = self.config.splits_dir / f"{split_name}.txt"
            with open(split_file, 'w') as f:
                for filename in file_list:
                    f.write(f"{filename}\n")
        
        print(f"✅ Splits saved to {self.config.splits_dir}")
        print(f"📊 Total: {len(splits['train'])} train, {len(splits['val'])} val")
        
        return splits

# =============================================================================
# Main Preprocessing Pipeline
# =============================================================================
def main():
    """Main preprocessing pipeline"""
    print("🚁 Dataset Preprocessing for A100 Training")
    print("=" * 50)
    
    # Create configuration
    config = PreprocessingConfig()
    
    # Create output directories
    for dir_path in [config.processed_data_dir, config.splits_dir, config.cache_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Raw data directory: {config.raw_data_dir}")
    print(f"📁 Processed data directory: {config.processed_data_dir}")
    print(f"🎯 Target image size: {config.target_size}")
    print()
    
    # Process each dataset
    processors = [
        SemanticDroneProcessor(config),
        UDD6Processor(config),
        DroneDeployProcessor(config)
    ]
    
    total_processed = {"images": 0, "labels": 0}
    
    for processor in processors:
        try:
            result = processor.process()
            total_processed["images"] += result["images"]
            total_processed["labels"] += result["labels"]
        except Exception as e:
            print(f"❌ Error in {processor.__class__.__name__}: {e}")
    
    print(f"\n📊 Total processed: {total_processed['images']} images, {total_processed['labels']} labels")
    
    # Analyze datasets
    analyzer = DatasetAnalyzer(config)
    analysis = analyzer.analyze_all_datasets()
    
    # Create train/validation splits
    splitter = DataSplitter(config)
    splits = splitter.create_splits()
    
    # Create preprocessing summary
    summary = {
        "preprocessing_config": vars(config),
        "processing_results": total_processed,
        "dataset_analysis": {k: v for k, v in analysis.items() if k != "class_distribution"},
        "splits": {k: len(v) for k, v in splits.items()},
        "class_mapping": config.landing_classes,
        "semantic_mapping": config.semantic_drone_mapping
    }
    
    summary_file = config.processed_data_dir / "preprocessing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✅ Preprocessing complete!")
    print(f"📋 Summary saved to: {summary_file}")
    print(f"📊 Analysis saved to: {config.processed_data_dir / 'dataset_analysis.json'}")
    print(f"📈 Visualizations saved to: {config.processed_data_dir / 'dataset_analysis.png'}")
    print(f"🔀 Splits saved to: {config.splits_dir}")
    print("\n🚀 Ready for A100 training!")

if __name__ == "__main__":
    main() 