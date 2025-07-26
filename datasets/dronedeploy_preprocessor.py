#!/usr/bin/env python3
"""
DroneDeploy Dataset Preprocessor
===============================

One-time preprocessing script to convert large DroneDeploy GeoTIFF files
into individual 512x512 patch files for ultra-fast training.

This solves the architectural bottleneck by:
1. Reading each large TIFF once 
2. Extracting all valid patches
3. Saving each patch as a small PNG file
4. Creating a metadata CSV for fast dataset loading

Run this script once before training to generate optimized patch files.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Dict
from tqdm import tqdm
import hashlib
import json

class DroneDeployPreprocessor:
    """
    Preprocess DroneDeploy dataset into individual patch files for maximum training speed.
    """
    
    # DroneDeploy classes → Landing classes (same as original)
    DRONEDEPLOY_TO_LANDING = {
        81: 2,   # Building → building (avoid)
        91: 0,   # Road → ground (safe primary)
        99: 4,   # Car → car (dynamic obstacle)
        105: 5,  # Background/Clutter → clutter (caution)
        132: 1,  # Trees → vegetation (safe secondary)
        155: 3,  # Pool/Water → water (critical hazard)
        0: 5,    # Unknown → clutter
        255: 5,  # Background → clutter
    }
    
    def __init__(
        self,
        source_data_root: str,
        output_data_root: str,
        patch_size: int = 512,
        stride_factor: float = 0.5,
        min_valid_pixels: float = 0.1
    ):
        self.source_root = Path(source_data_root)
        self.output_root = Path(output_data_root)
        self.patch_size = patch_size
        self.stride = int(patch_size * stride_factor)
        self.min_valid_pixels = min_valid_pixels
        
        # Create output directories
        self.patches_dir = self.output_root / "patches"
        self.images_dir = self.patches_dir / "images"
        self.labels_dir = self.patches_dir / "labels"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Create fast lookup table for class mapping
        self.mapping_lut = np.zeros(256, dtype=np.uint8)
        for dd_class, landing_class in self.DRONEDEPLOY_TO_LANDING.items():
            if dd_class < 256:
                self.mapping_lut[dd_class] = landing_class
        
        print(f"🔧 DroneDeploy Preprocessor initialized:")
        print(f"   Source: {self.source_root}")
        print(f"   Output: {self.output_root}")
        print(f"   Patch size: {patch_size}x{patch_size}")
        print(f"   Stride: {self.stride}")
    
    def find_image_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching image and label file pairs."""
        images_dir = self.source_root / 'images'
        labels_dir = self.source_root / 'labels'
        
        image_files = []
        
        print(f"🔍 Scanning for image pairs...")
        
        # Find all .tif images
        for img_path in images_dir.glob("*.tif"):
            # Handle naming pattern: image_name-ortho.tif → image_name-label.png
            img_stem = img_path.stem
            
            if img_stem.endswith('-ortho'):
                base_name = img_stem[:-6]  # Remove '-ortho'
                label_name = f"{base_name}-label.png"
            else:
                label_name = f"{img_stem}-label.png"
            
            label_path = labels_dir / label_name
            
            if label_path.exists():
                # Check file sizes
                try:
                    img_size = img_path.stat().st_size
                    label_size = label_path.stat().st_size
                    
                    if img_size > 1024 * 1024 and label_size > 1024:
                        image_files.append((img_path, label_path))
                        print(f"   ✓ {img_path.name} → {label_name}")
                    else:
                        print(f"   ⚠️  Skipping small file: {img_path.name}")
                        
                except Exception as e:
                    print(f"   ❌ Error checking {img_path.name}: {e}")
            else:
                print(f"   ⚠️  No label found for: {img_path.name}")
        
        print(f"✅ Found {len(image_files)} valid image pairs")
        return sorted(image_files)
    
    def map_to_landing_classes(self, dronedeploy_label: np.ndarray) -> np.ndarray:
        """Map DroneDeploy classes to landing classes using fast vectorized lookup."""
        clipped_label = np.clip(dronedeploy_label, 0, 255)
        return self.mapping_lut[clipped_label]
    
    def is_valid_patch(self, label_patch: np.ndarray) -> bool:
        """Check if patch has sufficient valid content."""
        # Count non-clutter pixels
        valid_pixels = np.sum(label_patch != 5)  # Not clutter
        total_pixels = label_patch.size
        valid_ratio = valid_pixels / total_pixels
        
        # Require minimum valid content
        if valid_ratio < self.min_valid_pixels:
            return False
        
        # Prefer patches with multiple classes
        unique_classes = len(np.unique(label_patch))
        return unique_classes >= 2
    
    def extract_patches_from_image(
        self, 
        image_path: Path, 
        label_path: Path,
        image_id: str
    ) -> List[Dict]:
        """Extract all valid patches from a single image pair."""
        
        print(f"   📷 Processing: {image_path.name}")
        
        # Load images
        image = cv2.imread(str(image_path))
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None or label is None:
            print(f"   ❌ Failed to load: {image_path.name}")
            return []
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        print(f"   📐 Image size: {w}x{h}")
        
        patch_metadata = []
        patch_count = 0
        
        # Extract patches
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                # Extract patch
                img_patch = image[y:y+self.patch_size, x:x+self.patch_size]
                label_patch = label[y:y+self.patch_size, x:x+self.patch_size]
                
                # Map to landing classes
                landing_patch = self.map_to_landing_classes(label_patch)
                
                # Validate patch
                if self.is_valid_patch(landing_patch):
                    # Generate unique patch filename
                    patch_name = f"{image_id}_{x:04d}_{y:04d}"
                    
                    # Save patch files
                    img_file = self.images_dir / f"{patch_name}.png"
                    label_file = self.labels_dir / f"{patch_name}.png"
                    
                    # Save image patch
                    cv2.imwrite(str(img_file), cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR))
                    
                    # Save label patch
                    cv2.imwrite(str(label_file), landing_patch)
                    
                    # Store metadata
                    patch_metadata.append({
                        'patch_id': patch_name,
                        'image_file': f"patches/images/{patch_name}.png",
                        'label_file': f"patches/labels/{patch_name}.png",
                        'source_image': str(image_path.name),
                        'source_label': str(label_path.name),
                        'x': x,
                        'y': y,
                        'patch_size': self.patch_size
                    })
                    
                    patch_count += 1
        
        print(f"   ✅ Generated {patch_count} patches")
        return patch_metadata
    
    def create_splits(self, all_patches: List[Dict]) -> Dict[str, List[Dict]]:
        """Create deterministic train/val/test splits."""
        
        # Group patches by source image for consistent splitting
        patches_by_source = {}
        for patch in all_patches:
            source = patch['source_image']
            if source not in patches_by_source:
                patches_by_source[source] = []
            patches_by_source[source].append(patch)
        
        # Sort source images deterministically
        source_images = sorted(patches_by_source.keys())
        
        # Create deterministic split based on hash (6:2:2)
        train_patches, val_patches, test_patches = [], [], []
        
        for source in source_images:
            # Use hash for deterministic assignment
            hash_val = int(hashlib.md5(source.encode()).hexdigest(), 16)
            split_val = hash_val % 10
            
            patches = patches_by_source[source]
            
            if split_val < 6:  # 60% train
                train_patches.extend(patches)
            elif split_val < 8:  # 20% val
                val_patches.extend(patches)
            else:  # 20% test
                test_patches.extend(patches)
        
        print(f"\n📊 Dataset splits:")
        print(f"   Train: {len(train_patches)} patches")
        print(f"   Val: {len(val_patches)} patches")
        print(f"   Test: {len(test_patches)} patches")
        
        return {
            'train': train_patches,
            'val': val_patches,
            'test': test_patches
        }
    
    def process_dataset(self):
        """Main processing function - convert entire dataset to patch files."""
        
        print(f"\n🚀 Starting DroneDeploy dataset preprocessing...")
        
        # Find image pairs
        image_pairs = self.find_image_pairs()
        
        if not image_pairs:
            print("❌ No valid image pairs found!")
            return
        
        # Process each image pair
        all_patches = []
        
        for i, (img_path, label_path) in enumerate(tqdm(image_pairs, desc="Processing images")):
            # Create unique image ID
            image_id = f"img_{i:03d}_{img_path.stem}"
            
            # Extract patches
            patches = self.extract_patches_from_image(img_path, label_path, image_id)
            all_patches.extend(patches)
        
        print(f"\n✅ Total patches generated: {len(all_patches)}")
        
        # Create splits
        splits = self.create_splits(all_patches)
        
        # Save metadata files
        for split_name, split_patches in splits.items():
            metadata_file = self.output_root / f"{split_name}_metadata.csv"
            df = pd.DataFrame(split_patches)
            df.to_csv(metadata_file, index=False)
            print(f"💾 Saved {split_name} metadata: {metadata_file}")
        
        # Save processing parameters
        params = {
            'patch_size': self.patch_size,
            'stride': self.stride,
            'min_valid_pixels': self.min_valid_pixels,
            'total_patches': len(all_patches),
            'source_images': len(image_pairs),
            'class_mapping': self.DRONEDEPLOY_TO_LANDING
        }
        
        params_file = self.output_root / "preprocessing_params.json"
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        print(f"\n🎉 Preprocessing complete!")
        print(f"   Total patches: {len(all_patches)}")
        print(f"   Output directory: {self.output_root}")
        print(f"   Disk usage: ~{len(all_patches) * 0.5:.1f} MB")
        print(f"\n💡 Now you can use the optimized DroneDeploy dataset for ultra-fast training!")


def main():
    """Main preprocessing script."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess DroneDeploy dataset for fast training')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to source DroneDeploy dataset (with images/ and labels/ dirs)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output preprocessed dataset')
    parser.add_argument('--patch_size', type=int, default=512,
                        help='Patch size (default: 512)')
    parser.add_argument('--stride_factor', type=float, default=0.5,
                        help='Stride factor (default: 0.5)')
    parser.add_argument('--min_valid_pixels', type=float, default=0.1,
                        help='Minimum valid pixel ratio (default: 0.1)')
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = DroneDeployPreprocessor(
        source_data_root=args.source,
        output_data_root=args.output,
        patch_size=args.patch_size,
        stride_factor=args.stride_factor,
        min_valid_pixels=args.min_valid_pixels
    )
    
    # Process dataset
    preprocessor.process_dataset()


if __name__ == "__main__":
    main() 