#!/usr/bin/env python3
"""Quick test to check DroneDeploy label values"""

import cv2
import numpy as np
from pathlib import Path

# Check what's actually in the DroneDeploy label files
label_dir = Path("../datasets/drone_deploy_dataset_intermediate/dataset-medium/labels")

if label_dir.exists():
    label_files = list(label_dir.glob("*.png"))[:5]  # Check first 5 files
    
    print("🔍 Analyzing DroneDeploy label files...")
    
    all_unique_values = set()
    
    for i, label_file in enumerate(label_files):
        try:
            label = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)
            if label is not None:
                unique_values = np.unique(label)
                all_unique_values.update(unique_values)
                
                print(f"\nFile {i+1}: {label_file.name}")
                print(f"  Shape: {label.shape}")
                print(f"  Unique values: {sorted(unique_values)}")
                print(f"  Value distribution:")
                for val in sorted(unique_values):
                    count = np.sum(label == val)
                    percent = count / label.size * 100
                    print(f"    {val}: {count} pixels ({percent:.1f}%)")
        except Exception as e:
            print(f"Error reading {label_file}: {e}")
    
    print(f"\n📊 All unique values across files: {sorted(all_unique_values)}")
    
    # Check mapping
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
    
    print(f"\n🗺️  Mapping coverage:")
    mapped_values = set(DRONEDEPLOY_TO_LANDING.keys())
    unmapped_values = all_unique_values - mapped_values
    
    print(f"   Mapped values: {sorted(mapped_values)}")
    print(f"   Unmapped values: {sorted(unmapped_values)}")
    
    if unmapped_values:
        print(f"   ⚠️  WARNING: {len(unmapped_values)} unmapped values will default to class 0 (ground)!")
        print(f"   This could explain class distribution issues.")
    
else:
    print(f"❌ Label directory not found: {label_dir}") 