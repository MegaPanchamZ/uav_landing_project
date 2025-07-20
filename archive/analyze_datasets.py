#!/usr/bin/env python3
"""
Dataset Label Analysis for Staged Fine-Tuning

Step 1: Analyze DroneDeploy dataset labels (intermediate fine-tuning)
Step 2: Analyze UDD dataset labels (task-specific fine-tuning)
Step 3: Create proper label mappings for the staged approach
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
import json

def analyze_drone_deploy_labels():
    """Analyze DroneDeploy dataset labels for intermediate fine-tuning."""
    
    print("🌍 STEP 1: Analyzing DroneDeploy Dataset (Intermediate Fine-Tuning)")
    print("=" * 70)
    
    labels_dir = Path("../datasets/drone_deploy_dataset_intermediate/dataset-medium/labels")
    
    if not labels_dir.exists():
        print(f"❌ DroneDeploy labels not found: {labels_dir}")
        return None
    
    label_files = list(labels_dir.glob("*.png"))
    print(f"Found {len(label_files)} label files")
    
    # Analyze color distribution
    all_colors = []
    color_counts = Counter()
    
    print("\n📊 Analyzing color values in labels...")
    for i, label_file in enumerate(label_files[:5]):  # Sample first 5
        print(f"  Analyzing {label_file.name}...")
        
        label_img = cv2.imread(str(label_file))
        if label_img is None:
            continue
            
        # Get unique colors
        pixels = label_img.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)
        
        for color in unique_colors:
            color_tuple = tuple(color)  # BGR format
            color_counts[color_tuple] += 1
            all_colors.append(color_tuple)
    
    # Display most common colors
    print(f"\n🎨 Most Common Colors in DroneDeploy Labels:")
    for color_bgr, count in color_counts.most_common(10):
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])  # BGR to RGB
        print(f"  BGR: {str(color_bgr):15} RGB: {str(color_rgb):15} (appears in {count} images)")
    
    # Known DroneDeploy mapping from intermediate_training.md
    known_mapping = {
        (75, 25, 230): "BUILDING",
        (180, 30, 145): "CLUTTER", 
        (75, 180, 60): "VEGETATION",
        (48, 130, 245): "WATER",
        (255, 255, 255): "GROUND",
        (200, 130, 0): "CAR",
        (255, 0, 255): "IGNORE"
    }
    
    print(f"\n📋 Expected DroneDeploy Classes (from docs):")
    for rgb, class_name in known_mapping.items():
        bgr = (rgb[2], rgb[1], rgb[0])
        found = "✅" if bgr in color_counts else "❌"
        print(f"  {found} {class_name:12} - RGB: {rgb} BGR: {bgr}")
    
    return {
        'dataset': 'DroneDeploy',
        'num_files': len(label_files),
        'colors_found': dict(color_counts.most_common(10)),
        'known_mapping': known_mapping,
        'purpose': 'intermediate_fine_tuning'
    }

def analyze_udd_labels():
    """Analyze UDD dataset labels for task-specific fine-tuning."""
    
    print(f"\n🚁 STEP 2: Analyzing UDD Dataset (Task-Specific Fine-Tuning)")
    print("=" * 60)
    
    udd_base = Path("../datasets/UDD")
    
    # Check different UDD versions
    udd_dirs = []
    for version in ['UDD5', 'UDD6']:
        version_path = udd_base / "UDD" / version
        if version_path.exists():
            udd_dirs.append(version_path)
    
    if not udd_dirs:
        print(f"❌ UDD dataset not found in {udd_base}")
        return None
    
    print(f"Found UDD versions: {[d.name for d in udd_dirs]}")
    
    all_udd_info = {}
    
    for udd_dir in udd_dirs:
        print(f"\n📁 Analyzing {udd_dir.name}...")
        
        # Look for label directories
        possible_label_dirs = [
            udd_dir / "TrainingSet" / "Labels",
            udd_dir / "TestSet" / "Labels", 
            udd_dir / "labels",
            udd_dir / "Labels"
        ]
        
        labels_dir = None
        for label_dir in possible_label_dirs:
            if label_dir.exists():
                labels_dir = label_dir
                break
        
        if not labels_dir:
            print(f"  ❌ No labels directory found")
            continue
            
        print(f"  📂 Labels found in: {labels_dir.relative_to(udd_dir)}")
        
        # Find label files
        label_files = []
        for ext in ['*.png', '*.jpg', '*.tif']:
            label_files.extend(list(labels_dir.glob(ext)))
        
        if not label_files:
            print(f"  ❌ No label files found")
            continue
            
        print(f"  📊 Found {len(label_files)} label files")
        
        # Analyze colors in first few labels
        color_counts = Counter()
        for i, label_file in enumerate(label_files[:3]):
            label_img = cv2.imread(str(label_file))
            if label_img is not None:
                pixels = label_img.reshape(-1, 3)
                unique_colors = np.unique(pixels, axis=0)
                for color in unique_colors:
                    color_counts[tuple(color)] += 1
        
        print(f"  🎨 Top colors found:")
        for color_bgr, count in color_counts.most_common(8):
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            print(f"    BGR: {str(color_bgr):15} RGB: {str(color_rgb):15}")
        
        all_udd_info[udd_dir.name] = {
            'labels_dir': str(labels_dir),
            'num_files': len(label_files),
            'colors_found': dict(color_counts.most_common(8))
        }
    
    # Known UDD classes from path.md
    udd_classes = {
        "Road": "safe landing surface",
        "Vegetation": "potentially safe if short grass", 
        "Facade": "high obstacle (building wall)",
        "Roof": "high obstacle (building roof)",
        "Vehicle": "high obstacle (car/truck)",
        "Other": "ignore class"
    }
    
    print(f"\n📋 Expected UDD Classes (for UAV landing):")
    for class_name, description in udd_classes.items():
        print(f"  • {class_name:12} - {description}")
    
    return {
        'dataset': 'UDD',
        'versions': all_udd_info,
        'known_classes': udd_classes,
        'purpose': 'task_specific_fine_tuning'
    }

def create_staged_mapping():
    """Create the staged fine-tuning class mapping strategy."""
    
    print(f"\n🎯 STEP 3: Staged Fine-Tuning Strategy")
    print("=" * 40)
    
    # Stage 1: DroneDeploy (7 classes) - General aerial view
    stage1_mapping = {
        "BUILDING": 1,    # Urban structures
        "VEGETATION": 2,  # Green areas  
        "GROUND": 3,      # Bare ground/roads
        "WATER": 4,       # Water bodies
        "CAR": 5,         # Vehicles
        "CLUTTER": 6,     # Mixed/unclear areas
        "IGNORE": 0       # Background/ignore
    }
    
    print("Stage 1 - DroneDeploy Intermediate Fine-Tuning (7 classes):")
    for class_name, class_id in stage1_mapping.items():
        print(f"  {class_id}: {class_name:12} - General aerial features")
    
    # Stage 2: UDD (4 classes) - UAV landing specific  
    stage2_mapping = {
        "ignore": 0,           # Background/other
        "safe_landing": 1,     # Safe flat surfaces (road, short vegetation)
        "high_obstacle": 2,    # Buildings, vehicles (facade, roof, vehicle)
        "unsafe_terrain": 3    # Dangerous areas (water equivalent)
    }
    
    print(f"\nStage 2 - UDD Task-Specific Fine-Tuning (4 classes):")
    for class_name, class_id in stage2_mapping.items():
        print(f"  {class_id}: {class_name:15} - Landing safety classification")
    
    # Class evolution from Stage 1 to Stage 2
    class_evolution = {
        # Stage 1 -> Stage 2 mapping
        "BUILDING": "high_obstacle",
        "VEGETATION": "safe_landing",  # Assuming short grass
        "GROUND": "safe_landing", 
        "WATER": "unsafe_terrain",
        "CAR": "high_obstacle",
        "CLUTTER": "unsafe_terrain",
        "IGNORE": "ignore"
    }
    
    print(f"\n🔄 Class Evolution (Stage 1 -> Stage 2):")
    for s1_class, s2_class in class_evolution.items():
        print(f"  {s1_class:12} -> {s2_class}")
    
    return {
        'stage1': stage1_mapping,
        'stage2': stage2_mapping, 
        'evolution': class_evolution
    }

def main():
    """Main analysis function."""
    
    print("🛩️  Staged Fine-Tuning Dataset Analysis")
    print("=" * 50)
    print("BiSeNetV2 -> DroneDeploy -> UDD-6 -> Landing Detection")
    
    # Analyze both datasets
    drone_deploy_info = analyze_drone_deploy_labels()
    udd_info = analyze_udd_labels()
    mapping_strategy = create_staged_mapping()
    
    # Save analysis results
    analysis_result = {
        'timestamp': '2025-07-20',
        'datasets': {
            'drone_deploy': drone_deploy_info,
            'udd': udd_info
        },
        'mapping_strategy': mapping_strategy
    }
    
    with open('dataset_analysis.json', 'w') as f:
        json.dump(analysis_result, f, indent=2, default=str)
    
    print(f"\n💾 Analysis saved to dataset_analysis.json")
    
    print(f"\n🚀 Next Steps:")
    print("1. Create DroneDeploy label converter (RGB -> class IDs)")
    print("2. Create UDD label converter (RGB -> class IDs)")  
    print("3. Implement Stage 1 training script (BiSeNetV2 + DroneDeploy)")
    print("4. Implement Stage 2 training script (Stage1 model + UDD)")
    print("5. Export final model to ONNX for deployment")

if __name__ == "__main__":
    main()
