#!/usr/bin/env python3
"""
Dataset Mapping Verification Script
==================================

Verifies that all three datasets (SDD, DroneDeploy, UDD6) have 
consistent class mappings for progressive training.
"""

import numpy as np
from datasets.semantic_drone_dataset import SemanticDroneDataset
from datasets.dronedeploy_1024_dataset import DroneDeploy1024Dataset
from datasets.udd6_dataset import UDD6Dataset

def print_mapping_analysis():
    """Print detailed analysis of all dataset mappings."""
    
    print("🔍 DATASET MAPPING VERIFICATION")
    print("=" * 60)
    
    # Unified class names (target)
    unified_classes = {
        0: "ground",       # Safe flat landing surfaces
        1: "vegetation",   # Trees, grass (emergency acceptable)
        2: "building",     # Buildings, obstacles (avoid)
        3: "water",        # Water bodies (critical hazard)
        4: "car",          # Vehicles (dynamic obstacles)
        5: "clutter"       # Unknown/mixed areas
    }
    
    print("🎯 TARGET UNIFIED CLASSES:")
    for class_id, name in unified_classes.items():
        print(f"   {class_id}: {name}")
    print()
    
    # Check Semantic Drone Dataset
    print("🛩️  SEMANTIC DRONE DATASET (Stage 1):")
    sdd_dataset = SemanticDroneDataset(
        data_root="../datasets/Aerial_Semantic_Segmentation_Drone_Dataset/dataset/semantic_drone_dataset",
        split="train",
        class_mapping="unified_6_class"
    )
    
    print("   Original → Landing Class Mapping:")
    for orig_class, landing_class in sdd_dataset.class_mapping.items():
        orig_name = sdd_dataset.original_classes.get(orig_class, f"unknown_{orig_class}")
        landing_name = unified_classes[landing_class]
        print(f"     {orig_class:2d}: {orig_name:15s} → {landing_class}: {landing_name}")
    
    print(f"   Landing Classes: {sdd_dataset.landing_classes}")
    print()
    
    # Check DroneDeploy Dataset
    print("🏢 DRONEDEPLOY DATASET (Stage 2):")
    dd_mapping = DroneDeploy1024Dataset.DRONEDEPLOY_TO_LANDING
    dd_classes = DroneDeploy1024Dataset.LANDING_CLASSES
    
    print("   Original → Landing Class Mapping:")
    for orig_class, landing_class in dd_mapping.items():
        landing_name = dd_classes[landing_class]
        print(f"     {orig_class:3d}: {'':15s} → {landing_class}: {landing_name}")
    
    print(f"   Landing Classes: {dd_classes}")
    print()
    
    # Check UDD6 Dataset
    print("🏙️  UDD6 DATASET (Stage 3):")
    udd_classes = {
        0: "other", 1: "facade", 2: "road", 3: "vegetation", 4: "vehicle", 5: "roof"
    }
    udd_mapping = {0: 5, 1: 2, 2: 0, 3: 1, 4: 4, 5: 2}
    udd_landing = {
        0: "ground", 1: "vegetation", 2: "obstacle", 3: "water", 4: "vehicle", 5: "other"
    }
    
    print("   Original → Landing Class Mapping:")
    for orig_class, landing_class in udd_mapping.items():
        orig_name = udd_classes[orig_class]
        landing_name = udd_landing[landing_class]
        print(f"     {orig_class}: {orig_name:15s} → {landing_class}: {landing_name}")
    
    print(f"   Landing Classes: {udd_landing}")
    print()
    
    # Consistency Check
    print("✅ CONSISTENCY CHECK:")
    
    # Check if all datasets map to the same 6 classes
    all_consistent = True
    
    for class_id in range(6):
        sdd_name = sdd_dataset.landing_classes[class_id]
        dd_name = dd_classes[class_id]
        udd_name = udd_landing[class_id]
        
        if class_id == 2:  # Building vs obstacle
            consistent = (sdd_name == "building" and dd_name == "building" and 
                         udd_name in ["obstacle", "building"])
        elif class_id == 3:  # Water (not present in UDD6)
            consistent = (sdd_name == "water" and dd_name == "water")
        else:
            consistent = (sdd_name == dd_name == udd_name or 
                         (sdd_name == dd_name and class_id == 3))  # Water exception
        
        status = "✅" if consistent else "❌"
        print(f"   Class {class_id}: {status} SDD:{sdd_name} | DD:{dd_name} | UDD:{udd_name}")
        
        if not consistent:
            all_consistent = False
    
    print()
    if all_consistent:
        print("🎉 ALL DATASETS HAVE CONSISTENT MAPPINGS!")
    else:
        print("⚠️  INCONSISTENCIES DETECTED - TRAINING MAY BE UNSTABLE")
    
    print()
    
    # Show class distribution expectations
    print("📊 EXPECTED CLASS CHARACTERISTICS:")
    print("   0: ground     - Should be RARE in aerial images")
    print("   1: vegetation - Should be COMMON (trees, grass)")
    print("   2: building   - Should be MODERATE (urban structures)")
    print("   3: water      - Should be RARE (pools, rivers)")
    print("   4: car        - Should be MODERATE (vehicles)")
    print("   5: clutter    - Should be SMALL (mixed/unknown)")

if __name__ == "__main__":
    print_mapping_analysis() 