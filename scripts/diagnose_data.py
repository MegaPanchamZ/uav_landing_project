#!/usr/bin/env python3
"""
Dataset Ground Truth Visualizer
===============================

This script visualizes the ground truth from the SemanticDroneDataset after
the 24 original classes have been mapped to the 4 landing-relevant classes.

This is a critical diagnostic tool to ensure the model is being taught the
correct semantic concepts.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import sys
import random

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets.semantic_drone_dataset import SemanticDroneDataset, create_semantic_drone_transforms

# --- Configuration ---
DATA_ROOT = "../datasets/Aerial_Semantic_Segmentation_Drone_Dataset/dataset/semantic_drone_dataset"
OUTPUT_DIR = "outputs/diagnostics"
NUM_SAMPLES = 10  # Number of random samples to visualize
CLASS_MAPPING = "enhanced_4_class"

# Define colors for the 4 landing classes for visualization
# (B, G, R) format for OpenCV
COLOR_MAP = {
    0: (0, 0, 0),          # 0: background (Black)
    1: (0, 255, 0),        # 1: safe_landing (Green)
    2: (0, 255, 255),      # 2: caution (Yellow)
    3: (0, 0, 255),        # 3: danger (Red)
}

LEGEND_LABELS = {
    "safe_landing (1)": (0, 255, 0),
    "caution (2)": (0, 255, 255),
    "danger (3)": (0, 0, 255),
    "background (0)": (0, 0, 0),
}

def create_legend_image(labels_colors, width=400, height=200, font_scale=1.0):
    """Creates an image with a legend for the color map."""
    legend = np.full((height, width, 3), 255, dtype=np.uint8)
    
    y_pos = 30
    for label, color in labels_colors.items():
        cv2.rectangle(legend, (20, y_pos - 20), (70, y_pos + 10), color, -1)
        cv2.putText(legend, label, (90, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, (0, 0, 0), 2)
        y_pos += 40
        
    return legend

def visualize_sample(dataset, index, output_dir):
    """Loads a sample, visualizes it, and saves the result."""
    
    # Get a sample without any augmentations to see the raw data
    sample = dataset[index]
    
    # --- 1. Get the raw image and mask ---
    # The dataset returns tensors, so we need to convert them back to numpy images
    image_tensor = sample['image']
    mask_tensor = sample['mask']
    
    # Convert image tensor to OpenCV format (H, W, C) and scale to 0-255
    image_np = image_tensor.permute(1, 2, 0).numpy()
    image_np = (image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    image_np = image_np.astype(np.uint8)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) # Convert RGB to BGR for OpenCV
    
    # Convert mask tensor to numpy array
    mask_np = mask_tensor.numpy().astype(np.uint8)
    
    # --- 2. Create the colorized segmentation mask ---
    color_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        color_mask[mask_np == class_id] = color
        
    # --- 3. Blend the original image and the color mask ---
    blended_image = cv2.addWeighted(image_np, 0.6, color_mask, 0.4, 0)
    
    # --- 4. Create the final output image ---
    # Concatenate original image, color mask, and blended image side-by-side
    h, w, _ = image_np.shape
    
    # Create a header for the visualization
    header = np.full((50, w * 3, 3), 255, dtype=np.uint8)
    cv2.putText(header, f"Sample {index} - {Path(sample['image_path']).name}", (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Create labels for each image
    label_orig = np.full((50, w, 3), 200, dtype=np.uint8)
    cv2.putText(label_orig, "Original Image", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    
    label_mask = np.full((50, w, 3), 200, dtype=np.uint8)
    cv2.putText(label_mask, "Ground Truth Mask", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    label_blend = np.full((50, w, 3), 200, dtype=np.uint8)
    cv2.putText(label_blend, "Blended View", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    top_row = np.concatenate((label_orig, label_mask, label_blend), axis=1)
    bottom_row = np.concatenate((image_np, color_mask, blended_image), axis=1)
    
    final_image = np.concatenate((header, top_row, bottom_row), axis=0)
    
    # --- 5. Add legend to the final image ---
    legend = create_legend_image(LEGEND_LABELS, width=w, height=h)
    # Create a new canvas to place the main visualization and the legend
    final_with_legend_w = bottom_row.shape[1] + legend.shape[1]
    final_with_legend_h = bottom_row.shape[0]
    final_canvas = np.full((final_with_legend_h, final_with_legend_w, 3), 255, dtype=np.uint8)
    final_canvas[:bottom_row.shape[0], :bottom_row.shape[1]] = bottom_row
    final_canvas[:legend.shape[0], bottom_row.shape[1]:] = legend


    # --- 6. Save the final image ---
    output_path = Path(output_dir) / f"ground_truth_sample_{index}.png"
    cv2.imwrite(str(output_path), final_canvas)
    print(f"Saved visualization to {output_path}")

def main():
    """Main function to run the visualization."""
    print("--- Ground Truth Visualizer ---")
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")
    
    # Use a simple transform that only resizes and converts to tensor, no augmentation
    # This ensures we see the "true" data before augmentation is applied.
    vis_transform = create_semantic_drone_transforms(
        input_size=(768, 768), # Use a larger size for better visualization
        is_training=False # No augmentations
    )
    
    # Create the dataset instance for the 'train' split
    try:
        dataset = SemanticDroneDataset(
            data_root=DATA_ROOT,
            split="train",
            transform=vis_transform,
            class_mapping=CLASS_MAPPING
        )
    except ValueError as e:
        print(f"Error creating dataset: {e}")
        print(f"Please ensure the dataset exists at '{DATA_ROOT}'")
        return

    if len(dataset) == 0:
        print("Dataset is empty. Cannot generate visualizations.")
        return

    # Generate and save visualizations for a few random samples
    random_indices = random.sample(range(len(dataset)), min(NUM_SAMPLES, len(dataset)))
    
    for index in random_indices:
        visualize_sample(dataset, index, output_dir)
        
    print("\nVisualization script finished.")
    print(f"Check the '{output_dir}' directory for the output images.")

if __name__ == "__main__":
    main()
