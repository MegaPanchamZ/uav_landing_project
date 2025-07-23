#!/usr/bin/env python3
"""
Test the Fine-Tuned UAV Landing Detection Model

This script loads the fine-tuned ONNX model and tests it on sample images.
"""

import cv2
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Color mapping for visualization
LANDING_COLORS = {
    0: [0, 0, 0],        # background - black
    1: [0, 255, 0],      # suitable - green
    2: [255, 0, 0],      # obstacle - red  
    3: [255, 255, 0]     # unsafe - yellow
}

LANDING_CLASSES = {
    0: "background",
    1: "suitable",      # Safe landing zones
    2: "obstacle",      # Buildings, cars, etc.
    3: "unsafe"         # Water, clutter, etc.
}

class UAVLandingDetector:
    """ONNX-based UAV landing zone detector."""
    
    def __init__(self, model_path, device="auto"):
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Set up ONNX Runtime
        if device == "auto":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif device == "cuda":
            providers = ['CUDAExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
            
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f" Model loaded: {self.model_path}")
        print(f"   Input: {self.input_name}")
        print(f"   Output: {self.output_name}")
        print(f"   Provider: {self.session.get_providers()[0]}")
        
    def preprocess_image(self, image, target_size=(512, 512)):
        """Preprocess image for the model."""
        # Resize
        image = cv2.resize(image, target_size)
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Normalize (same as training)
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Add batch dimension and convert to CHW format
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        image = np.expand_dims(image, axis=0)   # Add batch dim
        
        return image.astype(np.float32)
        
    def predict(self, image):
        """Run inference on an image."""
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        
        # Get segmentation map
        prediction = outputs[0][0]  # Remove batch dimension
        segmentation_map = np.argmax(prediction, axis=0)  # CHW -> HW
        
        return segmentation_map
        
    def visualize_prediction(self, image, prediction, save_path=None):
        """Visualize the prediction on the original image."""
        # Create colored mask
        colored_mask = np.zeros((*prediction.shape, 3), dtype=np.uint8)
        for class_id, color in LANDING_COLORS.items():
            mask = (prediction == class_id)
            colored_mask[mask] = color
            
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        if len(image.shape) == 3:
            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Segmentation mask
        axes[1].imshow(colored_mask)
        axes[1].set_title('Landing Zone Segmentation')
        axes[1].axis('off')
        
        # Overlay
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_rgb = cv2.resize(image_rgb, prediction.shape[::-1])
        
        # Alpha blend
        alpha = 0.6
        overlay = (alpha * image_rgb + (1-alpha) * colored_mask).astype(np.uint8)
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        # Add legend
        legend_elements = []
        for class_id, class_name in LANDING_CLASSES.items():
            color = [c/255.0 for c in LANDING_COLORS[class_id]]
            legend_elements.append(plt.Rectangle((0,0),1,1, color=color, label=class_name))
        
        fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.15, 0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" Visualization saved: {save_path}")
        else:
            plt.show()
            
    def analyze_landing_zones(self, prediction):
        """Analyze the prediction for landing zone statistics."""
        total_pixels = prediction.size
        
        print("\\n Landing Zone Analysis:")
        print("=" * 30)
        
        suitable_pixels = np.sum(prediction == 1)
        suitable_percentage = (suitable_pixels / total_pixels) * 100
        
        if suitable_percentage > 30:
            safety_level = "🟢 SAFE"
        elif suitable_percentage > 15:
            safety_level = "🟡 MARGINAL" 
        else:
            safety_level = "🔴 UNSAFE"
            
        print(f"Safety Assessment: {safety_level}")
        print(f"Suitable Landing Area: {suitable_percentage:.1f}%")
        print()
        
        for class_id, class_name in LANDING_CLASSES.items():
            count = np.sum(prediction == class_id)
            percentage = (count / total_pixels) * 100
            print(f"  {class_name:10}: {count:6d} pixels ({percentage:5.1f}%)")
            
        # Find largest suitable area
        if suitable_pixels > 0:
            suitable_mask = (prediction == 1).astype(np.uint8)
            contours, _ = cv2.findContours(suitable_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_area = max(cv2.contourArea(cnt) for cnt in contours)
                print(f"\\nLargest suitable area: {largest_area:.0f} pixels")
                
                # Estimate real-world size (assuming ~1m/pixel at 100m altitude)
                estimated_size = np.sqrt(largest_area)  # Side length in pixels
                print(f"Estimated landing zone size: ~{estimated_size:.1f}m x {estimated_size:.1f}m")

def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned UAV landing detection model")
    parser.add_argument("--model", type=str, default="fine_tuned_models/bisenetv2_uav_landing.onnx",
                       help="Path to ONNX model file")
    parser.add_argument("--image", type=str,
                       help="Path to test image")
    parser.add_argument("--data_path", type=str, 
                       default="../../datasets/drone_deploy_dataset_intermediate/dataset-medium",
                       help="Path to dataset (for random testing)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--save", type=str,
                       help="Path to save visualization")
    parser.add_argument("--random", action="store_true",
                       help="Test on random image from dataset")
    
    args = parser.parse_args()
    
    print("🚁 UAV Landing Detection Model Test")
    print("=" * 40)
    
    # Load model
    detector = UAVLandingDetector(args.model, args.device)
    
    # Get test image
    if args.image:
        # Use specified image
        image = cv2.imread(args.image)
        if image is None:
            print(f"❌ Could not load image: {args.image}")
            return
        print(f"📸 Testing on: {args.image}")
        
    elif args.random or not args.image:
        # Use random image from dataset
        data_path = Path(args.data_path)
        image_dir = data_path / "images"
        
        if not image_dir.exists():
            print(f"❌ Dataset not found: {data_path}")
            print("Please provide --image or --data_path")
            return
            
        # Get random image
        image_files = list(image_dir.glob("*.tif"))
        if not image_files:
            print(f"❌ No images found in {image_dir}")
            return
            
        import random
        random_file = random.choice(image_files)
        image = cv2.imread(str(random_file))
        print(f"📸 Testing on random image: {random_file.name}")
        
    # Run prediction
    print("🔮 Running inference...")
    prediction = detector.predict(image)
    
    # Analyze results
    detector.analyze_landing_zones(prediction)
    
    # Visualize
    print("\\n🎨 Creating visualization...")
    detector.visualize_prediction(image, prediction, args.save)
    
    print("\\n Testing complete!")

if __name__ == "__main__":
    main()
