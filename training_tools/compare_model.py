#!/usr/bin/env python3
"""
Compare Fine-Tuned Model Predictions with Ground Truth

This script compares the model predictions with the actual ground truth labels
to evaluate model performance visually and quantitatively.
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Color mappings
LANDING_COLORS = {
    0: [0, 0, 0],        # background - black
    1: [0, 255, 0],      # suitable - green
    2: [255, 0, 0],      # obstacle - red  
    3: [255, 255, 0]     # unsafe - yellow
}

LANDING_CLASSES = {
    0: "background",
    1: "suitable",
    2: "obstacle", 
    3: "unsafe"
}

class ModelComparator:
    """Compare model predictions with ground truth."""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        
        # Load ONNX model
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"✅ Model loaded: {self.model_path}")
        
    def preprocess_image(self, image, target_size=(512, 512)):
        """Preprocess image for the model."""
        # Resize
        image = cv2.resize(image, target_size)
        
        # Convert to RGB
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # CHW format with batch dimension
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        return image.astype(np.float32)
        
    def predict(self, image):
        """Run inference on an image."""
        input_tensor = self.preprocess_image(image)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        prediction = outputs[0][0]
        return np.argmax(prediction, axis=0)
        
    def load_ground_truth(self, label_path):
        """Load ground truth label."""
        label = cv2.imread(str(label_path), cv2.IMREAD_COLOR)
        if label is None:
            return None
            
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        label = cv2.resize(label, (512, 512))
        
        # Convert RGB to class indices
        gt_map = np.zeros((512, 512), dtype=np.uint8)
        
        # Map RGB colors to classes
        suitable_mask = np.all(label == [0, 255, 0], axis=2)  # Green
        obstacle_mask = np.all(label == [255, 0, 0], axis=2)   # Red
        unsafe_mask = np.all(label == [255, 255, 0], axis=2)   # Yellow
        
        gt_map[suitable_mask] = 1
        gt_map[obstacle_mask] = 2
        gt_map[unsafe_mask] = 3
        # Everything else is background (0)
        
        return gt_map
        
    def create_colored_mask(self, mask):
        """Convert class mask to colored visualization."""
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in LANDING_COLORS.items():
            colored_mask[mask == class_id] = color
        return colored_mask
        
    def compute_metrics(self, y_true, y_pred):
        """Compute detailed metrics."""
        # Flatten arrays
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Overall accuracy
        accuracy = np.mean(y_true_flat == y_pred_flat)
        
        # Per-class metrics
        report = classification_report(
            y_true_flat, y_pred_flat, 
            labels=list(range(4)),
            target_names=list(LANDING_CLASSES.values()),
            output_dict=True,
            zero_division=0
        )
        
        # IoU for each class
        iou_scores = {}
        for class_id in range(4):
            true_mask = (y_true_flat == class_id)
            pred_mask = (y_pred_flat == class_id)
            
            intersection = np.logical_and(true_mask, pred_mask).sum()
            union = np.logical_or(true_mask, pred_mask).sum()
            
            if union == 0:
                iou_scores[class_id] = 0.0
            else:
                iou_scores[class_id] = intersection / union
                
        # Mean IoU
        mean_iou = np.mean(list(iou_scores.values()))
        
        return {
            'accuracy': accuracy,
            'mean_iou': mean_iou,
            'iou_per_class': iou_scores,
            'classification_report': report
        }
        
    def visualize_comparison(self, image, ground_truth, prediction, metrics, save_path=None):
        """Create comprehensive comparison visualization."""
        # Create colored masks
        gt_colored = self.create_colored_mask(ground_truth)
        pred_colored = self.create_colored_mask(prediction)
        
        # Create difference mask
        diff_mask = (ground_truth != prediction).astype(np.uint8) * 255
        diff_colored = np.stack([diff_mask, np.zeros_like(diff_mask), np.zeros_like(diff_mask)], axis=2)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        if len(image.shape) == 3:
            axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image', fontsize=14)
        axes[0, 0].axis('off')
        
        # Ground truth
        axes[0, 1].imshow(gt_colored)
        axes[0, 1].set_title('Ground Truth', fontsize=14)
        axes[0, 1].axis('off')
        
        # Prediction
        axes[0, 2].imshow(pred_colored)
        axes[0, 2].set_title('Model Prediction', fontsize=14)
        axes[0, 2].axis('off')
        
        # Difference
        axes[1, 0].imshow(diff_colored)
        axes[1, 0].set_title('Differences (Red = Wrong)', fontsize=14)
        axes[1, 0].axis('off')
        
        # Confusion matrix
        y_true_flat = ground_truth.flatten()
        y_pred_flat = prediction.flatten()
        cm = confusion_matrix(y_true_flat, y_pred_flat, labels=list(range(4)))
        
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 1], 
                   xticklabels=list(LANDING_CLASSES.values()),
                   yticklabels=list(LANDING_CLASSES.values()))
        axes[1, 1].set_title('Confusion Matrix', fontsize=14)
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        # Metrics text
        metrics_text = f"""Performance Metrics:
        
Overall Accuracy: {metrics['accuracy']:.3f}
Mean IoU: {metrics['mean_iou']:.3f}

IoU per Class:
• Background: {metrics['iou_per_class'][0]:.3f}
• Suitable: {metrics['iou_per_class'][1]:.3f}
• Obstacle: {metrics['iou_per_class'][2]:.3f}
• Unsafe: {metrics['iou_per_class'][3]:.3f}

F1-Scores:
• Background: {metrics['classification_report']['background']['f1-score']:.3f}
• Suitable: {metrics['classification_report']['suitable']['f1-score']:.3f}
• Obstacle: {metrics['classification_report']['obstacle']['f1-score']:.3f}
• Unsafe: {metrics['classification_report']['unsafe']['f1-score']:.3f}
        """
        
        axes[1, 2].text(0.05, 0.95, metrics_text, transform=axes[1, 2].transAxes,
                        verticalalignment='top', fontsize=10, fontfamily='monospace')
        axes[1, 2].set_title('Performance Metrics', fontsize=14)
        axes[1, 2].axis('off')
        
        # Add legend
        legend_elements = []
        for class_id, class_name in LANDING_CLASSES.items():
            color = [c/255.0 for c in LANDING_COLORS[class_id]]
            legend_elements.append(plt.Rectangle((0,0),1,1, color=color, label=class_name))
        
        fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=4)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Comparison saved: {save_path}")
        else:
            plt.show()
            
def main():
    parser = argparse.ArgumentParser(description="Compare model predictions with ground truth")
    parser.add_argument("--model", type=str, 
                       default="fine_tuned_models/bisenetv2_uav_landing.onnx",
                       help="Path to ONNX model")
    parser.add_argument("--data_path", type=str,
                       default="../../datasets/drone_deploy_dataset_intermediate/dataset-medium",
                       help="Path to dataset")
    parser.add_argument("--image_id", type=str,
                       help="Specific image ID to test (e.g., '107f24d6e9_F1BE1D4184INSPIRE')")
    parser.add_argument("--save", type=str,
                       help="Path to save comparison visualization")
    parser.add_argument("--random", action="store_true",
                       help="Test on random image")
    
    args = parser.parse_args()
    
    print("🔍 Model vs Ground Truth Comparison")
    print("=" * 40)
    
    # Initialize comparator
    comparator = ModelComparator(args.model)
    
    # Get test image
    data_path = Path(args.data_path)
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"
    
    if args.image_id:
        # Use specific image
        image_file = images_dir / f"{args.image_id}-ortho.tif"
        label_file = labels_dir / f"{args.image_id}-label.png"
    elif args.random:
        # Random image
        image_files = list(images_dir.glob("*-ortho.tif"))
        if not image_files:
            print("❌ No images found")
            return
            
        import random
        image_file = random.choice(image_files)
        # Get corresponding label
        image_id = image_file.stem.replace("-ortho", "")
        label_file = labels_dir / f"{image_id}-label.png"
    else:
        print("Please specify --image_id or --random")
        return
        
    # Load image and label
    print(f"📸 Testing: {image_file.name}")
    
    image = cv2.imread(str(image_file))
    if image is None:
        print(f"❌ Could not load image: {image_file}")
        return
        
    ground_truth = comparator.load_ground_truth(label_file)
    if ground_truth is None:
        print(f"❌ Could not load label: {label_file}")
        return
        
    # Run prediction
    print("🔮 Running inference...")
    prediction = comparator.predict(image)
    
    # Compute metrics
    print("📊 Computing metrics...")
    metrics = comparator.compute_metrics(ground_truth, prediction)
    
    # Print results
    print(f"\\n🎯 Results:")
    print(f"Overall Accuracy: {metrics['accuracy']:.1%}")
    print(f"Mean IoU: {metrics['mean_iou']:.1%}")
    print("\\nPer-class IoU:")
    for class_id, iou in metrics['iou_per_class'].items():
        class_name = LANDING_CLASSES[class_id]
        print(f"  {class_name:10}: {iou:.1%}")
        
    # Create visualization
    print("\\n🎨 Creating comparison...")
    comparator.visualize_comparison(image, ground_truth, prediction, metrics, args.save)
    
    print("\\n✅ Comparison complete!")

if __name__ == "__main__":
    main()
