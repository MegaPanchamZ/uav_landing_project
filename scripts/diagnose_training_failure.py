#!/usr/bin/env python3
"""
Training Failure Diagnostic Tool
===============================

This script analyzes the catastrophic failure in progressive training by:
1. Visualizing model predictions on validation data
2. Checking class distributions in datasets
3. Verifying data loading and class mappings
4. Analyzing loss components
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple
import seaborn as sns
from collections import Counter

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from datasets.drone_deploy_dataset import DroneDeployDataset, create_drone_deploy_transforms
from datasets.udd_dataset import UDDDataset, create_udd_transforms  
from datasets.semantic_drone_dataset import SemanticDroneDataset, create_semantic_drone_transforms
from models.enhanced_architectures import create_enhanced_model
from torch.utils.data import DataLoader


class TrainingDiagnostic:
    """Comprehensive diagnostic tool for training failures."""
    
    def __init__(self, model_paths: Dict[str, str]):
        self.model_paths = model_paths
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory for diagnostic plots
        self.output_dir = Path("outputs/diagnostics")
        self.output_dir.mkdir(exist_ok=True)
        
        print("🔍 Training Failure Diagnostic Tool")
        print("=" * 50)
    
    def load_model(self, model_path: str, in_channels: int = 3) -> torch.nn.Module:
        """Load a trained model checkpoint."""
        print(f"📂 Loading model: {model_path}")
        
        # Create model architecture
        model = create_enhanced_model(
            model_type="mmseg_bisenetv2",
            num_classes=4,
            in_channels=in_channels,
            uncertainty_estimation=True,
            pretrained_path=None  # Don't load pretrained, we'll load our checkpoint
        )
        
        # Load checkpoint (fix PyTorch 2.6 weights_only issue)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"✅ Model loaded from stage {checkpoint.get('stage', 'unknown')}")
        print(f"   Stage name: {checkpoint.get('stage_name', 'unknown')}")
        print(f"   Metrics: {checkpoint.get('metrics', {})}")
        
        return model, checkpoint
    
    def analyze_dataset_distribution(self, dataset_name: str, dataset) -> Dict[str, Any]:
        """Analyze class distribution in a dataset."""
        print(f"\n📊 Analyzing {dataset_name} Dataset Distribution")
        
        class_counts = Counter()
        total_pixels = 0
        
        # Sample a subset to avoid memory issues
        sample_size = min(50, len(dataset))
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        
        for idx in indices:
            if hasattr(dataset, '__getitem__'):
                try:
                    sample = dataset[idx]
                    if isinstance(sample, dict) and 'mask' in sample:
                        mask = sample['mask']
                    else:
                        # Handle different dataset formats
                        continue
                        
                    if isinstance(mask, torch.Tensor):
                        mask = mask.numpy()
                    
                    unique, counts = np.unique(mask, return_counts=True)
                    for cls, count in zip(unique, counts):
                        class_counts[int(cls)] += int(count)
                        total_pixels += int(count)
                        
                except Exception as e:
                    print(f"   ⚠️ Error processing sample {idx}: {e}")
                    continue
        
        # Calculate percentages
        class_percentages = {cls: (count / total_pixels) * 100 
                           for cls, count in class_counts.items()}
        
        print(f"   📈 Class Distribution (from {sample_size} samples):")
        for cls in sorted(class_percentages.keys()):
            print(f"     Class {cls}: {class_percentages[cls]:.2f}% ({class_counts[cls]:,} pixels)")
        
        return {
            'class_counts': dict(class_counts),
            'class_percentages': class_percentages,
            'total_pixels': total_pixels,
            'sample_size': sample_size
        }
    
    def visualize_predictions(self, model: torch.nn.Module, dataset, dataset_name: str, 
                            num_samples: int = 8) -> None:
        """Visualize model predictions vs ground truth."""
        print(f"\n🎨 Visualizing Predictions for {dataset_name}")
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 3))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        class_colors = ['black', 'red', 'green', 'blue']  # Background, Unsafe, Safe, Optimal
        
        sample_count = 0
        with torch.no_grad():
            for batch in dataloader:
                if sample_count >= num_samples:
                    break
                
                try:
                    image = batch['image'].to(self.device)
                    mask_true = batch['mask']
                    
                    # Get prediction
                    output = model(image)
                    if isinstance(output, dict):
                        logits = output['main']
                    else:
                        logits = output
                    
                    # Convert to prediction
                    pred_probs = F.softmax(logits, dim=1)
                    pred_mask = torch.argmax(logits, dim=1)
                    
                    # Move to CPU for visualization
                    image_np = image[0].cpu().numpy()
                    mask_true_np = mask_true[0].numpy()
                    pred_mask_np = pred_mask[0].cpu().numpy()
                    pred_probs_np = pred_probs[0].cpu().numpy()
                    
                    # Display image (first 3 channels only)
                    if image_np.shape[0] >= 3:
                        image_display = np.transpose(image_np[:3], (1, 2, 0))
                        # Normalize for display
                        image_display = (image_display - image_display.min()) / (image_display.max() - image_display.min())
                    else:
                        image_display = np.zeros((image_np.shape[1], image_np.shape[2], 3))
                    
                    # Plot
                    axes[sample_count, 0].imshow(image_display)
                    axes[sample_count, 0].set_title(f'Input Image')
                    axes[sample_count, 0].axis('off')
                    
                    # Ground truth
                    axes[sample_count, 1].imshow(mask_true_np, cmap='tab10', vmin=0, vmax=3)
                    axes[sample_count, 1].set_title(f'Ground Truth')
                    axes[sample_count, 1].axis('off')
                    
                    # Prediction
                    axes[sample_count, 2].imshow(pred_mask_np, cmap='tab10', vmin=0, vmax=3)
                    axes[sample_count, 2].set_title(f'Prediction\nMax Prob: {pred_probs_np.max():.3f}')
                    axes[sample_count, 2].axis('off')
                    
                    # Print prediction statistics
                    unique_pred, counts_pred = np.unique(pred_mask_np, return_counts=True)
                    unique_true, counts_true = np.unique(mask_true_np, return_counts=True)
                    
                    print(f"   Sample {sample_count + 1}:")
                    print(f"     True classes: {dict(zip(unique_true, counts_true))}")
                    print(f"     Pred classes: {dict(zip(unique_pred, counts_pred))}")
                    print(f"     Max confidence: {pred_probs_np.max():.3f}")
                    print(f"     Min confidence: {pred_probs_np.min():.3f}")
                    
                    sample_count += 1
                    
                except Exception as e:
                    print(f"   ⚠️ Error visualizing sample: {e}")
                    continue
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{dataset_name}_predictions.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved visualization: {self.output_dir / f'{dataset_name}_predictions.png'}")
    
    def analyze_loss_components(self, model: torch.nn.Module, dataset, dataset_name: str) -> Dict[str, float]:
        """Analyze different components of the loss."""
        print(f"\n🔍 Analyzing Loss Components for {dataset_name}")
        
        from losses.safety_aware_losses import CombinedSafetyLoss
        criterion = CombinedSafetyLoss(num_classes=4)
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        
        total_losses = {
            'total_loss': 0.0,
            'ce_loss': 0.0,
            'focal_loss': 0.0,
            'boundary_loss': 0.0,
            'safety_loss': 0.0
        }
        
        sample_count = 0
        max_samples = 20  # Limit to avoid long analysis
        
        with torch.no_grad():
            for batch in dataloader:
                if sample_count >= max_samples:
                    break
                
                try:
                    images = batch['image'].to(self.device)
                    masks = batch['mask'].to(self.device)
                    
                    outputs = model(images)
                    loss_dict = criterion(outputs, masks)
                    
                    # Accumulate losses
                    for key in total_losses.keys():
                        if key in loss_dict:
                            total_losses[key] += loss_dict[key].item()
                    
                    sample_count += 1
                    
                except Exception as e:
                    print(f"   ⚠️ Error analyzing batch: {e}")
                    continue
        
        # Average losses
        avg_losses = {key: total_loss / sample_count for key, total_loss in total_losses.items()}
        
        print(f"   📊 Average Loss Components (over {sample_count} batches):")
        for key, value in avg_losses.items():
            print(f"     {key}: {value:.4f}")
        
        return avg_losses
    
    def check_data_loading(self, dataset_name: str, dataset) -> bool:
        """Check if data loading is working correctly."""
        print(f"\n🔍 Checking Data Loading for {dataset_name}")
        
        try:
            # Test basic loading
            sample = dataset[0]
            print(f"   ✅ Basic loading works")
            print(f"   📊 Sample keys: {list(sample.keys()) if isinstance(sample, dict) else 'Not a dict'}")
            
            if isinstance(sample, dict):
                if 'image' in sample and 'mask' in sample:
                    image = sample['image']
                    mask = sample['mask']
                    
                    print(f"   🖼️ Image shape: {image.shape if hasattr(image, 'shape') else 'No shape'}")
                    print(f"   🎯 Mask shape: {mask.shape if hasattr(mask, 'shape') else 'No shape'}")
                    
                    if hasattr(mask, 'unique'):
                        unique_classes = mask.unique() if hasattr(mask, 'unique') else np.unique(mask)
                        print(f"   🏷️ Unique classes in mask: {unique_classes}")
                    
                    return True
                else:
                    print(f"   ❌ Missing 'image' or 'mask' keys")
                    return False
            else:
                print(f"   ❌ Sample is not a dictionary")
                return False
                
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            return False
    
    def run_full_diagnostic(self):
        """Run complete diagnostic analysis."""
        print("\n🚀 Starting Full Diagnostic Analysis")
        print("=" * 50)
        
        # Dataset configurations
        datasets_config = {
            'DroneDeploy': {
                'path': '../datasets/drone_deploy_dataset_intermediate/dataset-medium',
                'class': DroneDeployDataset,
                'transform': create_drone_deploy_transforms(is_training=False),
                'channels': 4,
                'model_path': 'outputs/complete_test/adaptive_stage_1_dronedeploy_best.pth'
            },
            'UDD': {
                'path': '../datasets/UDD/UDD/UDD5',
                'class': UDDDataset,
                'transform': create_udd_transforms(is_training=False),
                'channels': 3,
                'model_path': 'outputs/complete_test/adaptive_stage_2_udd_best.pth'
            },
            'Semantic Drone': {
                'path': '../datasets/Aerial_Semantic_Segmentation_Drone_Dataset/dataset/semantic_drone_dataset',
                'class': SemanticDroneDataset,
                'transform': create_semantic_drone_transforms(is_training=False),
                'channels': 3,
                'model_path': 'outputs/complete_test/adaptive_stage_3_semantic_drone_best.pth'
            }
        }
        
        results = {}
        
        for dataset_name, config in datasets_config.items():
            print(f"\n{'='*20} {dataset_name} Analysis {'='*20}")
            
            try:
                # Create dataset
                if dataset_name == 'DroneDeploy':
                    dataset = config['class'](
                        data_root=config['path'],
                        split='val',
                        transform=config['transform'],
                        use_height=True
                    )
                else:
                    dataset = config['class'](
                        data_root=config['path'],
                        split='val',
                        transform=config['transform']
                    )
                
                # Check data loading
                data_loading_ok = self.check_data_loading(dataset_name, dataset)
                
                if data_loading_ok:
                    # Analyze dataset distribution
                    distribution = self.analyze_dataset_distribution(dataset_name, dataset)
                    
                    # Load and test model if it exists
                    model_path = Path(config['model_path'])
                    if model_path.exists():
                        model, checkpoint = self.load_model(str(model_path), config['channels'])
                        
                        # Visualize predictions
                        self.visualize_predictions(model, dataset, dataset_name)
                        
                        # Analyze loss components
                        loss_analysis = self.analyze_loss_components(model, dataset, dataset_name)
                        
                        results[dataset_name] = {
                            'data_loading': True,
                            'distribution': distribution,
                            'loss_analysis': loss_analysis,
                            'checkpoint_metrics': checkpoint.get('metrics', {})
                        }
                    else:
                        print(f"   ⚠️ Model checkpoint not found: {model_path}")
                        results[dataset_name] = {
                            'data_loading': True,
                            'distribution': distribution,
                            'model_available': False
                        }
                else:
                    results[dataset_name] = {'data_loading': False}
                    
            except Exception as e:
                print(f"   ❌ Failed to analyze {dataset_name}: {e}")
                results[dataset_name] = {'error': str(e)}
        
        # Summary report
        self.generate_summary_report(results)
        
        return results
    
    def generate_summary_report(self, results: Dict[str, Any]):
        """Generate a summary report of the diagnostic analysis."""
        print(f"\n📋 DIAGNOSTIC SUMMARY REPORT")
        print("=" * 60)
        
        report_lines = ["# Training Failure Diagnostic Report\n"]
        
        for dataset_name, result in results.items():
            report_lines.append(f"## {dataset_name} Dataset\n")
            
            if 'error' in result:
                report_lines.append(f"❌ **Error:** {result['error']}\n")
                continue
            
            if not result.get('data_loading', False):
                report_lines.append("❌ **Data Loading Failed**\n")
                continue
            
            report_lines.append("✅ **Data Loading:** OK\n")
            
            # Distribution analysis
            if 'distribution' in result:
                dist = result['distribution']
                report_lines.append("### Class Distribution:\n")
                for cls, pct in sorted(dist['class_percentages'].items()):
                    report_lines.append(f"- Class {cls}: {pct:.2f}%\n")
                report_lines.append("\n")
            
            # Loss analysis
            if 'loss_analysis' in result:
                loss = result['loss_analysis']
                report_lines.append("### Loss Analysis:\n")
                for loss_type, value in loss.items():
                    report_lines.append(f"- {loss_type}: {value:.4f}\n")
                report_lines.append("\n")
            
            # Checkpoint metrics
            if 'checkpoint_metrics' in result:
                metrics = result['checkpoint_metrics']
                report_lines.append("### Model Performance:\n")
                if 'miou' in metrics:
                    report_lines.append(f"- Mean IoU: {metrics['miou']:.4f}\n")
                if 'overall_accuracy' in metrics:
                    report_lines.append(f"- Overall Accuracy: {metrics['overall_accuracy']:.4f}\n")
                report_lines.append("\n")
        
        # Save report (fix Unicode encoding issue)
        report_path = self.output_dir / "diagnostic_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
        
        print(f"📄 Full report saved: {report_path}")
        
        # Print key findings
        print("\n🔍 KEY FINDINGS:")
        for dataset_name, result in results.items():
            if 'loss_analysis' in result:
                total_loss = result['loss_analysis'].get('total_loss', 0)
                miou = result.get('checkpoint_metrics', {}).get('miou', 0)
                print(f"   {dataset_name}: Loss={total_loss:.2f}, IoU={miou:.4f}")


def main():
    diagnostic = TrainingDiagnostic({})
    results = diagnostic.run_full_diagnostic()
    
    print("\n🎯 NEXT STEPS RECOMMENDED:")
    print("1. Review prediction visualizations in outputs/diagnostics/")
    print("2. Check class distributions for severe imbalances")
    print("3. Verify that models are not predicting single classes")
    print("4. Consider training individual datasets separately")


if __name__ == "__main__":
    main() 