#!/usr/bin/env python3
"""
UAV Landing System - Model Testing and Evaluation
================================================

Comprehensive testing script for evaluating trained models:
- Load and test any checkpoint
- Compute detailed metrics (mIoU, per-class IoU, accuracy, etc.)
- Generate confusion matrices and visualizations
- Test on multiple datasets
- Export results to various formats

Usage:
    python test.py --checkpoint outputs/stage2_best.pth --dataset dronedeploy
    python test.py --checkpoint outputs/stage3_best.pth --test_all_datasets
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
import cv2

# Import our components
from models.mobilenetv3_edge_model import create_edge_model
from datasets.semantic_drone_dataset import SemanticDroneDataset, create_semantic_drone_transforms
from datasets.dronedeploy_1024_dataset import DroneDeploy1024Dataset, create_dronedeploy_datasets
from datasets.udd6_dataset import UDD6Dataset, create_udd6_transforms


class ModelEvaluator:
    """Comprehensive model evaluation with detailed metrics and visualizations."""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """Initialize evaluator with model checkpoint."""
        
        # Auto-detect device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Load checkpoint
        print(f"📁 Loading checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract checkpoint info
        self.checkpoint_info = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'stage': checkpoint.get('stage', 'unknown'),
            'metrics': checkpoint.get('metrics', {}),
            'timestamp': checkpoint.get('timestamp', 'unknown')
        }
        
        # Create and load model
        self.model = create_edge_model(
            model_type='enhanced',
            num_classes=6,
            use_uncertainty=True,
            pretrained=False
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Class information
        self.class_names = ['ground', 'vegetation', 'obstacle', 'water', 'vehicle', 'other']
        self.class_colors = [
            [128, 64, 128],    # ground - purple
            [107, 142, 35],    # vegetation - olive
            [70, 70, 70],      # obstacle - dark gray
            [0, 0, 142],       # water - dark blue
            [0, 0, 70],        # vehicle - dark red
            [102, 102, 156]    # other - light purple
        ]
        
        print(f"✅ Model loaded successfully")
        print(f"   Device: {self.device}")
        print(f"   Checkpoint epoch: {self.checkpoint_info['epoch']}")
        print(f"   Checkpoint stage: {self.checkpoint_info['stage']}")
        if self.checkpoint_info['metrics']:
            print(f"   Checkpoint mIoU: {self.checkpoint_info['metrics'].get('miou', 'N/A')}")
    
    def evaluate_dataset(
        self, 
        dataset_name: str,
        dataset_path: str,
        split: str = 'val',
        save_predictions: bool = False,
        max_samples: Optional[int] = None
    ) -> Dict:
        """Evaluate model on a specific dataset."""
        
        print(f"\n🔍 Evaluating on {dataset_name} dataset")
        print(f"   Path: {dataset_path}")
        print(f"   Split: {split}")
        
        # Create dataset
        dataset = self._create_dataset(dataset_name, dataset_path, split)
        if dataset is None:
            return {}
        
        # Limit samples if requested
        if max_samples and len(dataset) > max_samples:
            print(f"   Limiting to {max_samples} samples (out of {len(dataset)})")
            # Use a subset
            indices = np.random.choice(len(dataset), max_samples, replace=False)
            dataset = torch.utils.data.Subset(dataset, indices)
        
        print(f"   Samples: {len(dataset)}")
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,  # Fixed batch size for testing
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Initialize metrics
        all_predictions = []
        all_targets = []
        all_confidences = []
        inference_times = []
        
        # Class-wise metrics
        intersection = torch.zeros(6, device=self.device)
        union = torch.zeros(6, device=self.device)
        class_correct = torch.zeros(6, device=self.device)
        class_total = torch.zeros(6, device=self.device)
        
        total_correct = 0
        total_pixels = 0
        
        print(f"🚀 Running inference...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc='Testing')):
                images = batch['image'].to(self.device, non_blocking=True)
                targets = batch['mask'].to(self.device, non_blocking=True).long()
                
                # Measure inference time
                start_time = time.time()
                
                # Forward pass
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    predictions = outputs['main']
                    confidence = outputs.get('uncertainty', None)
                else:
                    predictions = outputs
                    confidence = None
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Get predicted classes and probabilities
                pred_probs = F.softmax(predictions, dim=1)
                pred_classes = predictions.argmax(dim=1)
                
                # Store for detailed analysis
                all_predictions.extend(pred_classes.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
                
                if confidence is not None:
                    all_confidences.extend(confidence.cpu().numpy().flatten())
                
                # Compute metrics
                correct = (pred_classes == targets).sum()
                total_correct += correct.item()
                total_pixels += targets.numel()
                
                # Per-class metrics
                for class_id in range(6):
                    pred_mask = (pred_classes == class_id)
                    target_mask = (targets == class_id)
                    
                    # IoU computation
                    intersection[class_id] += (pred_mask & target_mask).sum().float()
                    union[class_id] += (pred_mask | target_mask).sum().float()
                    
                    # Class accuracy
                    if target_mask.sum() > 0:
                        class_correct[class_id] += (pred_classes[target_mask] == class_id).sum().float()
                        class_total[class_id] += target_mask.sum().float()
                
                # Save predictions if requested
                if save_predictions and batch_idx < 10:  # Save first 10 batches
                    self._save_prediction_visualization(
                        images, targets, pred_classes, batch_idx, dataset_name
                    )
        
        # Compute final metrics
        overall_accuracy = total_correct / total_pixels
        iou_per_class = intersection / (union + 1e-8)
        miou = iou_per_class.mean().item()
        class_accuracy = class_correct / (class_total + 1e-8)
        
        # Inference statistics
        avg_inference_time = np.mean(inference_times)
        fps = len(dataloader.dataset) / sum(inference_times)
        
        # Create confusion matrix
        cm = confusion_matrix(all_targets, all_predictions, labels=list(range(6)))
        
        # Detailed metrics
        results = {
            'dataset': dataset_name,
            'split': split,
            'num_samples': len(dataset),
            'overall_accuracy': float(overall_accuracy),
            'mean_iou': float(miou),
            'mean_class_accuracy': float(class_accuracy.mean()),
            'per_class_iou': {
                self.class_names[i]: float(iou_per_class[i]) 
                for i in range(6)
            },
            'per_class_accuracy': {
                self.class_names[i]: float(class_accuracy[i]) 
                for i in range(6)
            },
            'confusion_matrix': cm.tolist(),
            'inference_time': {
                'avg_time_per_batch': float(avg_inference_time),
                'fps': float(fps),
                'total_time': float(sum(inference_times))
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Print summary
        print(f"\n📊 Results for {dataset_name}:")
        print(f"   Overall Accuracy: {overall_accuracy:.4f}")
        print(f"   Mean IoU: {miou:.4f}")
        print(f"   Mean Class Accuracy: {class_accuracy.mean():.4f}")
        print(f"   Inference Speed: {fps:.1f} FPS")
        
        print(f"\n📋 Per-class IoU:")
        for i, class_name in enumerate(self.class_names):
            print(f"   {class_name}: {iou_per_class[i]:.4f}")
        
        return results
    
    def _create_dataset(self, dataset_name: str, dataset_path: str, split: str):
        """Create dataset based on name."""
        
        try:
            if dataset_name.lower() in ['sdd', 'semantic_drone']:
                transforms = create_semantic_drone_transforms(
                    input_size=(512, 512),
                    is_training=False
                )
                return SemanticDroneDataset(
                    data_root=dataset_path,
                    split=split,
                    transform=transforms,
                    class_mapping="advanced_6_class"
                )
            
            elif dataset_name.lower() in ['dronedeploy', 'dd']:
                datasets = create_dronedeploy_datasets(
                    data_root=dataset_path,
                    patch_size=512,
                    augmentation=False
                )
                return datasets.get(split, datasets['val'])
            
            elif dataset_name.lower() in ['udd6', 'udd']:
                transforms = create_udd6_transforms(is_training=False)
                return UDD6Dataset(
                    data_root=dataset_path,
                    split=split,
                    transform=transforms
                )
            
            else:
                print(f"❌ Unknown dataset: {dataset_name}")
                return None
                
        except Exception as e:
            print(f"❌ Failed to load {dataset_name} dataset: {e}")
            return None
    
    def _save_prediction_visualization(
        self, 
        images: torch.Tensor, 
        targets: torch.Tensor, 
        predictions: torch.Tensor, 
        batch_idx: int,
        dataset_name: str
    ):
        """Save visualization of predictions vs ground truth."""
        
        save_dir = Path(f'test_results/visualizations/{dataset_name}')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Process first image in batch
        image = images[0].cpu().numpy().transpose(1, 2, 0)
        target = targets[0].cpu().numpy()
        prediction = predictions[0].cpu().numpy()
        
        # Denormalize image (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        # Create colored masks
        target_colored = self._colorize_mask(target)
        prediction_colored = self._colorize_mask(prediction)
        
        # Create comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        axes[1].imshow(target_colored)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(prediction_colored)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'batch_{batch_idx:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert class mask to colored visualization."""
        colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        for class_id in range(6):
            mask_indices = mask == class_id
            colored[mask_indices] = self.class_colors[class_id]
        
        return colored
    
    def generate_confusion_matrix_plot(self, results: Dict, save_path: str = None):
        """Generate and save confusion matrix visualization."""
        
        cm = np.array(results['confusion_matrix'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(f'Confusion Matrix - {results["dataset"]}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Confusion matrix saved: {save_path}")
        
        plt.show()
    
    def save_results(self, results: Dict, output_path: str):
        """Save detailed results to JSON file."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add checkpoint info to results
        results['checkpoint_info'] = self.checkpoint_info
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved: {output_path}")
    
    def compare_models(self, other_results: List[Dict]) -> Dict:
        """Compare this model with other model results."""
        
        comparison = {
            'models': [],
            'best_miou': 0.0,
            'best_model': None
        }
        
        for result in other_results:
            model_info = {
                'checkpoint': result.get('checkpoint_info', {}),
                'miou': result.get('mean_iou', 0.0),
                'accuracy': result.get('overall_accuracy', 0.0),
                'per_class_iou': result.get('per_class_iou', {})
            }
            comparison['models'].append(model_info)
            
            if model_info['miou'] > comparison['best_miou']:
                comparison['best_miou'] = model_info['miou']
                comparison['best_model'] = model_info
        
        return comparison


def main():
    parser = argparse.ArgumentParser(description='UAV Landing System - Model Testing')
    
    # Model and testing configuration
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='dronedeploy',
                        choices=['sdd', 'semantic_drone', 'dronedeploy', 'dd', 'udd6', 'udd'],
                        help='Dataset to test on')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to dataset (auto-detected if not provided)')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split to test on')
    
    # Testing options
    parser.add_argument('--test_all_datasets', action='store_true',
                        help='Test on all available datasets')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to test (for quick evaluation)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save prediction visualizations')
    parser.add_argument('--save_confusion_matrix', action='store_true',
                        help='Save confusion matrix plot')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    print("🔬 UAV Landing System - Model Testing")
    print("===================================")
    
    # Create evaluator
    evaluator = ModelEvaluator(args.checkpoint, args.device)
    
    # Auto-detect dataset paths if not provided
    dataset_paths = {
        'sdd': args.dataset_path or './datasets/semantic_drone_dataset',
        'semantic_drone': args.dataset_path or './datasets/semantic_drone_dataset',
        'dronedeploy': args.dataset_path or './datasets/drone_deploy_dataset',
        'dd': args.dataset_path or './datasets/drone_deploy_dataset',
        'udd6': args.dataset_path or './datasets/udd6_dataset',
        'udd': args.dataset_path or './datasets/udd6_dataset'
    }
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    try:
        if args.test_all_datasets:
            # Test on all datasets
            for dataset_name, dataset_path in dataset_paths.items():
                if dataset_name in ['semantic_drone', 'dd', 'udd']:  # Skip duplicates
                    continue
                
                if Path(dataset_path).exists():
                    results = evaluator.evaluate_dataset(
                        dataset_name=dataset_name,
                        dataset_path=dataset_path,
                        split=args.split,
                        save_predictions=args.save_predictions,
                        max_samples=args.max_samples
                    )
                    
                    if results:
                        all_results.append(results)
                        
                        # Save individual results
                        result_file = output_dir / f'results_{dataset_name}_{args.split}.json'
                        evaluator.save_results(results, result_file)
                        
                        # Save confusion matrix if requested
                        if args.save_confusion_matrix:
                            cm_file = output_dir / f'confusion_matrix_{dataset_name}_{args.split}.png'
                            evaluator.generate_confusion_matrix_plot(results, cm_file)
                else:
                    print(f"⚠️  Dataset path not found: {dataset_path}")
        
        else:
            # Test on single dataset
            dataset_path = dataset_paths.get(args.dataset, args.dataset_path)
            
            if not dataset_path or not Path(dataset_path).exists():
                print(f"❌ Dataset path not found: {dataset_path}")
                return
            
            results = evaluator.evaluate_dataset(
                dataset_name=args.dataset,
                dataset_path=dataset_path,
                split=args.split,
                save_predictions=args.save_predictions,
                max_samples=args.max_samples
            )
            
            if results:
                all_results.append(results)
                
                # Save results
                result_file = output_dir / f'results_{args.dataset}_{args.split}.json'
                evaluator.save_results(results, result_file)
                
                # Save confusion matrix if requested
                if args.save_confusion_matrix:
                    cm_file = output_dir / f'confusion_matrix_{args.dataset}_{args.split}.png'
                    evaluator.generate_confusion_matrix_plot(results, cm_file)
        
        # Save summary results
        if all_results:
            summary = {
                'checkpoint': args.checkpoint,
                'timestamp': datetime.now().isoformat(),
                'results': all_results,
                'best_result': max(all_results, key=lambda x: x['mean_iou'])
            }
            
            summary_file = output_dir / 'test_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n🎉 Testing completed!")
            print(f"   Best mIoU: {summary['best_result']['mean_iou']:.4f} on {summary['best_result']['dataset']}")
            print(f"   Results saved in: {output_dir}")
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 