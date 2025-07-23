#!/usr/bin/env python3
"""
Safety-Aware Evaluation Metrics for UAV Landing Detection
=========================================================

Professional evaluation framework addressing safety-critical requirements:
- Safety-weighted IoU and accuracy metrics
- Uncertainty quality assessment
- Boundary precision evaluation
- Cross-domain generalization metrics
- Critical failure detection and analysis
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class SafetyAwareEvaluator:
    """
    Comprehensive safety-aware evaluation for UAV landing detection.
    
    Beyond standard segmentation metrics, incorporates:
    - Safety-critical error analysis
    - Uncertainty calibration assessment
    - Boundary quality evaluation
    - Cross-domain robustness testing
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        class_names: Optional[List[str]] = None,
        safety_weights: Optional[List[float]] = None,
        uncertainty_threshold: float = 0.5
    ):
        """
        Initialize safety evaluator.
        
        Args:
            num_classes: Number of segmentation classes
            class_names: Names of classes for reporting
            safety_weights: Safety importance weights for each class
            uncertainty_threshold: Threshold for high uncertainty regions
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.uncertainty_threshold = uncertainty_threshold
        
        # Safety weights: higher values indicate more safety-critical classes
        if safety_weights is None:
            safety_weights = [1.0, 3.0, 2.0, 5.0]  # [background, safe, caution, danger]
        self.safety_weights = np.array(safety_weights)
        
        # Initialize accumulators
        self.reset()
        
    def reset(self):
        """Reset all evaluation metrics."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.predictions_list = []
        self.targets_list = []
        self.uncertainties_list = []
        self.total_pixels = 0
        
        # Safety-specific metrics
        self.critical_errors = []
        self.boundary_errors = []
        self.uncertainty_errors = []
        
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None
    ):
        """
        Update evaluation metrics with new batch.
        
        Args:
            predictions: Predicted class labels [B, H, W]
            targets: Ground truth labels [B, H, W]
            uncertainties: Uncertainty estimates [B, H, W] (optional)
        """
        # Ensure predictions and targets have compatible shapes
        if predictions.shape != targets.shape:
            print(f"⚠️  Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
            # Take the minimum dimensions to avoid indexing errors
            min_h = min(predictions.shape[-2], targets.shape[-2])
            min_w = min(predictions.shape[-1], targets.shape[-1])
            predictions = predictions[..., :min_h, :min_w]
            targets = targets[..., :min_h, :min_w]
        
        # Convert to numpy
        pred_np = predictions.cpu().numpy().flatten()
        target_np = targets.cpu().numpy().flatten()
        
        # Update confusion matrix
        valid_mask = (target_np >= 0) & (target_np < self.num_classes)
        if valid_mask.sum() > 0:
            cm = confusion_matrix(
                target_np[valid_mask],
                pred_np[valid_mask],
                labels=list(range(self.num_classes))
            )
            self.confusion_matrix += cm
            self.total_pixels += valid_mask.sum()
        
        # Store for detailed analysis
        self.predictions_list.append(pred_np[valid_mask])
        self.targets_list.append(target_np[valid_mask])
        
        if uncertainties is not None:
            unc_np = uncertainties.cpu().numpy().flatten()
            self.uncertainties_list.append(unc_np[valid_mask])
        
        # Analyze critical errors
        self._analyze_critical_errors(pred_np[valid_mask], target_np[valid_mask])
        
        # Analyze boundary errors if high resolution
        if predictions.shape[-1] >= 256:  # Only for high-res predictions
            self._analyze_boundary_errors(predictions, targets)
    
    def _analyze_critical_errors(self, predictions: np.ndarray, targets: np.ndarray):
        """Analyze safety-critical misclassifications."""
        
        # Critical error: predicting safe when actually dangerous
        safe_pred_danger_true = (predictions == 1) & (targets == 3)  # safe->danger
        caution_pred_danger_true = (predictions == 2) & (targets == 3)  # caution->danger
        safe_pred_caution_true = (predictions == 1) & (targets == 2)  # safe->caution
        
        critical_error_count = (
            safe_pred_danger_true.sum() * 5.0 +  # Most critical
            caution_pred_danger_true.sum() * 3.0 +  # Very critical
            safe_pred_caution_true.sum() * 2.0  # Moderately critical
        )
        
        self.critical_errors.append(critical_error_count)
    
    def _analyze_boundary_errors(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Analyze errors near class boundaries."""
        
        # Simple boundary detection using edge detection
        for i in range(predictions.shape[0]):
            pred_edges = self._detect_edges(predictions[i])
            target_edges = self._detect_edges(targets[i])
            
            # Count mismatches near boundaries
            boundary_mismatch = ((pred_edges > 0) | (target_edges > 0)) & (predictions[i] != targets[i])
            boundary_error_rate = boundary_mismatch.float().mean().item()
            
            self.boundary_errors.append(boundary_error_rate)
    
    def _detect_edges(self, tensor: torch.Tensor) -> torch.Tensor:
        """Simple edge detection using Sobel operator."""
        # Convert to float and add channel dimension
        x = tensor.float().unsqueeze(0).unsqueeze(0)
        
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(0)
        
        # Apply convolution
        grad_x = torch.nn.functional.conv2d(x, sobel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(x, sobel_y, padding=1)
        
        # Compute magnitude
        magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        return magnitude.squeeze()
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""
        
        if self.total_pixels == 0:
            return {'error': 'No valid pixels processed'}
        
        metrics = {}
        
        # Standard segmentation metrics
        metrics.update(self._compute_standard_metrics())
        
        # Safety-aware metrics
        metrics.update(self._compute_safety_metrics())
        
        # Uncertainty metrics
        if self.uncertainties_list:
            metrics.update(self._compute_uncertainty_metrics())
        
        # Boundary metrics
        if self.boundary_errors:
            metrics.update(self._compute_boundary_metrics())
        
        return metrics
    
    def _compute_standard_metrics(self) -> Dict[str, float]:
        """Compute standard segmentation metrics."""
        
        cm = self.confusion_matrix
        metrics = {}
        
        # Overall accuracy
        overall_accuracy = np.diag(cm).sum() / cm.sum()
        metrics['overall_accuracy'] = overall_accuracy
        
        # Per-class metrics
        class_accuracies = []
        class_ious = []
        class_precisions = []
        class_recalls = []
        
        for i in range(self.num_classes):
            # True positives, false positives, false negatives
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            # Accuracy for this class
            class_acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            class_accuracies.append(class_acc)
            
            # IoU for this class
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            class_ious.append(iou)
            
            # Precision and Recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            class_precisions.append(precision)
            class_recalls.append(recall)
            
            # Per-class metrics
            metrics[f'{self.class_names[i]}_accuracy'] = class_acc
            metrics[f'{self.class_names[i]}_iou'] = iou
            metrics[f'{self.class_names[i]}_precision'] = precision
            metrics[f'{self.class_names[i]}_recall'] = recall
        
        # Mean metrics
        metrics['mean_accuracy'] = np.mean(class_accuracies)
        metrics['miou'] = np.mean(class_ious)
        metrics['mean_precision'] = np.mean(class_precisions)
        metrics['mean_recall'] = np.mean(class_recalls)
        
        return metrics
    
    def _compute_safety_metrics(self) -> Dict[str, float]:
        """Compute safety-aware metrics."""
        
        cm = self.confusion_matrix
        metrics = {}
        
        # Safety-weighted accuracy
        safety_weighted_correct = 0
        safety_weighted_total = 0
        
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                count = cm[i, j]
                weight = self.safety_weights[i]
                
                if i == j:  # Correct prediction
                    safety_weighted_correct += count * weight
                safety_weighted_total += count * weight
        
        safety_accuracy = safety_weighted_correct / safety_weighted_total if safety_weighted_total > 0 else 0.0
        metrics['safety_weighted_accuracy'] = safety_accuracy
        
        # Critical error rate
        total_critical_errors = sum(self.critical_errors) if self.critical_errors else 0
        critical_error_rate = total_critical_errors / self.total_pixels if self.total_pixels > 0 else 0.0
        metrics['critical_error_rate'] = critical_error_rate
        
        # Safety score (higher is better)
        safety_score = safety_accuracy * (1.0 - critical_error_rate)
        metrics['safety_score'] = safety_score
        
        # Conservative prediction rate (predicting danger when uncertain)
        conservative_predictions = 0
        total_predictions = 0
        
        for predictions, targets in zip(self.predictions_list, self.targets_list):
            # Count cases where model predicts more conservative class
            conservative_mask = predictions > targets  # Higher class index = more conservative
            conservative_predictions += conservative_mask.sum()
            total_predictions += len(predictions)
        
        conservative_rate = conservative_predictions / total_predictions if total_predictions > 0 else 0.0
        metrics['conservative_prediction_rate'] = conservative_rate
        
        return metrics
    
    def _compute_uncertainty_metrics(self) -> Dict[str, float]:
        """Compute uncertainty calibration metrics."""
        
        if not self.uncertainties_list:
            return {}
        
        metrics = {}
        
        # Combine all uncertainty and error data
        all_uncertainties = np.concatenate(self.uncertainties_list)
        all_predictions = np.concatenate(self.predictions_list)
        all_targets = np.concatenate(self.targets_list)
        
        # Prediction errors
        prediction_errors = (all_predictions != all_targets).astype(float)
        
        # Uncertainty calibration
        # High uncertainty should correlate with high error rate
        high_uncertainty_mask = all_uncertainties > self.uncertainty_threshold
        low_uncertainty_mask = all_uncertainties <= self.uncertainty_threshold
        
        if high_uncertainty_mask.sum() > 0:
            high_unc_error_rate = prediction_errors[high_uncertainty_mask].mean()
            metrics['high_uncertainty_error_rate'] = high_unc_error_rate
        
        if low_uncertainty_mask.sum() > 0:
            low_unc_error_rate = prediction_errors[low_uncertainty_mask].mean()
            metrics['low_uncertainty_error_rate'] = low_unc_error_rate
        
        # Uncertainty quality score
        # Good uncertainty should have high error rate in high uncertainty regions
        # and low error rate in low uncertainty regions
        if 'high_uncertainty_error_rate' in metrics and 'low_uncertainty_error_rate' in metrics:
            uncertainty_separation = metrics['high_uncertainty_error_rate'] - metrics['low_uncertainty_error_rate']
            metrics['uncertainty_quality'] = max(0.0, uncertainty_separation)
        
        # Average uncertainty
        metrics['mean_uncertainty'] = all_uncertainties.mean()
        
        return metrics
    
    def _compute_boundary_metrics(self) -> Dict[str, float]:
        """Compute boundary precision metrics."""
        
        if not self.boundary_errors:
            return {}
        
        metrics = {}
        
        # Average boundary error rate
        avg_boundary_error = np.mean(self.boundary_errors)
        metrics['boundary_error_rate'] = avg_boundary_error
        
        # Boundary precision (inverse of error rate)
        boundary_precision = 1.0 - avg_boundary_error
        metrics['boundary_precision'] = boundary_precision
        
        return metrics
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None, normalize: bool = True):
        """Plot confusion matrix with safety annotations."""
        
        cm = self.confusion_matrix.astype(float)
        if normalize:
            cm = cm / cm.sum(axis=1, keepdims=True)
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap with custom colormap for safety
        sns.heatmap(
            cm,
            annot=True,
            fmt='.3f' if normalize else 'd',
            cmap='Reds',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Probability' if normalize else 'Count'}
        )
        
        plt.title('Safety-Aware Confusion Matrix')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        
        # Add safety annotations
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j:  # Misclassification
                    # Determine safety criticality
                    if i == 3 and j == 1:  # danger -> safe
                        plt.text(j + 0.5, i + 0.7, '⚠️ CRITICAL', 
                                ha='center', va='center', color='red', fontweight='bold')
                    elif i == 2 and j == 1:  # caution -> safe
                        plt.text(j + 0.5, i + 0.7, '⚠️ HIGH', 
                                ha='center', va='center', color='orange', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def generate_safety_report(self) -> str:
        """Generate comprehensive safety evaluation report."""
        
        metrics = self.compute_metrics()
        
        report = """
═══════════════════════════════════════════════
    UAV LANDING SAFETY EVALUATION REPORT
═══════════════════════════════════════════════

OVERALL PERFORMANCE:
────────────────────
• Overall Accuracy: {:.3f}
• Mean IoU: {:.3f}
• Safety Score: {:.3f}

SAFETY-CRITICAL METRICS:
────────────────────────
• Critical Error Rate: {:.6f}
• Safety-Weighted Accuracy: {:.3f}
• Conservative Prediction Rate: {:.3f}

CLASS-SPECIFIC PERFORMANCE:
───────────────────────────
""".format(
            metrics.get('overall_accuracy', 0.0),
            metrics.get('miou', 0.0),
            metrics.get('safety_score', 0.0),
            metrics.get('critical_error_rate', 0.0),
            metrics.get('safety_weighted_accuracy', 0.0),
            metrics.get('conservative_prediction_rate', 0.0)
        )
        
        # Add per-class metrics
        for i, class_name in enumerate(self.class_names):
            iou = metrics.get(f'{class_name}_iou', 0.0)
            precision = metrics.get(f'{class_name}_precision', 0.0)
            recall = metrics.get(f'{class_name}_recall', 0.0)
            
            report += f"• {class_name.capitalize()}: IoU={iou:.3f}, Precision={precision:.3f}, Recall={recall:.3f}\n"
        
        # Add uncertainty metrics if available
        if 'uncertainty_quality' in metrics:
            report += f"""
UNCERTAINTY ASSESSMENT:
──────────────────────
• Uncertainty Quality: {metrics['uncertainty_quality']:.3f}
• High Uncertainty Error Rate: {metrics.get('high_uncertainty_error_rate', 0.0):.3f}
• Low Uncertainty Error Rate: {metrics.get('low_uncertainty_error_rate', 0.0):.3f}
• Mean Uncertainty: {metrics['mean_uncertainty']:.3f}
"""
        
        # Add boundary metrics if available
        if 'boundary_precision' in metrics:
            report += f"""
BOUNDARY PRECISION:
──────────────────
• Boundary Precision: {metrics['boundary_precision']:.3f}
• Boundary Error Rate: {metrics['boundary_error_rate']:.3f}
"""
        
        # Safety recommendations
        report += """
SAFETY RECOMMENDATIONS:
──────────────────────
"""
        
        critical_error_rate = metrics.get('critical_error_rate', 0.0)
        safety_score = metrics.get('safety_score', 0.0)
        
        if critical_error_rate > 0.01:
            report += "⚠️  HIGH CRITICAL ERROR RATE - Additional training recommended\n"
        if safety_score < 0.7:
            report += "⚠️  LOW SAFETY SCORE - Model not ready for deployment\n"
        if metrics.get('uncertainty_quality', 1.0) < 0.3:
            report += "⚠️  POOR UNCERTAINTY CALIBRATION - Uncertainty estimation needs improvement\n"
        
        if critical_error_rate <= 0.005 and safety_score >= 0.8:
            report += " SAFETY REQUIREMENTS MET - Model ready for careful deployment\n"
        
        report += "\n═══════════════════════════════════════════════\n"
        
        return report
    
    def save_evaluation_results(self, output_dir: str):
        """Save all evaluation results to directory."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics = self.compute_metrics()
        with open(output_path / 'metrics.json', 'w') as f:
            import json
            json.dump(metrics, f, indent=2)
        
        # Save confusion matrix plot
        fig = self.plot_confusion_matrix()
        fig.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save safety report
        report = self.generate_safety_report()
        with open(output_path / 'safety_report.txt', 'w') as f:
            f.write(report)
        
        print(f" Evaluation results saved to {output_path}")


if __name__ == "__main__":
    # Test the evaluator
    print("Testing Safety-Aware Evaluator...")
    
    # Create synthetic test data
    batch_size, height, width = 4, 128, 128
    num_classes = 4
    
    # Simulate predictions and targets
    predictions = torch.randint(0, num_classes, (batch_size, height, width))
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    uncertainties = torch.rand(batch_size, height, width)
    
    # Initialize evaluator
    evaluator = SafetyAwareEvaluator(
        num_classes=num_classes,
        class_names=['background', 'safe_landing', 'caution', 'danger']
    )
    
    # Update with test data
    evaluator.update(predictions, targets, uncertainties)
    
    # Compute metrics
    metrics = evaluator.compute_metrics()
    
    print(" Computed metrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")
    
    # Generate safety report
    report = evaluator.generate_safety_report()
    print("\n" + report)
    
    print(" Safety evaluator test completed!") 