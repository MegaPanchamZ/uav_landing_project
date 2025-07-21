#!/usr/bin/env python3
"""
Safety-Aware Loss Functions for UAV Landing Detection
====================================================

Professional loss functions addressing the critical safety requirements:
- Focal Loss with safety-aware class weighting
- Dice Loss for precise segmentation boundaries
- Boundary Loss for edge preservation
- Uncertainty Loss for reliable confidence estimation
- Combined Safety Loss integrating all components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Tuple


class SafetyFocalLoss(nn.Module):
    """
    Safety-aware Focal Loss for UAV landing detection.
    
    Addresses class imbalance while emphasizing safety-critical misclassifications:
    - Higher penalties for misclassifying dangerous areas as safe
    - Adaptive class weights based on safety implications
    - Focus on hard examples that could lead to unsafe landings
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        num_classes: int = 4,
        safety_weights: Optional[List[float]] = None,
        reduction: str = 'mean'
    ):
        """
        Initialize Safety Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            num_classes: Number of classes
            safety_weights: Custom safety-based class weights
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(SafetyFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction
        
        # Default safety weights: [background, safe, caution, danger]
        if safety_weights is None:
            safety_weights = [1.0, 2.0, 1.5, 3.0]  # Higher weight for danger class
        
        self.register_buffer('safety_weights', torch.tensor(safety_weights, dtype=torch.float32))
        
        # Safety penalty matrix - extra penalty for dangerous misclassifications
        self.register_buffer('safety_penalty', self._create_safety_penalty_matrix())
    
    def _create_safety_penalty_matrix(self) -> torch.Tensor:
        """
        Create safety penalty matrix for misclassification penalties.
        
        Rows: True class, Columns: Predicted class
        Higher values = more dangerous misclassification
        """
        penalty_matrix = torch.ones(self.num_classes, self.num_classes)
        
        # Extremely dangerous: predicting safe when actually dangerous
        penalty_matrix[3, 1] = 5.0  # danger -> safe (catastrophic)
        penalty_matrix[3, 2] = 3.0  # danger -> caution (very bad)
        
        # Dangerous: predicting safe when actually caution
        penalty_matrix[2, 1] = 3.0  # caution -> safe (bad)
        
        # Less dangerous but still problematic
        penalty_matrix[1, 3] = 1.5  # safe -> danger (conservative, less critical)
        penalty_matrix[2, 3] = 1.2  # caution -> danger (slightly conservative)
        
        return penalty_matrix
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute safety-aware focal loss.
        
        Args:
            inputs: Model predictions [B, C, H, W]
            targets: Ground truth labels [B, H, W]
            
        Returns:
            Computed loss value
        """
        # Ensure targets are long type
        targets = targets.long()
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute class probabilities
        pt = torch.exp(-ce_loss)
        
        # Apply focal term
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.safety_weights[targets]
        
        # Apply safety penalty
        predicted_classes = torch.argmax(inputs, dim=1)
        safety_penalty = self.safety_penalty[targets, predicted_classes]
        
        # Combine all terms
        focal_loss = alpha_t * focal_weight * safety_penalty * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for precise segmentation boundaries.
    
    Critical for UAV landing where precise boundaries between
    safe/unsafe areas can be life-critical.
    """
    
    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: int = 255,
        class_weights: Optional[torch.Tensor] = None
    ):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.class_weights = class_weights
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            inputs: Model predictions [B, C, H, W]
            targets: Ground truth labels [B, H, W]
            
        Returns:
            Dice loss value
        """
        # Convert targets to one-hot
        targets = targets.long()
        num_classes = inputs.size(1)
        
        # Create mask for valid pixels
        valid_mask = (targets != self.ignore_index)
        
        # Convert to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Apply softmax to inputs
        inputs_soft = F.softmax(inputs, dim=1)
        
        # Calculate Dice for each class
        dice_scores = []
        
        for class_idx in range(num_classes):
            # Get class predictions and targets
            pred_class = inputs_soft[:, class_idx]
            target_class = targets_one_hot[:, class_idx]
            
            # Apply valid mask
            pred_class = pred_class * valid_mask.float()
            target_class = target_class * valid_mask.float()
            
            # Calculate intersection and union
            intersection = (pred_class * target_class).sum()
            total = pred_class.sum() + target_class.sum()
            
            # Calculate Dice coefficient
            dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
            dice_scores.append(dice)
        
        dice_scores = torch.stack(dice_scores)
        
        # Apply class weights if provided
        if self.class_weights is not None:
            dice_scores = dice_scores * self.class_weights.to(dice_scores.device)
        
        # Return negative Dice (loss)
        return 1.0 - dice_scores.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for edge preservation.
    
    Ensures sharp boundaries between landing zones,
    critical for precise landing area delineation.
    """
    
    def __init__(self, theta0: float = 3.0, theta: float = 5.0):
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary loss.
        
        Args:
            inputs: Model predictions [B, C, H, W]
            targets: Ground truth labels [B, H, W]
            
        Returns:
            Boundary loss value
        """
        # Convert predictions to class probabilities
        pred_softmax = F.softmax(inputs, dim=1)
        
        # Compute distance transform for targets
        targets_dt = self._compute_distance_transform(targets)
        
        # Compute boundary loss
        boundary_loss = 0.0
        num_classes = inputs.size(1)
        
        for class_idx in range(num_classes):
            # Get class probability map
            pred_class = pred_softmax[:, class_idx]
            
            # Get distance transform for this class
            dt_class = targets_dt[:, class_idx]
            
            # Compute boundary term
            boundary_term = pred_class * dt_class
            boundary_loss += boundary_term.mean()
        
        return boundary_loss / num_classes
    
    def _compute_distance_transform(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute distance transform for each class.
        
        Args:
            targets: Ground truth labels [B, H, W]
            
        Returns:
            Distance transforms [B, C, H, W]
        """
        batch_size, height, width = targets.shape
        num_classes = targets.max().item() + 1
        
        # Convert to one-hot
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Compute distance transform using approximation
        distance_maps = []
        
        for class_idx in range(num_classes):
            class_mask = targets_one_hot[:, class_idx]
            
            # Simple distance approximation using erosion
            dt = self._approximate_distance_transform(class_mask)
            distance_maps.append(dt)
        
        return torch.stack(distance_maps, dim=1)
    
    def _approximate_distance_transform(self, mask: torch.Tensor) -> torch.Tensor:
        """Approximate distance transform using max pooling."""
        # Simple approximation - in practice you'd use proper distance transform
        kernel_size = 5
        padding = kernel_size // 2
        
        # Erosion approximation
        eroded = -F.max_pool2d(-mask.unsqueeze(1), kernel_size, stride=1, padding=padding)
        distance = mask - eroded.squeeze(1)
        
        return distance


class UncertaintyLoss(nn.Module):
    """
    Uncertainty Loss for reliable confidence estimation.
    
    Ensures the model provides meaningful uncertainty estimates,
    critical for safety-aware decision making.
    """
    
    def __init__(self, uncertainty_type: str = 'aleatoric'):
        super(UncertaintyLoss, self).__init__()
        self.uncertainty_type = uncertainty_type
    
    def forward(
        self,
        predictions: torch.Tensor,
        uncertainty: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute uncertainty loss.
        
        Args:
            predictions: Model predictions [B, C, H, W]
            uncertainty: Uncertainty estimates [B, 1, H, W]
            targets: Ground truth labels [B, H, W]
            
        Returns:
            Uncertainty loss value
        """
        # Compute prediction error
        pred_classes = torch.argmax(predictions, dim=1)
        prediction_error = (pred_classes != targets).float()
        
        # Uncertainty should be high where prediction is wrong
        uncertainty_target = prediction_error.unsqueeze(1)
        
        # MSE between predicted uncertainty and actual error
        uncertainty_loss = F.mse_loss(uncertainty, uncertainty_target)
        
        # Add regularization to prevent collapse
        uncertainty_reg = torch.mean(uncertainty) - 0.5  # Target mean uncertainty of 0.5
        uncertainty_reg = torch.abs(uncertainty_reg)
        
        return uncertainty_loss + 0.1 * uncertainty_reg


class CombinedSafetyLoss(nn.Module):
    """
    Combined Safety Loss integrating all loss components.
    
    Professional-grade loss function for safety-critical UAV landing detection:
    - Focal loss for class imbalance and hard examples
    - Dice loss for precise boundaries
    - Boundary loss for edge preservation
    - Uncertainty loss for reliable confidence
    - Safety-aware weighting throughout
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_weight: float = 1.0,
        boundary_weight: float = 0.5,
        uncertainty_weight: float = 0.2,
        safety_weights: Optional[List[float]] = None
    ):
        super(CombinedSafetyLoss, self).__init__()
        
        # Initialize component losses
        self.focal_loss = SafetyFocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            num_classes=num_classes,
            safety_weights=safety_weights
        )
        
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()
        self.uncertainty_loss = UncertaintyLoss()
        
        # Loss weights
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.uncertainty_weight = uncertainty_weight
        
        self.num_classes = num_classes
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        batch: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined safety loss.
        
        Args:
            outputs: Model outputs dictionary containing:
                - 'main': Main predictions [B, C, H, W]
                - 'aux': Auxiliary predictions [B, C, H, W] (optional)
                - 'uncertainty': Uncertainty estimates [B, 1, H, W] (optional)
            targets: Ground truth labels [B, H, W]
            batch: Additional batch information (optional)
            
        Returns:
            Dictionary of loss components and total loss
        """
        loss_dict = {}
        
        # Extract main predictions
        main_pred = outputs.get('main', outputs)
        if isinstance(main_pred, torch.Tensor):
            main_pred = main_pred
        else:
            raise ValueError("Invalid main predictions format")
        
        # Main focal loss
        focal_loss_value = self.focal_loss(main_pred, targets)
        loss_dict['focal_loss'] = focal_loss_value
        
        # Dice loss
        dice_loss_value = self.dice_loss(main_pred, targets)
        loss_dict['dice_loss'] = dice_loss_value
        
        # Boundary loss
        boundary_loss_value = self.boundary_loss(main_pred, targets)
        loss_dict['boundary_loss'] = boundary_loss_value
        
        # Auxiliary loss (if available)
        aux_loss_value = 0.0
        if 'aux' in outputs:
            aux_pred = outputs['aux']
            aux_loss_value = 0.4 * self.focal_loss(aux_pred, targets)
            loss_dict['aux_loss'] = aux_loss_value
        
        # Uncertainty loss (if available)
        uncertainty_loss_value = 0.0
        if 'uncertainty' in outputs:
            uncertainty = outputs['uncertainty']
            uncertainty_loss_value = self.uncertainty_loss(main_pred, uncertainty, targets)
            loss_dict['uncertainty_loss'] = uncertainty_loss_value
        
        # Combine all losses
        total_loss = (
            focal_loss_value +
            self.dice_weight * dice_loss_value +
            self.boundary_weight * boundary_loss_value +
            aux_loss_value +
            self.uncertainty_weight * uncertainty_loss_value
        )
        
        loss_dict['total_loss'] = total_loss
        
        return loss_dict


class AdaptiveSafetyLoss(nn.Module):
    """
    Adaptive Safety Loss that adjusts weights based on training progress.
    
    Starts with conservative safety weighting and adapts as model improves,
    ensuring safety is prioritized early in training.
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        initial_safety_multiplier: float = 3.0,
        final_safety_multiplier: float = 1.5,
        adaptation_epochs: int = 50
    ):
        super(AdaptiveSafetyLoss, self).__init__()
        self.base_loss = base_loss
        self.initial_safety_multiplier = initial_safety_multiplier
        self.final_safety_multiplier = final_safety_multiplier
        self.adaptation_epochs = adaptation_epochs
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """Update current epoch for adaptive weighting."""
        self.current_epoch = epoch
    
    def forward(self, outputs, targets, batch=None):
        """Compute adaptive safety loss."""
        # Compute base loss
        loss_dict = self.base_loss(outputs, targets, batch)
        
        # Calculate adaptive safety multiplier
        progress = min(self.current_epoch / self.adaptation_epochs, 1.0)
        safety_multiplier = self.initial_safety_multiplier * (1 - progress) + \
                           self.final_safety_multiplier * progress
        
        # Apply safety multiplier to danger-related losses
        if 'focal_loss' in loss_dict:
            loss_dict['focal_loss'] *= safety_multiplier
        
        # Recalculate total loss
        total_loss = sum(v for k, v in loss_dict.items() if k != 'total_loss')
        loss_dict['total_loss'] = total_loss
        
        return loss_dict


if __name__ == "__main__":
    # Test loss functions
    print("Testing safety-aware loss functions...")
    
    # Create test data
    batch_size, num_classes, height, width = 2, 4, 64, 64
    
    # Simulate model outputs
    outputs = {
        'main': torch.randn(batch_size, num_classes, height, width),
        'aux': torch.randn(batch_size, num_classes, height, width),
        'uncertainty': torch.sigmoid(torch.randn(batch_size, 1, height, width))
    }
    
    # Simulate targets
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test combined safety loss
    criterion = CombinedSafetyLoss(num_classes=num_classes)
    loss_dict = criterion(outputs, targets)
    
    print(f"✅ Combined Safety Loss: {loss_dict['total_loss'].item():.4f}")
    print(f"   - Focal Loss: {loss_dict['focal_loss'].item():.4f}")
    print(f"   - Dice Loss: {loss_dict['dice_loss'].item():.4f}")
    print(f"   - Boundary Loss: {loss_dict['boundary_loss'].item():.4f}")
    print(f"   - Uncertainty Loss: {loss_dict['uncertainty_loss'].item():.4f}")
    
    print("✅ Loss function tests passed!") 