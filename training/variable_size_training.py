#!/usr/bin/env python3
"""
Variable Size Training Support for UAV Landing Detection
======================================================

Custom training utilities for handling variable input sizes efficiently.
Includes custom collation functions, adaptive batching, and size-aware sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict
import random


def variable_size_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collation function for variable-sized inputs.
    
    Instead of stacking tensors, returns lists of tensors with different sizes.
    The model must handle variable sizes internally.
    """
    
    # Separate the batch into individual components
    images = [item['image'] for item in batch]
    masks = [item['mask'] for item in batch]
    
    # Create batch dictionary with lists instead of stacked tensors
    batch_dict = {
        'image': images,
        'mask': masks,
        'batch_size': len(batch)
    }
    
    # Add other metadata if present
    for key in ['image_path', 'base_idx', 'quality_score', 'scale']:
        if key in batch[0]:
            batch_dict[key] = [item[key] for item in batch]
    
    return batch_dict


def adaptive_size_collate_fn(batch: List[Dict[str, Any]], 
                           target_size: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    """
    Adaptive collation that resizes all inputs to the same size for standard training.
    
    Args:
        batch: List of batch items
        target_size: Target size to resize to. If None, uses the largest size in batch.
    """
    
    images = [item['image'] for item in batch]
    masks = [item['mask'] for item in batch]
    
    # Determine target size
    if target_size is None:
        # Use the largest size in the batch
        max_h = max(img.shape[-2] for img in images)
        max_w = max(img.shape[-1] for img in images)
        target_size = (max_h, max_w)
    
    # Resize all images and masks to target size
    resized_images = []
    resized_masks = []
    
    for img, mask in zip(images, masks):
        if img.shape[-2:] != target_size:
            img_resized = F.interpolate(
                img.unsqueeze(0),
                size=target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        else:
            img_resized = img
            
        if mask.shape[-2:] != target_size:
            mask_resized = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=target_size,
                mode='nearest'
            ).squeeze(0).squeeze(0).long()
        else:
            mask_resized = mask
            
        resized_images.append(img_resized)
        resized_masks.append(mask_resized)
    
    # Stack resized tensors
    batch_dict = {
        'image': torch.stack(resized_images),
        'mask': torch.stack(resized_masks),
        'target_size': target_size
    }
    
    # Add other metadata
    for key in ['image_path', 'base_idx', 'quality_score', 'scale']:
        if key in batch[0]:
            batch_dict[key] = [item[key] for item in batch]
    
    return batch_dict


class SizeGroupedSampler(Sampler):
    """
    Sampler that groups samples by similar sizes to enable more efficient batching.
    Reduces memory usage and padding overhead.
    """
    
    def __init__(self, dataset, batch_size: int = 8, shuffle: bool = True, 
                 size_tolerance: float = 0.1):
        """
        Args:
            dataset: Dataset with items that have 'scale' or size information
            batch_size: Target batch size
            shuffle: Whether to shuffle within size groups
            size_tolerance: Relative tolerance for grouping sizes (0.1 = 10%)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size_tolerance = size_tolerance
        
        # Group indices by size
        self.size_groups = self._group_by_size()
        
    def _group_by_size(self) -> Dict[Tuple[int, int], List[int]]:
        """Group dataset indices by similar input sizes."""
        size_groups = defaultdict(list)
        
        for idx in range(len(self.dataset)):
            try:
                # Try to get size information from dataset
                if hasattr(self.dataset, 'patch_index'):
                    # Cached dataset
                    size = self.dataset.patch_index[idx]['scale']
                elif hasattr(self.dataset, 'patches'):
                    # Multi-scale dataset
                    size = self.dataset.patches[idx]['scale']
                else:
                    # Default size
                    size = (512, 512)
                    
                # Group by similar sizes (with tolerance)
                size_key = self._get_size_group_key(size)
                size_groups[size_key].append(idx)
                
            except (IndexError, KeyError, AttributeError):
                # Fallback to default group
                size_groups[(512, 512)].append(idx)
        
        return dict(size_groups)
    
    def _get_size_group_key(self, size: Tuple[int, int]) -> Tuple[int, int]:
        """Get the group key for a given size with tolerance."""
        w, h = size
        
        # Round to nearest multiple based on tolerance
        tolerance_w = max(16, int(w * self.size_tolerance))
        tolerance_h = max(16, int(h * self.size_tolerance))
        
        group_w = ((w + tolerance_w // 2) // tolerance_w) * tolerance_w
        group_h = ((h + tolerance_h // 2) // tolerance_h) * tolerance_h
        
        return (group_w, group_h)
    
    def __iter__(self):
        """Generate batches grouped by size."""
        all_batches = []
        
        for size_key, indices in self.size_groups.items():
            if self.shuffle:
                random.shuffle(indices)
            
            # Create batches from this size group
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                all_batches.append(batch_indices)
        
        # Shuffle the order of batches
        if self.shuffle:
            random.shuffle(all_batches)
        
        # Yield individual indices
        for batch in all_batches:
            for idx in batch:
                yield idx
    
    def __len__(self):
        return len(self.dataset)


def create_variable_size_dataloader(
    dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    collate_strategy: str = "adaptive",
    target_size: Optional[Tuple[int, int]] = None,
    use_size_grouping: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader optimized for variable input sizes.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle data
        collate_strategy: 'variable' (lists) or 'adaptive' (resize to common size)
        target_size: Fixed target size for adaptive strategy
        use_size_grouping: Use size-grouped sampling for efficiency
        num_workers: Number of workers
        **kwargs: Additional DataLoader arguments
    
    Returns:
        DataLoader configured for variable sizes
    """
    
    # Choose collation function
    if collate_strategy == "variable":
        collate_fn = variable_size_collate_fn
    elif collate_strategy == "adaptive":
        collate_fn = lambda batch: adaptive_size_collate_fn(batch, target_size)
    else:
        raise ValueError(f"Unknown collate strategy: {collate_strategy}")
    
    # Choose sampler
    if use_size_grouping and shuffle:
        sampler = SizeGroupedSampler(dataset, batch_size, shuffle=True)
        # Don't use DataLoader's shuffle when using custom sampler
        shuffle = False
    else:
        sampler = None
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,  # Disable for variable sizes
        **kwargs
    )


class VariableSizeTrainer:
    """
    Trainer class specifically designed for variable input sizes.
    
    Handles:
    - Variable size batches
    - Adaptive memory management
    - Size-aware loss computation
    - Efficient gradient accumulation
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        
        # Ensure model can handle variable sizes
        if not self._check_model_compatibility():
            print("⚠️  Model may not support variable input sizes properly")
    
    def _check_model_compatibility(self) -> bool:
        """Check if model can handle variable input sizes."""
        # Check if it's one of our adaptive models
        from models.enhanced_architectures import AdaptiveSegmentationModel, FlexibleBiSeNetV2
        
        return isinstance(self.model, (AdaptiveSegmentationModel, FlexibleBiSeNetV2))
    
    def train_step(self, batch: Dict[str, Any], criterion, optimizer) -> Dict[str, float]:
        """
        Perform a training step with variable-sized inputs.
        
        Args:
            batch: Batch from variable_size_dataloader
            criterion: Loss function
            optimizer: Optimizer
            
        Returns:
            Dictionary with loss information
        """
        
        if isinstance(batch['image'], list):
            # Variable size batch - process each sample individually
            return self._train_step_variable(batch, criterion, optimizer)
        else:
            # Fixed size batch - standard processing
            return self._train_step_standard(batch, criterion, optimizer)
    
    def _train_step_variable(self, batch: Dict[str, Any], criterion, optimizer) -> Dict[str, float]:
        """Train step for variable-sized inputs."""
        images = batch['image']
        masks = batch['mask']
        
        total_loss = 0.0
        batch_size = len(images)
        
        # Process each sample individually
        for img, mask in zip(images, masks):
            # Move to device
            img = img.to(self.device).unsqueeze(0)  # Add batch dimension
            mask = mask.to(self.device).unsqueeze(0)
            
            # Forward pass
            outputs = self.model(img)
            
            # Compute loss
            if isinstance(outputs, dict):
                main_output = outputs['main']
            else:
                main_output = outputs
            
            loss = criterion(main_output, mask)
            
            # Normalize loss by batch size for proper gradient scaling
            loss = loss / batch_size
            
            # Backward pass
            loss.backward()
            
            total_loss += loss.item() * batch_size
        
        # Update parameters after accumulating gradients from all samples
        optimizer.step()
        optimizer.zero_grad()
        
        return {
            'loss': total_loss,
            'batch_size': batch_size
        }
    
    def _train_step_standard(self, batch: Dict[str, Any], criterion, optimizer) -> Dict[str, float]:
        """Standard training step for fixed-size inputs."""
        images = batch['image'].to(self.device)
        masks = batch['mask'].to(self.device)
        
        # Forward pass
        outputs = self.model(images)
        
        # Compute loss
        if isinstance(outputs, dict):
            main_output = outputs['main']
        else:
            main_output = outputs
        
        loss = criterion(main_output, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {
            'loss': loss.item(),
            'batch_size': images.size(0)
        }


# Usage example
if __name__ == "__main__":
    print("🔄 Variable Size Training Module")
    print("=" * 50)
    print("""
    This module provides utilities for training with variable input sizes:
    
    1. Custom collation functions:
       - variable_size_collate_fn: Returns lists of different-sized tensors
       - adaptive_size_collate_fn: Resizes all to common size
    
    2. Size-grouped sampling:
       - Groups similar sizes together for efficiency
       - Reduces memory usage and padding overhead
    
    3. Variable size trainer:
       - Handles gradient accumulation for variable sizes
       - Compatible with adaptive models
    
    Example usage:
    
    # Create adaptive model
    model = create_enhanced_model(
        model_type="flexible_bisenetv2",
        variable_input_size=True
    )
    
    # Create variable size dataloader
    dataloader = create_variable_size_dataloader(
        dataset=cached_dataset,
        batch_size=8,
        collate_strategy="adaptive",  # or "variable"
        use_size_grouping=True
    )
    
    # Create trainer
    trainer = VariableSizeTrainer(model, device)
    
    # Training loop
    for batch in dataloader:
        loss_info = trainer.train_step(batch, criterion, optimizer)
    """) 