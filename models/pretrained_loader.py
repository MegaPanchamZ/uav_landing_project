#!/usr/bin/env python3
"""
Pretrained Model Loader for UAV Landing Detection
================================================

Sophisticated adapter for loading and adapting pretrained BiSeNetV2 models
(especially Cityscapes-trained) for UAV landing detection tasks.

Features:
- Intelligent weight adaptation from Cityscapes (19 classes) to landing (4 classes)
- Layer-wise learning rate scheduling
- Frozen backbone option for fast fine-tuning
- Automatic architecture matching and weight filtering
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import re


class PretrainedModelLoader:
    """
    Advanced loader for adapting pretrained BiSeNetV2 models to UAV landing detection.
    
    Handles:
    - Cityscapes→Landing class adaptation (19→4 classes)
    - Architecture compatibility checking
    - Intelligent weight transfer with size mismatches
    - Layer freezing strategies
    """
    
    def __init__(
        self,
        model_paths_root: str = "../../model_pths",
        verbose: bool = True
    ):
        """
        Initialize pretrained model loader.
        
        Args:
            model_paths_root: Root directory containing pretrained models
            verbose: Enable detailed logging
        """
        self.model_paths_root = Path(model_paths_root)
        self.verbose = verbose
        
        # Available pretrained models
        self.available_models = self._scan_available_models()
        
        # Cityscapes class mapping for reference
        self.cityscapes_classes = {
            0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
            5: "pole", 6: "traffic_light", 7: "traffic_sign", 8: "vegetation",
            9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
            14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle"
        }
        
        # Landing classes for our task
        self.landing_classes = {
            0: "background", 1: "safe_landing", 2: "caution", 3: "danger"
        }
        
        if self.verbose:
            print(f"🔍 PretrainedModelLoader initialized:")
            print(f"   Model root: {self.model_paths_root}")
            print(f"   Available models: {len(self.available_models)}")
    
    def _scan_available_models(self) -> Dict[str, Dict]:
        """Scan and catalog available pretrained models."""
        
        models = {}
        
        if not self.model_paths_root.exists():
            return models
        
        for model_file in self.model_paths_root.glob("*.pth"):
            model_info = self._parse_model_filename(model_file.name)
            if model_info:
                models[model_file.stem] = {
                    'path': model_file,
                    'info': model_info,
                    'size_mb': model_file.stat().st_size / (1024 * 1024)
                }
        
        return models
    
    def _parse_model_filename(self, filename: str) -> Optional[Dict]:
        """Parse model filename to extract information."""
        
        # Pattern for standard model naming
        # Example: bisenetv2_fcn_4x4_1024x1024_160k_cityscapes_20210902_015551-bcf10f09.pth
        
        info = {
            'architecture': 'unknown',
            'dataset': 'unknown',
            'resolution': None,
            'iterations': None,
            'date': None,
            'hash': None
        }
        
        # Extract architecture
        if 'bisenetv2' in filename.lower():
            info['architecture'] = 'bisenetv2'
        elif 'bisenet' in filename.lower():
            info['architecture'] = 'bisenet'
        
        # Extract dataset
        if 'cityscapes' in filename.lower():
            info['dataset'] = 'cityscapes'
            info['num_classes'] = 19
        elif 'ade20k' in filename.lower():
            info['dataset'] = 'ade20k'
            info['num_classes'] = 150
        
        # Extract resolution
        res_match = re.search(r'(\d+)x(\d+)', filename)
        if res_match:
            info['resolution'] = (int(res_match.group(1)), int(res_match.group(2)))
        
        # Extract iterations
        iter_match = re.search(r'(\d+)k', filename)
        if iter_match:
            info['iterations'] = int(iter_match.group(1)) * 1000
        
        # Extract date
        date_match = re.search(r'(\d{8})', filename)
        if date_match:
            info['date'] = date_match.group(1)
        
        # Extract hash
        hash_match = re.search(r'-([a-f0-9]{8})', filename)
        if hash_match:
            info['hash'] = hash_match.group(1)
        
        return info
    
    def list_available_models(self) -> None:
        """Display available pretrained models."""
        
        print("📋 Available Pretrained Models:")
        print("=" * 60)
        
        if not self.available_models:
            print("   No pretrained models found")
            return
        
        for name, details in self.available_models.items():
            info = details['info']
            size_mb = details['size_mb']
            
            print(f"\n🏗️ {name}")
            print(f"   Architecture: {info['architecture']}")
            print(f"   Dataset: {info['dataset']} ({info.get('num_classes', 'unknown')} classes)")
            print(f"   Resolution: {info.get('resolution', 'unknown')}")
            print(f"   Iterations: {info.get('iterations', 'unknown')}")
            print(f"   Size: {size_mb:.1f} MB")
            print(f"   Path: {details['path']}")
    
    def load_pretrained_weights(
        self,
        model: nn.Module,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        adaptation_strategy: str = "intelligent",
        freeze_backbone: bool = False,
        freeze_layers: Optional[List[str]] = None
    ) -> Tuple[nn.Module, Dict]:
        """
        Load and adapt pretrained weights to target model.
        
        Args:
            model: Target model to load weights into
            model_name: Name of pretrained model (from available_models)
            model_path: Direct path to model file
            adaptation_strategy: How to handle mismatched layers ('intelligent', 'strict', 'ignore')
            freeze_backbone: Freeze backbone layers for fast fine-tuning
            freeze_layers: Specific layer patterns to freeze
            
        Returns:
            Tuple of (adapted_model, adaptation_info)
        """
        
        # Determine source model path
        if model_path:
            source_path = Path(model_path)
        elif model_name:
            if model_name not in self.available_models:
                raise ValueError(f"Model '{model_name}' not found. Available: {list(self.available_models.keys())}")
            source_path = self.available_models[model_name]['path']
        else:
            # Auto-select best available model
            source_path = self._auto_select_model()
        
        if not source_path.exists():
            raise FileNotFoundError(f"Model file not found: {source_path}")
        
        if self.verbose:
            print(f"🔄 Loading pretrained weights from: {source_path.name}")
        
        # Load pretrained state dict
        try:
            pretrained_state = torch.load(source_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'state_dict' in pretrained_state:
                pretrained_weights = pretrained_state['state_dict']
            elif 'model' in pretrained_state:
                pretrained_weights = pretrained_state['model']
            else:
                pretrained_weights = pretrained_state
                
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained weights: {e}")
        
        # Get target model state dict
        model_state = model.state_dict()
        
        # Adapt weights based on strategy
        if adaptation_strategy == "intelligent":
            adapted_weights, adaptation_info = self._intelligent_adaptation(
                pretrained_weights, model_state, source_path
            )
        elif adaptation_strategy == "strict":
            adapted_weights, adaptation_info = self._strict_adaptation(
                pretrained_weights, model_state
            )
        elif adaptation_strategy == "ignore":
            adapted_weights, adaptation_info = self._ignore_mismatches_adaptation(
                pretrained_weights, model_state
            )
        else:
            raise ValueError(f"Unknown adaptation strategy: {adaptation_strategy}")
        
        # Load adapted weights
        model.load_state_dict(adapted_weights, strict=False)
        
        # Apply layer freezing
        if freeze_backbone or freeze_layers:
            freeze_info = self._apply_layer_freezing(model, freeze_backbone, freeze_layers)
            adaptation_info.update(freeze_info)
        
        if self.verbose:
            self._print_adaptation_summary(adaptation_info)
        
        return model, adaptation_info
    
    def _auto_select_model(self) -> Path:
        """Automatically select the best available pretrained model."""
        
        if not self.available_models:
            raise ValueError("No pretrained models available")
        
        # Prioritize Cityscapes BiSeNetV2 models
        preferences = [
            ('bisenetv2', 'cityscapes'),
            ('bisenet', 'cityscapes'),
            ('bisenetv2', 'ade20k'),
        ]
        
        for arch_pref, dataset_pref in preferences:
            for name, details in self.available_models.items():
                info = details['info']
                if info['architecture'] == arch_pref and info['dataset'] == dataset_pref:
                    if self.verbose:
                        print(f" Auto-selected: {name}")
                    return details['path']
        
        # Fallback to first available
        first_model = next(iter(self.available_models.values()))
        if self.verbose:
            print(f"⚠️ Fallback selection: {first_model['path'].name}")
        return first_model['path']
    
    def _intelligent_adaptation(
        self, 
        pretrained_weights: Dict, 
        model_state: Dict,
        source_path: Path
    ) -> Tuple[Dict, Dict]:
        """Intelligently adapt pretrained weights to target model."""
        
        adapted_weights = model_state.copy()
        adaptation_info = {
            'source_model': source_path.name,
            'strategy': 'intelligent',
            'loaded_layers': [],
            'skipped_layers': [],
            'adapted_layers': [],
            'total_params_loaded': 0,
            'total_params_model': sum(p.numel() for p in model_state.values())
        }
        
        # Clean pretrained weights (remove module. prefix if present)
        cleaned_pretrained = {}
        for key, value in pretrained_weights.items():
            clean_key = key.replace('module.', '')
            cleaned_pretrained[clean_key] = value
        
        for name, target_param in model_state.items():
            if name in cleaned_pretrained:
                pretrained_param = cleaned_pretrained[name]
                
                if target_param.shape == pretrained_param.shape:
                    # Direct match - load weights
                    adapted_weights[name] = pretrained_param
                    adaptation_info['loaded_layers'].append(name)
                    adaptation_info['total_params_loaded'] += pretrained_param.numel()
                    
                elif self._is_classifier_layer(name) and len(target_param.shape) == len(pretrained_param.shape):
                    # Classifier layer with different number of classes
                    adapted_param = self._adapt_classifier_layer(
                        pretrained_param, target_param, name
                    )
                    adapted_weights[name] = adapted_param
                    adaptation_info['adapted_layers'].append(f"{name} ({pretrained_param.shape}→{target_param.shape})")
                    adaptation_info['total_params_loaded'] += min(pretrained_param.numel(), target_param.numel())
                    
                else:
                    # Shape mismatch - skip but log
                    adaptation_info['skipped_layers'].append(f"{name} (shape mismatch: {pretrained_param.shape} vs {target_param.shape})")
            else:
                # Layer not found in pretrained model
                adaptation_info['skipped_layers'].append(f"{name} (not found in pretrained)")
        
        return adapted_weights, adaptation_info
    
    def _strict_adaptation(self, pretrained_weights: Dict, model_state: Dict) -> Tuple[Dict, Dict]:
        """Strict adaptation - only load exact matches."""
        
        adapted_weights = model_state.copy()
        adaptation_info = {
            'strategy': 'strict',
            'loaded_layers': [],
            'skipped_layers': [],
            'total_params_loaded': 0,
            'total_params_model': sum(p.numel() for p in model_state.values())
        }
        
        # Clean pretrained weights
        cleaned_pretrained = {}
        for key, value in pretrained_weights.items():
            clean_key = key.replace('module.', '')
            cleaned_pretrained[clean_key] = value
        
        for name, target_param in model_state.items():
            if name in cleaned_pretrained:
                pretrained_param = cleaned_pretrained[name]
                
                if target_param.shape == pretrained_param.shape:
                    adapted_weights[name] = pretrained_param
                    adaptation_info['loaded_layers'].append(name)
                    adaptation_info['total_params_loaded'] += pretrained_param.numel()
                else:
                    adaptation_info['skipped_layers'].append(f"{name} (shape mismatch)")
            else:
                adaptation_info['skipped_layers'].append(f"{name} (not found)")
        
        return adapted_weights, adaptation_info
    
    def _ignore_mismatches_adaptation(self, pretrained_weights: Dict, model_state: Dict) -> Tuple[Dict, Dict]:
        """Load all possible weights, ignore mismatches silently."""
        
        adapted_weights = model_state.copy()
        adaptation_info = {
            'strategy': 'ignore_mismatches',
            'loaded_layers': [],
            'total_params_loaded': 0,
            'total_params_model': sum(p.numel() for p in model_state.values())
        }
        
        # Clean and load compatible weights
        for key, value in pretrained_weights.items():
            clean_key = key.replace('module.', '')
            if clean_key in model_state and model_state[clean_key].shape == value.shape:
                adapted_weights[clean_key] = value
                adaptation_info['loaded_layers'].append(clean_key)
                adaptation_info['total_params_loaded'] += value.numel()
        
        return adapted_weights, adaptation_info
    
    def _is_classifier_layer(self, layer_name: str) -> bool:
        """Check if layer is a classifier/head layer."""
        classifier_patterns = [
            'classifier', 'head', 'fc', 'conv_last', 'aux_head', 'decode_head'
        ]
        return any(pattern in layer_name.lower() for pattern in classifier_patterns)
    
    def _adapt_classifier_layer(
        self, 
        pretrained_param: torch.Tensor, 
        target_param: torch.Tensor,
        layer_name: str
    ) -> torch.Tensor:
        """Adapt classifier layer from different number of classes."""
        
        if len(pretrained_param.shape) == 4:  # Conv2d weight [out_channels, in_channels, H, W]
            # Use subset of pretrained classes or initialize new ones
            target_classes = target_param.shape[0]
            pretrained_classes = pretrained_param.shape[0]
            
            if target_classes <= pretrained_classes:
                # Use first N classes from pretrained (background + most relevant)
                adapted_param = pretrained_param[:target_classes].clone()
            else:
                # Initialize new classes with average of pretrained classes
                adapted_param = target_param.clone()
                adapted_param[:pretrained_classes] = pretrained_param
                # Initialize new classes with mean of existing classes
                if pretrained_classes > 0:
                    mean_weight = pretrained_param.mean(dim=0, keepdim=True)
                    for i in range(pretrained_classes, target_classes):
                        adapted_param[i] = mean_weight.squeeze(0)
            
        elif len(pretrained_param.shape) == 1:  # Bias [out_channels]
            target_classes = target_param.shape[0]
            pretrained_classes = pretrained_param.shape[0]
            
            if target_classes <= pretrained_classes:
                adapted_param = pretrained_param[:target_classes].clone()
            else:
                adapted_param = target_param.clone()
                adapted_param[:pretrained_classes] = pretrained_param
                # Initialize new biases to zero
                adapted_param[pretrained_classes:] = 0.0
        else:
            # Fallback to target parameter shape
            adapted_param = target_param.clone()
        
        return adapted_param
    
    def _apply_layer_freezing(
        self, 
        model: nn.Module, 
        freeze_backbone: bool,
        freeze_layers: Optional[List[str]]
    ) -> Dict:
        """Apply layer freezing strategy."""
        
        freeze_info = {
            'frozen_layers': [],
            'trainable_layers': [],
            'frozen_params': 0,
            'trainable_params': 0
        }
        
        for name, param in model.named_parameters():
            should_freeze = False
            
            # Freeze backbone if requested
            if freeze_backbone and self._is_backbone_layer(name):
                should_freeze = True
            
            # Freeze specific layer patterns
            if freeze_layers:
                for pattern in freeze_layers:
                    if pattern in name:
                        should_freeze = True
                        break
            
            if should_freeze:
                param.requires_grad = False
                freeze_info['frozen_layers'].append(name)
                freeze_info['frozen_params'] += param.numel()
            else:
                param.requires_grad = True
                freeze_info['trainable_layers'].append(name)
                freeze_info['trainable_params'] += param.numel()
        
        return freeze_info
    
    def _is_backbone_layer(self, layer_name: str) -> bool:
        """Check if layer is part of the backbone."""
        backbone_patterns = [
            'backbone', 'encoder', 'resnet', 'efficientnet', 'conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4'
        ]
        return any(pattern in layer_name.lower() for pattern in backbone_patterns)
    
    def _print_adaptation_summary(self, adaptation_info: Dict):
        """Print detailed adaptation summary."""
        
        print(f"\n📊 Pretrained Weight Adaptation Summary:")
        print(f"   Strategy: {adaptation_info['strategy']}")
        print(f"   Source: {adaptation_info.get('source_model', 'Unknown')}")
        
        loaded_count = len(adaptation_info.get('loaded_layers', []))
        adapted_count = len(adaptation_info.get('adapted_layers', []))
        skipped_count = len(adaptation_info.get('skipped_layers', []))
        
        print(f"   Loaded layers: {loaded_count}")
        if adapted_count > 0:
            print(f"   Adapted layers: {adapted_count}")
        print(f"   Skipped layers: {skipped_count}")
        
        params_loaded = adaptation_info.get('total_params_loaded', 0)
        params_total = adaptation_info.get('total_params_model', 1)
        load_percentage = (params_loaded / params_total) * 100
        
        print(f"   Parameters loaded: {params_loaded:,} / {params_total:,} ({load_percentage:.1f}%)")
        
        # Freezing info
        if 'frozen_params' in adaptation_info:
            frozen_params = adaptation_info['frozen_params']
            trainable_params = adaptation_info['trainable_params']
            total_params = frozen_params + trainable_params
            
            print(f"   Frozen parameters: {frozen_params:,} / {total_params:,} ({frozen_params/total_params*100:.1f}%)")
            print(f"   Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
        
        # Show adapted layers details
        if adapted_count > 0:
            print(f"\n🔧 Adapted Layers:")
            for layer in adaptation_info['adapted_layers']:
                print(f"   {layer}")


# Convenience functions
def load_cityscapes_bisenetv2(
    model: nn.Module,
    model_paths_root: str = "../../model_pths",
    freeze_backbone: bool = False
) -> Tuple[nn.Module, Dict]:
    """
    Convenience function to load Cityscapes BiSeNetV2 weights.
    
    Args:
        model: Target model
        model_paths_root: Path to pretrained models
        freeze_backbone: Whether to freeze backbone for fast fine-tuning
        
    Returns:
        Tuple of (model_with_weights, adaptation_info)
    """
    loader = PretrainedModelLoader(model_paths_root, verbose=True)
    
    # Try to find best Cityscapes BiSeNetV2 model
    cityscapes_models = [
        name for name, details in loader.available_models.items()
        if details['info']['architecture'] == 'bisenetv2' and details['info']['dataset'] == 'cityscapes'
    ]
    
    if cityscapes_models:
        # Use the first Cityscapes BiSeNetV2 model found
        best_model = cityscapes_models[0]
        print(f" Using Cityscapes BiSeNetV2: {best_model}")
    else:
        best_model = None
        print("⚠️ No Cityscapes BiSeNetV2 found, using auto-selection")
    
    return loader.load_pretrained_weights(
        model=model,
        model_name=best_model,
        adaptation_strategy="intelligent",
        freeze_backbone=freeze_backbone
    )


if __name__ == "__main__":
    # Test the pretrained loader
    loader = PretrainedModelLoader("../../model_pths")
    loader.list_available_models() 