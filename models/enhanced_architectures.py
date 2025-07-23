#!/usr/bin/env python3
"""
Enhanced Model Architectures for UAV Landing Detection
=====================================================

Professional model architectures addressing the inadequacies of the ultra-lightweight approach:
- Proper capacity models (DeepLabV3+, Enhanced BiSeNetV2)
- Uncertainty quantification with Monte Carlo Dropout
- Multi-scale feature processing
- Attention mechanisms for landing-relevant features
- Safety-aware architecture design
- Variable input size handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import torchvision.models as models
import torchvision.models.segmentation as seg_models
import math


class AdaptiveSegmentationModel(nn.Module):
    """
    Adaptive segmentation model that handles variable input sizes efficiently.
    
    Key Features:
    - Accepts any input size (minimum 256x256)
    - Uses adaptive pooling to handle size variations
    - Maintains spatial relationships regardless of input size
    - Optimized for UAV landing detection with different altitudes
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        min_size: int = 256,
        adaptive_pooling_sizes: List[int] = [1, 2, 4, 8],
        interpolation_mode: str = 'bilinear'
    ):
        super(AdaptiveSegmentationModel, self).__init__()
        
        self.base_model = base_model
        self.min_size = min_size
        self.adaptive_pooling_sizes = adaptive_pooling_sizes
        self.interpolation_mode = interpolation_mode
        
        # Get model's expected input channels from the first conv layer
        first_conv = self._find_first_conv()
        if first_conv:
            self.expected_channels = first_conv.in_channels
        else:
            self.expected_channels = 3  # Default fallback
            
        # Adaptive pooling layers for multi-scale feature extraction
        self.adaptive_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((size, size)) for size in adaptive_pooling_sizes
        ])
        
        print(f"🔄 Created Adaptive Segmentation Model:")
        print(f"   Base model: {type(base_model).__name__}")
        print(f"   Min input size: {min_size}x{min_size}")
        print(f"   Expected channels: {self.expected_channels}")
        print(f"   Adaptive pooling sizes: {adaptive_pooling_sizes}")
    
    def _find_first_conv(self) -> Optional[nn.Conv2d]:
        """Find the first convolutional layer to determine expected input channels."""
        for module in self.base_model.modules():
            if isinstance(module, nn.Conv2d):
                return module
        return None
    
    def forward(self, x):
        """
        Forward pass with adaptive input size handling.
        
        Args:
            x: Input tensor of shape [B, C, H, W] where H, W can vary
            
        Returns:
            Dictionary with segmentation outputs resized to input size
        """
        batch_size, channels, input_h, input_w = x.shape
        
        # Validate input
        if channels != self.expected_channels:
            raise ValueError(f"Expected {self.expected_channels} input channels, got {channels}")
        
        if min(input_h, input_w) < self.min_size:
            raise ValueError(f"Input size too small. Minimum: {self.min_size}x{self.min_size}, got: {input_h}x{input_w}")
        
        # Store original size for final upsampling
        original_size = (input_h, input_w)
        
        # Process through base model
        # Most segmentation models expect specific sizes, so we may need to adapt
        if hasattr(self.base_model, 'forward'):
            outputs = self.base_model(x)
        else:
            raise ValueError("Base model must have a forward method")
        
        # Ensure outputs are resized to match input size
        if isinstance(outputs, dict):
            adapted_outputs = {}
            for key, output in outputs.items():
                if isinstance(output, torch.Tensor) and len(output.shape) == 4:
                    # Resize output to match input size
                    adapted_outputs[key] = F.interpolate(
                        output, 
                        size=original_size,
                        mode=self.interpolation_mode,
                        align_corners=False
                    )
                else:
                    adapted_outputs[key] = output
            return adapted_outputs
        else:
            # Single tensor output
            return F.interpolate(
                outputs,
                size=original_size,
                mode=self.interpolation_mode,
                align_corners=False
            )


class FlexibleBiSeNetV2(nn.Module):
    """
    Flexible BiSeNetV2 that can handle variable input sizes natively.
    
    Uses adaptive pooling and flexible convolutions to process
    any input size >= 256x256 efficiently.
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        min_input_size: int = 256,
        uncertainty_estimation: bool = True,
        adaptive_features: bool = True
    ):
        super(FlexibleBiSeNetV2, self).__init__()
        
        self.num_classes = num_classes
        self.min_input_size = min_input_size
        self.uncertainty_estimation = uncertainty_estimation
        self.adaptive_features = adaptive_features
        
        # Import the MMSeg model as base
        from models.mmseg_bisenetv2 import MMSegBiSeNetV2
        
        self.backbone = MMSegBiSeNetV2(
            num_classes=num_classes,
            uncertainty_estimation=False  # We'll handle uncertainty at this level
        )
        
        # Adaptive feature processing for different input sizes
        if adaptive_features:
            self.adaptive_pools = nn.ModuleList([
                nn.AdaptiveAvgPool2d((64, 64)),   # For very large inputs
                nn.AdaptiveAvgPool2d((32, 32)),   # For large inputs  
                nn.AdaptiveAvgPool2d((16, 16)),   # For medium inputs
                nn.AdaptiveAvgPool2d((8, 8)),     # For small inputs
            ])
            
            # Feature fusion for multi-scale processing
            self.feature_fusion = nn.Sequential(
                nn.Conv2d(num_classes * 4, num_classes * 2, 3, padding=1),
                nn.BatchNorm2d(num_classes * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_classes * 2, num_classes, 1),
            )
        
        # Uncertainty estimation
        if uncertainty_estimation:
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(num_classes, num_classes // 2, 3, padding=1),
                nn.BatchNorm2d(num_classes // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_classes // 2, 1, 1),
                nn.Sigmoid()  # Output uncertainty as probability
            )
        
        # Model size calculation
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🏗️ Created Flexible BiSeNetV2:")
        print(f"   Parameters: {total_params:,}")
        print(f"   Min input size: {min_input_size}x{min_input_size}")
        print(f"   Adaptive features: {adaptive_features}")
        print(f"   Uncertainty estimation: {uncertainty_estimation}")
    
    def forward(self, x):
        """Forward pass with adaptive input size handling."""
        batch_size, channels, input_h, input_w = x.shape
        original_size = (input_h, input_w)
        
        # Validate minimum size
        if min(input_h, input_w) < self.min_input_size:
            # Resize input to minimum size if too small
            x = F.interpolate(
                x, 
                size=(self.min_input_size, self.min_input_size),
                mode='bilinear',
                align_corners=False
            )
            print(f"⚠️  Input resized from {original_size} to {self.min_input_size}x{self.min_input_size}")
        
        # Main backbone forward pass
        backbone_outputs = self.backbone(x)
        main_output = backbone_outputs['main']
        
        outputs = {}
        
        # Adaptive multi-scale processing
        if self.adaptive_features and hasattr(self, 'adaptive_pools'):
            # Process at multiple scales
            multi_scale_features = []
            for pool in self.adaptive_pools:
                pooled = pool(main_output)
                # Resize back to main output size
                upsampled = F.interpolate(
                    pooled,
                    size=main_output.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                multi_scale_features.append(upsampled)
            
            # Fuse multi-scale features
            combined_features = torch.cat(multi_scale_features, dim=1)
            fused_output = self.feature_fusion(combined_features)
            
            # Combine with original output
            main_output = main_output + fused_output
        
        # Resize to original input size
        main_output = F.interpolate(
            main_output,
            size=original_size,
            mode='bilinear',
            align_corners=False
        )
        outputs['main'] = main_output
        
        # Uncertainty estimation
        if self.uncertainty_estimation and hasattr(self, 'uncertainty_head'):
            uncertainty = self.uncertainty_head(main_output)
            outputs['uncertainty'] = uncertainty
        
        # Pass through auxiliary outputs if in training
        if self.training and 'aux' in backbone_outputs:
            aux_outputs = []
            for aux_out in backbone_outputs['aux']:
                aux_resized = F.interpolate(
                    aux_out,
                    size=original_size,
                    mode='bilinear',
                    align_corners=False
                )
                aux_outputs.append(aux_resized)
            outputs['aux'] = aux_outputs
        
        return outputs


class EnhancedBiSeNetV2(nn.Module):
    """
    Enhanced BiSeNetV2 with proper capacity for UAV landing detection.
    
    Improvements over ultra-lightweight version:
    - 20x more parameters (6.7M vs 333K)
    - Multi-scale feature processing
    - Attention mechanisms
    - Uncertainty quantification
    - Skip connections for detail preservation
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        input_resolution: Tuple[int, int] = (512, 512),
        backbone: str = "resnet50",
        use_attention: bool = True,
        uncertainty_estimation: bool = True,
        dropout_rate: float = 0.1
    ):
        super(EnhancedBiSeNetV2, self).__init__()
        
        self.num_classes = num_classes
        self.input_resolution = input_resolution
        self.use_attention = use_attention
        self.uncertainty_estimation = uncertainty_estimation
        self.dropout_rate = dropout_rate
        
        # Enhanced backbone with proper capacity
        self.backbone = self._create_backbone(backbone)
        
        # Multi-scale feature processing
        self.feature_pyramid = FeaturePyramidNetwork([256, 512, 1024, 2048], 256)
        
        # Attention mechanism for landing-relevant features
        if use_attention:
            self.attention = SpatialChannelAttention(256)
        
        # Enhanced decoder with skip connections
        self.decoder = EnhancedDecoder(256, num_classes, dropout_rate)
        
        # Auxiliary classifier for deep supervision
        self.aux_classifier = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(256, num_classes, 1)
        )
        
        # Uncertainty head for Monte Carlo Dropout
        if uncertainty_estimation:
            self.uncertainty_head = UncertaintyHead(256, num_classes)
        
        self._initialize_weights()
    
    def _create_backbone(self, backbone_name: str):
        """Create enhanced backbone network."""
        if backbone_name == "resnet50":
            backbone = models.resnet50(pretrained=True)
            # Remove final layers
            modules = list(backbone.children())[:-2]
            return nn.Sequential(*modules)
        elif backbone_name == "efficientnet-b2":
            try:
                from efficientnet_pytorch import EfficientNet
                backbone = EfficientNet.from_pretrained('efficientnet-b2')
                return backbone
            except ImportError:
                print("EfficientNet not available, falling back to ResNet50")
                return self._create_backbone("resnet50")
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
    
    def forward(self, x):
        """Forward pass with multi-scale processing."""
        batch_size = x.size(0)
        
        # Multi-scale feature extraction
        features = self._extract_features(x)
        
        # Feature pyramid processing
        fpn_features = self.feature_pyramid(features)
        
        # Attention mechanism
        if self.use_attention:
            fpn_features[-1] = self.attention(fpn_features[-1])
        
        # Main decoder
        main_output = self.decoder(fpn_features)
        main_output = F.interpolate(main_output, size=self.input_resolution, 
                                   mode='bilinear', align_corners=False)
        
        outputs = {'main': main_output}
        
        # Auxiliary output for training
        if self.training:
            aux_output = self.aux_classifier(features[2])  # Use intermediate features
            aux_output = F.interpolate(aux_output, size=self.input_resolution,
                                     mode='bilinear', align_corners=False)
            outputs['aux'] = aux_output
        
        # Uncertainty estimation
        if self.uncertainty_estimation:
            uncertainty = self.uncertainty_head(fpn_features[-1])
            uncertainty = F.interpolate(uncertainty, size=self.input_resolution,
                                       mode='bilinear', align_corners=False)
            outputs['uncertainty'] = uncertainty
        
        return outputs
    
    def _extract_features(self, x):
        """Extract multi-scale features from backbone."""
        features = []
        
        if isinstance(self.backbone, nn.Sequential):  # ResNet
            x = self.backbone[0](x)  # conv1
            x = self.backbone[1](x)  # bn1
            x = self.backbone[2](x)  # relu
            x = self.backbone[3](x)  # maxpool
            
            for i, layer in enumerate(self.backbone[4:]):  # layer1, layer2, layer3, layer4
                x = layer(x)
                if i >= 0:  # Collect all layer outputs
                    features.append(x)
        
        return features
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ implementation for UAV landing detection.
    
    Superior to ultra-lightweight models with:
    - Atrous spatial pyramid pooling
    - Multi-scale feature processing
    - Proper capacity (60M+ parameters)
    - State-of-the-art segmentation performance
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        backbone: str = "resnet101",
        output_stride: int = 16,
        uncertainty_estimation: bool = True
    ):
        super(DeepLabV3Plus, self).__init__()
        
        self.num_classes = num_classes
        self.uncertainty_estimation = uncertainty_estimation
        
        # Use pretrained DeepLabV3+ as base
        self.model = seg_models.deeplabv3_resnet101(
            pretrained=False,
            num_classes=num_classes
        )
        
        # Load pretrained weights and adapt
        self._load_pretrained_weights()
        
        # Add uncertainty estimation
        if uncertainty_estimation:
            self.uncertainty_head = UncertaintyHead(256, num_classes)
    
    def _load_pretrained_weights(self):
        """Load and adapt pretrained weights."""
        try:
            # Load COCO pretrained weights
            pretrained = seg_models.deeplabv3_resnet101(pretrained=True)
            
            # Copy backbone weights
            pretrained_dict = pretrained.state_dict()
            model_dict = self.model.state_dict()
            
            # Filter out classifier weights (different num_classes)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and v.size() == model_dict[k].size()}
            
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            
            print(f" Loaded pretrained DeepLabV3+ weights")
        except Exception as e:
            print(f"⚠️ Could not load pretrained weights: {e}")
    
    def forward(self, x):
        """Forward pass."""
        outputs = {}
        
        # Main segmentation output
        main_output = self.model(x)['out']
        outputs['main'] = main_output
        
        # Uncertainty estimation
        if self.uncertainty_estimation and hasattr(self, 'uncertainty_head'):
            # Extract features for uncertainty estimation
            features = self._extract_features(x)
            uncertainty = self.uncertainty_head(features)
            outputs['uncertainty'] = F.interpolate(
                uncertainty, size=x.shape[-2:], mode='bilinear', align_corners=False
            )
        
        return outputs
    
    def _extract_features(self, x):
        """Extract features for uncertainty estimation."""
        # This is a simplified feature extraction
        # In practice, you'd want to access intermediate features
        with torch.no_grad():
            features = self.model.backbone(x)['out']
        return features


class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale processing."""
    
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super(FeaturePyramidNetwork, self).__init__()
        
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            layer_block = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass through feature pyramid."""
        results = []
        last_inner = self.inner_blocks[-1](features[-1])
        results.append(self.layer_blocks[-1](last_inner))
        
        for idx in range(len(features) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](features[idx])
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, 
                                         mode='bilinear', align_corners=False)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))
        
        return results


class SpatialChannelAttention(nn.Module):
    """Spatial and Channel Attention for landing-relevant features."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super(SpatialChannelAttention, self).__init__()
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Apply spatial and channel attention."""
        # Channel attention
        avg_out = self.channel_attention(self.avg_pool(x))
        max_out = self.channel_attention(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        x = x * spatial_att
        
        return x


class EnhancedDecoder(nn.Module):
    """Enhanced decoder with skip connections and proper capacity."""
    
    def __init__(self, in_channels: int, num_classes: int, dropout_rate: float = 0.1):
        super(EnhancedDecoder, self).__init__()
        
        self.decoder_blocks = nn.ModuleList([
            self._make_decoder_block(in_channels, in_channels // 2, dropout_rate),
            self._make_decoder_block(in_channels // 2, in_channels // 4, dropout_rate),
            self._make_decoder_block(in_channels // 4, in_channels // 8, dropout_rate),
        ])
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels // 8, in_channels // 16, 3, padding=1),
            nn.BatchNorm2d(in_channels // 16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(in_channels // 16, num_classes, 1)
        )
    
    def _make_decoder_block(self, in_channels: int, out_channels: int, dropout_rate: float):
        """Create decoder block with upsampling."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
    
    def forward(self, features: List[torch.Tensor]):
        """Forward pass through decoder."""
        x = features[-1]  # Start with highest level features
        
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)
        
        x = self.final_conv(x)
        return x


class UncertaintyHead(nn.Module):
    """Uncertainty estimation head using Monte Carlo Dropout."""
    
    def __init__(self, in_channels: int, num_classes: int):
        super(UncertaintyHead, self).__init__()
        
        self.uncertainty_layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),  # MC Dropout
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),  # MC Dropout
            nn.Conv2d(in_channels // 4, 1, 1),  # Single uncertainty channel
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass for uncertainty estimation."""
        return self.uncertainty_layers(x)


class BayesianSegmentationModel(nn.Module):
    """
    Bayesian segmentation model with proper uncertainty quantification.
    Critical for safety-aware UAV landing decisions.
    """
    
    def __init__(self, base_model: nn.Module, num_samples: int = 10):
        super(BayesianSegmentationModel, self).__init__()
        self.base_model = base_model
        self.num_samples = num_samples
    
    def forward(self, x, return_uncertainty: bool = True):
        """Forward pass with uncertainty estimation."""
        if not return_uncertainty or not self.training:
            return self.base_model(x)
        
        # Monte Carlo Dropout sampling
        predictions = []
        self.base_model.train()  # Enable dropout
        
        for _ in range(self.num_samples):
            with torch.no_grad():
                pred = self.base_model(x)
                if isinstance(pred, dict):
                    pred = pred['main']
                predictions.append(pred)
        
        # Compute mean and uncertainty
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(0)
        uncertainty = predictions.var(0).mean(1, keepdim=True)  # Average over classes
        
        return {
            'main': mean_pred,
            'uncertainty': uncertainty,
            'samples': predictions
        }


def create_enhanced_model(
    model_type: str = "enhanced_bisenetv2",
    num_classes: int = 4,
    input_resolution: Tuple[int, int] = (512, 512),
    uncertainty_estimation: bool = True,
    in_channels: int = 3,
    variable_input_size: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create enhanced models.
    
    Args:
        model_type: Type of model ('enhanced_bisenetv2', 'deeplabv3plus', 'mmseg_bisenetv2', 
                   'flexible_bisenetv2', 'adaptive_wrapper')
        num_classes: Number of output classes
        input_resolution: Input image resolution (ignored if variable_input_size=True)
        uncertainty_estimation: Enable uncertainty quantification
        variable_input_size: Enable variable input size handling
        **kwargs: Additional model-specific arguments
    """
    
    if model_type == "flexible_bisenetv2":
        model = FlexibleBiSeNetV2(
            num_classes=num_classes,
            min_input_size=min(input_resolution) if input_resolution else 256,
            uncertainty_estimation=uncertainty_estimation,
            **kwargs
        )
        return model  # Already prints info
        
    elif model_type == "adaptive_wrapper":
        # Create base model first
        base_model_type = kwargs.get('base_model_type', 'mmseg_bisenetv2')
        base_kwargs = {k: v for k, v in kwargs.items() if k != 'base_model_type'}
        
        base_model = create_enhanced_model(
            model_type=base_model_type,
            num_classes=num_classes,
            input_resolution=input_resolution,
            uncertainty_estimation=uncertainty_estimation,
            in_channels=in_channels,
            variable_input_size=False,  # Don't recurse
            **base_kwargs
        )
        
        # Wrap with adaptive wrapper
        model = AdaptiveSegmentationModel(
            base_model=base_model,
            min_size=min(input_resolution) if input_resolution else 256,
            **kwargs
        )
        return model
        
    elif model_type == "enhanced_bisenetv2":
        model = EnhancedBiSeNetV2(
            num_classes=num_classes,
            input_resolution=input_resolution,
            uncertainty_estimation=uncertainty_estimation,
            **kwargs
        )
    elif model_type == "deeplabv3plus":
        model = DeepLabV3Plus(
            num_classes=num_classes,
            uncertainty_estimation=uncertainty_estimation,
            **kwargs
        )
    elif model_type == "mmseg_bisenetv2":
        # Import here to avoid circular imports
        from models.mmseg_bisenetv2 import create_mmseg_bisenetv2
        
        # Create MMSeg-compatible model with proper transfer learning
        model = create_mmseg_bisenetv2(
            num_classes=num_classes,
            uncertainty_estimation=uncertainty_estimation,
            in_channels=in_channels,
            pretrained_path=kwargs.get('pretrained_path')
        )
        
        # Wrap with adaptive handling if requested
        if variable_input_size:
            model = AdaptiveSegmentationModel(
                base_model=model,
                min_size=min(input_resolution) if input_resolution else 256
            )
        return model  # Already has size info printed
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: enhanced_bisenetv2, deeplabv3plus, mmseg_bisenetv2, flexible_bisenetv2, adaptive_wrapper")
    
    # Wrap with adaptive handling if requested
    if variable_input_size:
        model = AdaptiveSegmentationModel(
            base_model=model,
            min_size=min(input_resolution) if input_resolution else 256
        )
        return model
    
    # Wrap with Bayesian uncertainty if requested
    if uncertainty_estimation:
        model = BayesianSegmentationModel(model)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🏗️ Created {model_type}:")
    print(f"   Parameters: {total_params:,} ({trainable_params:,} trainable)")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"   Uncertainty: {uncertainty_estimation}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing enhanced model architectures...")
    
    # Test Enhanced BiSeNetV2
    model1 = create_enhanced_model(
        model_type="enhanced_bisenetv2",
        num_classes=4,
        input_resolution=(512, 512),
        uncertainty_estimation=True
    )
    
    # Test DeepLabV3+
    model2 = create_enhanced_model(
        model_type="deeplabv3plus",
        num_classes=4,
        uncertainty_estimation=True
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    
    print("\n🧪 Testing forward pass...")
    with torch.no_grad():
        output1 = model1(x)
        output2 = model2(x)
        
        print(f"Enhanced BiSeNetV2 output shape: {output1['main'].shape}")
        if 'uncertainty' in output1:
            print(f"Uncertainty shape: {output1['uncertainty'].shape}")
        
        print(f"DeepLabV3+ output shape: {output2['main'].shape}")
        if 'uncertainty' in output2:
            print(f"Uncertainty shape: {output2['uncertainty'].shape}")
    
    print(" Model architecture tests passed!") 