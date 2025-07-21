#!/usr/bin/env python3
"""
MMSegmentation-Compatible BiSeNetV2 for UAV Landing Detection
=============================================================

This implementation matches the structure of MMSegmentation BiSeNetV2 models
to enable proper loading of pretrained Cityscapes weights.

Architecture matches:
- backbone.detail.* (detail branch)
- backbone.semantic.* (semantic branch)
- decode_head.* (decoder head)
- auxiliary_head.* (auxiliary heads)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


class ConvBNReLU(nn.Module):
    """Conv-BN-ReLU block."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DetailBranch(nn.Module):
    """Detail branch for capturing fine-grained details."""
    
    def __init__(self):
        super().__init__()
        
        # Stage 1
        self.detail_branch = nn.ModuleList([
            nn.ModuleList([
                ConvBNReLU(3, 64, 3, 2, 1),
                ConvBNReLU(64, 64, 3, 1, 1),
            ]),
            nn.ModuleList([
                ConvBNReLU(64, 64, 3, 2, 1),
                ConvBNReLU(64, 64, 3, 1, 1),
                ConvBNReLU(64, 64, 3, 1, 1),
            ]),
            nn.ModuleList([
                ConvBNReLU(64, 128, 3, 2, 1),
                ConvBNReLU(128, 128, 3, 1, 1),
                ConvBNReLU(128, 128, 3, 1, 1),
            ])
        ])
    
    def forward(self, x):
        for stage in self.detail_branch:
            for layer in stage:
                x = layer(x)
        return x


class StemBlock(nn.Module):
    """Stem block for semantic branch."""
    
    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()
        self.conv_3x3 = ConvBNReLU(in_channels, out_channels, 3, 2, 1)
        self.conv_1x1 = ConvBNReLU(in_channels, out_channels//2, 1, 1, 0)
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.conv_after_pool = ConvBNReLU(out_channels//2, out_channels//2, 3, 1, 1)
        # Add 1x1 conv to reduce concatenated channels to desired output
        self.final_conv = nn.Conv2d(out_channels + out_channels//2, out_channels, 1, 1, 0)
        
    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)
        conv_1x1 = self.conv_1x1(x)
        pool = self.pool(conv_1x1)
        conv_after_pool = self.conv_after_pool(pool)
        concat = torch.cat([conv_after_pool, conv_3x3], dim=1)
        # concat is 8 + 16 = 24 channels, reduce to 16 channels
        output = self.final_conv(concat)
        return output


class GatherAndExpansionLayer(nn.Module):
    """Gather-and-Expansion layer."""
    
    def __init__(self, in_channels, out_channels, expansion=6, stride=1):
        super().__init__()
        self.stride = stride
        mid_channels = in_channels * expansion
        
        self.conv1 = ConvBNReLU(in_channels, in_channels, 3, 1, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.dwconv(out)
        out = self.conv2(out)
        
        if self.shortcut is not None:
            identity = self.shortcut(x)
        
        return F.relu(out + identity, inplace=True)


class ContextEmbeddingBlock(nn.Module):
    """Context Embedding block."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv_gap = ConvBNReLU(in_channels, out_channels, 1, 1, 0)
        self.conv_last = ConvBNReLU(out_channels, out_channels, 3, 1, 1)
    
    def forward(self, x):
        identity = x
        gap = self.gap(x)
        gap = self.conv_gap(gap)
        gap = F.interpolate(gap, size=x.shape[2:], mode='bilinear', align_corners=False)
        out = identity + gap
        out = self.conv_last(out)
        return out


class SemanticBranch(nn.Module):
    """Semantic branch for capturing contextual information."""
    
    def __init__(self):
        super().__init__()
        
        # Stem
        self.stem = StemBlock(3, 16)
        
        # Stages - Fixed channel dimensions to match MMSeg pretrained
        self.stage3 = nn.Sequential(
            GatherAndExpansionLayer(16, 32, stride=2),
            GatherAndExpansionLayer(32, 32),
        )
        
        self.stage4 = nn.Sequential(
            GatherAndExpansionLayer(32, 64, stride=2),
            GatherAndExpansionLayer(64, 64),
        )
        
        self.stage5 = nn.Sequential(
            GatherAndExpansionLayer(64, 128, stride=2),
            GatherAndExpansionLayer(128, 128),
            GatherAndExpansionLayer(128, 128),
            GatherAndExpansionLayer(128, 128),
        )
        
        # Context Embedding - Match MMSeg output channels
        self.stage4_CEBlock = ContextEmbeddingBlock(64, 64)
        self.stage5_CEBlock = ContextEmbeddingBlock(128, 128)
    
    def forward(self, x):
        outputs = {}
        
        # Stem
        x = self.stem(x)
        
        # Stage 3
        x = self.stage3(x)
        outputs['stage3'] = x
        
        # Stage 4
        x = self.stage4(x)
        x = self.stage4_CEBlock(x)
        outputs['stage4'] = x
        
        # Stage 5
        x = self.stage5(x)
        x = self.stage5_CEBlock(x)
        outputs['stage5'] = x
        
        return outputs


class BiSeNetV2Backbone(nn.Module):
    """BiSeNetV2 backbone with detail and semantic branches."""
    
    def __init__(self):
        super().__init__()
        self.detail = DetailBranch()
        self.semantic = SemanticBranch()
    
    def forward(self, x):
        # Detail branch (1/8 resolution)
        detail_out = self.detail(x)
        
        # Semantic branch (multiple scales)
        semantic_outs = self.semantic(x)
        
        return {
            'detail': detail_out,
            'semantic_stage3': semantic_outs['stage3'],  # 1/8
            'semantic_stage4': semantic_outs['stage4'],  # 1/16
            'semantic_stage5': semantic_outs['stage5'],  # 1/32
        }


class BilateralGuidedAggregationLayer(nn.Module):
    """Bilateral Guided Aggregation Layer."""
    
    def __init__(self, detail_channels=128, semantic_channels=64, out_channels=1024):
        super().__init__()
        
        # Detail branch processing
        self.detail_dwconv = nn.Sequential(
            nn.Conv2d(detail_channels, detail_channels, 3, 1, 1, groups=detail_channels, bias=False),
            nn.BatchNorm2d(detail_channels),
        )
        self.detail_conv = nn.Sequential(
            nn.Conv2d(detail_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        # Semantic branch processing
        self.semantic_conv = nn.Sequential(
            nn.Conv2d(semantic_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.semantic_dwconv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        # Aggregation
        self.conv_out = ConvBNReLU(out_channels, out_channels, 3, 1, 1)
    
    def forward(self, detail_feat, semantic_feat):
        # Ensure same spatial size
        if detail_feat.shape[2:] != semantic_feat.shape[2:]:
            semantic_feat = F.interpolate(
                semantic_feat, size=detail_feat.shape[2:], 
                mode='bilinear', align_corners=False
            )
        
        # Detail processing
        detail_dwconv = self.detail_dwconv(detail_feat)
        detail_conv = self.detail_conv(detail_dwconv)
        
        # Semantic processing  
        semantic_conv = self.semantic_conv(semantic_feat)
        semantic_dwconv = self.semantic_dwconv(semantic_conv)
        
        # Element-wise sum and activation
        aggregated = F.relu(detail_conv + semantic_dwconv, inplace=True)
        aggregated = F.relu(aggregated + semantic_conv, inplace=True)
        
        # Final conv
        out = self.conv_out(aggregated)
        return out


class SegmentationHead(nn.Module):
    """Segmentation head."""
    
    def __init__(self, in_channels, num_classes, dropout_ratio=0.1):
        super().__init__()
        mid_channels = in_channels
        
        self.convs = nn.ModuleList([
            ConvBNReLU(in_channels, mid_channels, 3, 1, 1)
        ])
        
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
            
        self.conv_seg = nn.Conv2d(mid_channels, num_classes, 1, 1, 0)
    
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
            
        output = self.conv_seg(x)
        return output


class MMSegBiSeNetV2(nn.Module):
    """
    MMSegmentation-compatible BiSeNetV2 for UAV landing detection.
    
    This implementation matches the structure of pretrained MMSeg BiSeNetV2 models
    to enable proper weight loading and adaptation.
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 3,
        dropout_ratio: float = 0.1,
        uncertainty_estimation: bool = False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.uncertainty_estimation = uncertainty_estimation
        
        # Backbone
        self.backbone = BiSeNetV2Backbone()
        
        # Bilateral Guided Aggregation - outputs 1024 channels to match MMSeg
        self.bga = BilateralGuidedAggregationLayer(
            detail_channels=128,
            semantic_channels=64,  # stage4 output 
            out_channels=1024      # Match MMSeg decode_head input
        )
        
        # Main segmentation head - 1024 input channels to match MMSeg
        self.decode_head = SegmentationHead(
            in_channels=1024,      # Match MMSeg: decode_head.conv_seg expects 1024
            num_classes=num_classes,
            dropout_ratio=dropout_ratio
        )
        
        # Feature transformation layers to match MMSeg auxiliary head inputs
        self.stage3_transform = nn.Conv2d(32, 16, 1, 1, 0)    # 32→16 for auxiliary_head.0
        self.stage5_transform = nn.Conv2d(128, 256, 1, 1, 0)  # 128→256 for auxiliary_head.2
        
        # Auxiliary heads - Match exact channel dimensions from MMSeg inspection
        self.auxiliary_head = nn.ModuleList([
            SegmentationHead(16, num_classes, dropout_ratio),   # stage3→16: auxiliary_head.0.conv_seg
            SegmentationHead(64, num_classes, dropout_ratio),   # stage4→64: auxiliary_head.1.conv_seg  
            SegmentationHead(256, num_classes, dropout_ratio),  # stage5→256: auxiliary_head.2.conv_seg
            SegmentationHead(1024, num_classes, dropout_ratio), # aggregated→1024: auxiliary_head.3.conv_seg
        ])
        
        # Uncertainty head (if enabled)
        if uncertainty_estimation:
            self.uncertainty_head = SegmentationHead(
                in_channels=1024,  # Use aggregated features
                num_classes=1,
                dropout_ratio=dropout_ratio
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        input_size = x.shape[2:]
        
        # Backbone
        backbone_outs = self.backbone(x)
        
        detail_feat = backbone_outs['detail']          # 1/8
        semantic_stage3 = backbone_outs['semantic_stage3']  # 1/8
        semantic_stage4 = backbone_outs['semantic_stage4']  # 1/16
        semantic_stage5 = backbone_outs['semantic_stage5']  # 1/32
        
        # Bilateral Guided Aggregation
        # Use semantic_stage4 as the main semantic feature
        aggregated = self.bga(detail_feat, semantic_stage4)
        
        # Main segmentation output
        main_out = self.decode_head(aggregated)
        main_out = F.interpolate(main_out, size=input_size, mode='bilinear', align_corners=False)
        
        outputs = {'main': main_out}
        
        # Auxiliary outputs (for training) - Transform features to match MMSeg channel expectations
        if self.training:
            aux_outs = []
            
            # Transform semantic features to match MMSeg auxiliary head inputs
            stage3_16 = self.stage3_transform(semantic_stage3)   # 32→16 for auxiliary_head.0
            stage4_64 = semantic_stage4                          # 64 channels - direct use for auxiliary_head.1
            stage5_256 = self.stage5_transform(semantic_stage5)  # 128→256 for auxiliary_head.2
            aggregated_1024 = aggregated                         # 1024 channels - direct use for auxiliary_head.3
            
            features = [stage3_16, stage4_64, stage5_256, aggregated_1024]
            
            for i, (aux_head, feat) in enumerate(zip(self.auxiliary_head, features)):
                aux_out = aux_head(feat)
                aux_out = F.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=False)
                aux_outs.append(aux_out)
            
            outputs['aux'] = aux_outs
        
        # Uncertainty output
        if self.uncertainty_estimation:
            uncertainty_out = self.uncertainty_head(aggregated)
            uncertainty_out = F.interpolate(uncertainty_out, size=input_size, mode='bilinear', align_corners=False)
            uncertainty_out = torch.sigmoid(uncertainty_out)
            outputs['uncertainty'] = uncertainty_out
        
        return outputs
    
    def load_pretrained_weights(self, pretrained_path: str, num_classes_pretrained: int = 19):
        """
        Load pretrained weights and adapt classifier layers.
        
        Args:
            pretrained_path: Path to pretrained model
            num_classes_pretrained: Number of classes in pretrained model (19 for Cityscapes)
        """
        print(f"🔄 Loading MMSeg BiSeNetV2 weights from: {pretrained_path}")
        
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            pretrained_state = checkpoint['state_dict']
        else:
            pretrained_state = checkpoint
        
        # Get current model state
        model_state = self.state_dict()
        
        # Load compatible weights
        loaded_layers = []
        adapted_layers = []
        skipped_layers = []
        
        for name, param in model_state.items():
            if name in pretrained_state:
                pretrained_param = pretrained_state[name]
                
                if param.shape == pretrained_param.shape:
                    # Direct loading
                    model_state[name] = pretrained_param
                    loaded_layers.append(name)
                    
                elif 'conv_seg' in name and len(param.shape) == 4:
                    # Adapt classifier layers from 19 to target classes
                    if param.shape[0] <= pretrained_param.shape[0]:
                        # Use subset of pretrained classes
                        adapted_param = pretrained_param[:param.shape[0]]
                        model_state[name] = adapted_param
                        adapted_layers.append(f"{name} ({pretrained_param.shape}→{param.shape})")
                    else:
                        # Initialize with pretrained + random for new classes
                        adapted_param = param.clone()
                        adapted_param[:pretrained_param.shape[0]] = pretrained_param
                        model_state[name] = adapted_param
                        adapted_layers.append(f"{name} ({pretrained_param.shape}→{param.shape})")
                        
                elif 'conv_seg' in name and len(param.shape) == 1:
                    # Adapt bias
                    if param.shape[0] <= pretrained_param.shape[0]:
                        adapted_param = pretrained_param[:param.shape[0]]
                        model_state[name] = adapted_param
                        adapted_layers.append(f"{name} ({pretrained_param.shape}→{param.shape})")
                    else:
                        adapted_param = param.clone()
                        adapted_param[:pretrained_param.shape[0]] = pretrained_param
                        model_state[name] = adapted_param
                        adapted_layers.append(f"{name} ({pretrained_param.shape}→{param.shape})")
                else:
                    skipped_layers.append(f"{name} (shape mismatch: {pretrained_param.shape} vs {param.shape})")
            else:
                skipped_layers.append(f"{name} (not found in pretrained)")
        
        # Load adapted state dict
        self.load_state_dict(model_state, strict=False)
        
        # Print summary
        print(f"✅ Pretrained weight loading completed:")
        print(f"   Loaded layers: {len(loaded_layers)}")
        print(f"   Adapted layers: {len(adapted_layers)}")
        print(f"   Skipped layers: {len(skipped_layers)}")
        
        if adapted_layers:
            print(f"🔧 Adapted classifier layers:")
            for layer in adapted_layers:
                print(f"   {layer}")
        
        return {
            'loaded_layers': loaded_layers,
            'adapted_layers': adapted_layers,
            'skipped_layers': skipped_layers
        }


def create_mmseg_bisenetv2(
    num_classes: int = 4,
    uncertainty_estimation: bool = True,
    pretrained_path: Optional[str] = None
) -> MMSegBiSeNetV2:
    """
    Create MMSeg-compatible BiSeNetV2 model.
    
    Args:
        num_classes: Number of output classes
        uncertainty_estimation: Enable uncertainty quantification
        pretrained_path: Path to pretrained Cityscapes model
        
    Returns:
        MMSegBiSeNetV2 model
    """
    model = MMSegBiSeNetV2(
        num_classes=num_classes,
        uncertainty_estimation=uncertainty_estimation
    )
    
    if pretrained_path:
        model.load_pretrained_weights(pretrained_path)
    
    # Calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🏗️ Created MMSeg BiSeNetV2:")
    print(f"   Parameters: {total_params:,} ({trainable_params:,} trainable)")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"   Uncertainty: {uncertainty_estimation}")
    
    return model


if __name__ == "__main__":
    # Test the model
    model = create_mmseg_bisenetv2(
        num_classes=4,
        uncertainty_estimation=True
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        outputs = model(x)
    
    print(f"\n🧪 Forward pass test:")
    print(f"   Input shape: {x.shape}")
    print(f"   Main output: {outputs['main'].shape}")
    if 'uncertainty' in outputs:
        print(f"   Uncertainty: {outputs['uncertainty'].shape}")
    if 'aux' in outputs:
        print(f"   Auxiliary outputs: {len(outputs['aux'])}")
        for i, aux in enumerate(outputs['aux']):
            print(f"     Aux {i}: {aux.shape}")
    
    print("✅ MMSeg BiSeNetV2 test completed!") 