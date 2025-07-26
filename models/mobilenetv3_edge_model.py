#!/usr/bin/env python3
"""
MobileNetV3-Based Edge Model for UAV Landing
===========================================

Implements the edge-optimized architecture from EDGE_OPTIMIZED_STRATEGY.md:
- MobileNetV3-Small backbone (2.5MB, optimized for mobile)
- Lightweight segmentation head for 6 landing classes
- <15-25ms inference on edge hardware
- Multi-stage progressive training compatible
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from typing import Dict, Optional, Union
import warnings

class EdgeLandingNet(nn.Module):
    """
    MobileNetV3-Small + Custom Segmentation Head for UAV Landing Detection.
    
    Architecture:
    - Backbone: MobileNetV3-Small (2.5MB, ImageNet pretrained)
    - Head: Lightweight segmentation decoder
    - Output: 6 landing classes
    - Speed: ~15-25ms inference on modern edge hardware
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        backbone_pretrained: bool = True,
        dropout: float = 0.2,
        use_uncertainty: bool = False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_uncertainty = use_uncertainty
        
        # MobileNetV3-Small backbone
        if backbone_pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self.backbone = mobilenet_v3_small(weights=weights)
        else:
            self.backbone = mobilenet_v3_small(weights=None)
        
        # Remove classifier head - we only need features
        self.backbone.classifier = nn.Identity()
        
        # Get feature dimensions
        # MobileNetV3-Small features output: [B, 576, H/16, W/16]
        backbone_out_channels = 576
        
        # Lightweight segmentation head
        self.seg_head = nn.Sequential(
            # Feature compression and upsampling
            nn.Conv2d(backbone_out_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            
            # Intermediate upsampling (16x -> 8x)
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            
            # Further upsampling (8x -> 4x)
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Final upsampling to match input (4x -> 1x)
            nn.ConvTranspose2d(32, num_classes, 4, stride=4, padding=0)
        )
        
        # Uncertainty estimation head (optional)
        if use_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(backbone_out_channels, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.Conv2d(64, 1, 1),  # Single channel uncertainty
                nn.Sigmoid()  # Uncertainty in [0, 1]
            )
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"🚁 EdgeLandingNet initialized:")
        print(f"   Backbone: MobileNetV3-Small ({'pretrained' if backbone_pretrained else 'random'})")
        print(f"   Classes: {num_classes}")
        print(f"   Uncertainty: {use_uncertainty}")
        print(f"   Parameters: {self.count_parameters():,}")
    
    def _initialize_weights(self):
        """Initialize segmentation head weights."""
        for module in [self.seg_head]:
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        if hasattr(self, 'uncertainty_head'):
            for m in self.uncertainty_head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            If use_uncertainty=False: Segmentation logits [B, num_classes, H, W]
            If use_uncertainty=True: Dict with 'main' (logits) and 'uncertainty' keys
        """
        # Extract features with MobileNetV3
        features = self.backbone.features(x)  # [B, 576, H/16, W/16]
        
        # Generate segmentation
        seg_logits = self.seg_head(features)  # [B, num_classes, H, W]
        
        if not self.use_uncertainty:
            return seg_logits
        
        # Generate uncertainty map if requested
        uncertainty = self.uncertainty_head(features)  # [B, 1, H/16, W/16]
        # Upsample uncertainty to match segmentation resolution
        uncertainty = F.interpolate(
            uncertainty, 
            size=seg_logits.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        return {
            'main': seg_logits,
            'uncertainty': uncertainty
        }
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Estimate model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)


class EnhancedEdgeLandingNet(nn.Module):
    """
    Enhanced edge-optimized landing detection network with multi-scale features.
    
    Features:
    - MobileNetV3-Small backbone (2.5MB)
    - Multi-scale feature fusion
    - Lightweight segmentation head
    - Optional uncertainty estimation
    """
    
    def __init__(
        self, 
        num_classes: int = 6, 
        backbone_pretrained: bool = True,
        dropout: float = 0.2,
        use_uncertainty: bool = False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_uncertainty = use_uncertainty
        
        # Load MobileNetV3-Small backbone
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if backbone_pretrained else None
        mobilenet = mobilenet_v3_small(weights=weights)
        
        # Extract feature extraction stages
        self.stem = nn.Sequential(*list(mobilenet.features)[:2])    # [B, 16, H/2, W/2]
        self.stage1 = nn.Sequential(*list(mobilenet.features)[2:4]) # [B, 16, H/4, W/4]  
        self.stage2 = nn.Sequential(*list(mobilenet.features)[4:7]) # [B, 24, H/8, W/8]
        self.stage3 = nn.Sequential(*list(mobilenet.features)[7:11]) # [B, 40, H/16, W/16]
        self.stage4 = nn.Sequential(*list(mobilenet.features)[11:]) # [B, 576, H/16, W/16]
        
        # Fix: Calculate actual output channels from MobileNetV3
        # Let's create a test tensor to get the actual dimensions
        test_x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            x1 = self.stem(test_x)
            x2 = self.stage1(x1) 
            x3 = self.stage2(x2)
            x4 = self.stage3(x3)
            x5 = self.stage4(x4)
            
        # Get actual channel dimensions
        stage3_channels = x3.shape[1]  # Should be 24
        stage4_channels = x4.shape[1]  # Should be 40  
        stage5_channels = x5.shape[1]  # Should be 576
        
        total_channels = stage3_channels + stage4_channels + stage5_channels
        print(f"Actual fusion channels: {stage3_channels} + {stage4_channels} + {stage5_channels} = {total_channels}")
        
        # Multi-scale feature fusion
        self.fusion_conv = nn.Conv2d(total_channels, 256, 1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(256)
        self.fusion_relu = nn.ReLU(inplace=True)
        
        # Lightweight decoder
        self.decoder = nn.Sequential(
            # Upsampling block 1: 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Upsampling block 2: 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Upsampling block 3: 64x64 -> 128x128
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Upsampling block 4: 128x128 -> 256x256
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Final prediction: 256x256 -> 512x512
            nn.ConvTranspose2d(16, num_classes, 4, stride=2, padding=1),
        )
        
        # Optional uncertainty estimation
        if use_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.Conv2d(128, 1, 1),  # Single channel for uncertainty
                nn.Sigmoid()
            )
        
        # Initialize weights
        self._init_weights()
        
        print(f"🚁 EnhancedEdgeLandingNet initialized:")
        print(f"   Multi-scale features: ✓")
        print(f"   Parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"   Estimated size: {sum(p.numel() for p in self.parameters()) * 4 / 1e6:.1f}MB")
    
    def _init_weights(self):
        """Initialize weights for custom layers."""
        for m in [self.fusion_conv, self.decoder]:
            for module in m.modules():
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with multi-scale feature fusion."""
        # Multi-scale feature extraction
        x1 = self.stem(x)         # [B, 16, H/2, W/2]
        x2 = self.stage1(x1)      # [B, 16, H/4, W/4]
        x3 = self.stage2(x2)      # [B, 24, H/8, W/8]
        x4 = self.stage3(x3)      # [B, 40, H/16, W/16]
        x5 = self.stage4(x4)      # [B, 576, H/16, W/16]
        
        # Resize low-level features to match high-level features
        x3_up = F.interpolate(x3, size=x5.shape[-2:], mode='bilinear', align_corners=False)
        x4_up = F.interpolate(x4, size=x5.shape[-2:], mode='bilinear', align_corners=False)
        
        # Feature fusion
        fused = torch.cat([x5, x4_up, x3_up], dim=1)  # [B, 576+40+24, H/16, W/16]
        fused = self.fusion_relu(self.fusion_bn(self.fusion_conv(fused)))
        
        # Decode to segmentation
        seg_logits = self.decoder(fused)
        
        if not self.use_uncertainty:
            return seg_logits
        
        # Generate uncertainty
        uncertainty = self.uncertainty_head(fused)
        uncertainty = F.interpolate(
            uncertainty, 
            size=seg_logits.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        return {
            'main': seg_logits,
            'uncertainty': uncertainty
        }
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)


def create_edge_model(
    model_type: str = "standard",
    num_classes: int = 6,
    input_size: int = 512,
    use_uncertainty: bool = False,
    pretrained: bool = True
) -> nn.Module:
    """
    Factory function to create edge-optimized models.
    
    Args:
        model_type: 'standard' or 'enhanced'
        num_classes: Number of output classes
        input_size: Input image size (not used directly but for reference)
        use_uncertainty: Whether to include uncertainty estimation
        pretrained: Use ImageNet pretrained backbone
        
    Returns:
        Edge-optimized model instance
    """
    
    if model_type == "standard":
        model = EdgeLandingNet(
            num_classes=num_classes,
            backbone_pretrained=pretrained,
            use_uncertainty=use_uncertainty
        )
    elif model_type == "enhanced":
        model = EnhancedEdgeLandingNet(
            num_classes=num_classes,
            backbone_pretrained=pretrained,
            use_uncertainty=use_uncertainty
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'standard' or 'enhanced'")
    
    return model


def benchmark_model_speed(model: nn.Module, input_size: int = 512, device: str = 'cuda'):
    """Benchmark model inference speed."""
    import time
    
    model.eval()
    model = model.to(device)
    
    # Warm up
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    num_runs = 100
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / num_runs * 1000
    fps = 1000 / avg_time_ms
    
    print(f"📊 Model Benchmark ({device}):")
    print(f"   Average inference time: {avg_time_ms:.2f}ms")
    print(f"   FPS: {fps:.1f}")
    print(f"   Input size: {input_size}x{input_size}")
    
    return avg_time_ms, fps


if __name__ == "__main__":
    # Test model creation and benchmarking
    print("🚁 Testing Edge Models...")
    
    # Test standard model
    print("\n=== Standard EdgeLandingNet ===")
    model_std = create_edge_model(
        model_type="standard",
        num_classes=6,
        use_uncertainty=True,
        pretrained=True
    )
    
    # Test enhanced model
    print("\n=== Enhanced EdgeLandingNet ===")
    model_enh = create_edge_model(
        model_type="enhanced", 
        num_classes=6,
        use_uncertainty=True,
        pretrained=True
    )
    
    # Test forward pass
    print("\n🧪 Testing forward pass...")
    dummy_input = torch.randn(2, 3, 512, 512)
    
    with torch.no_grad():
        output_std = model_std(dummy_input)
        output_enh = model_enh(dummy_input)
    
    print(f"Standard output shapes:")
    print(f"   Main: {output_std['main'].shape}")
    print(f"   Uncertainty: {output_std['uncertainty'].shape}")
    
    print(f"Enhanced output shapes:")
    print(f"   Main: {output_enh['main'].shape}")
    print(f"   Uncertainty: {output_enh['uncertainty'].shape}")
    
    # Benchmark if CUDA available
    if torch.cuda.is_available():
        print(f"\n⚡ Speed Benchmarks:")
        benchmark_model_speed(model_std, input_size=512, device='cuda')
        benchmark_model_speed(model_enh, input_size=512, device='cuda')
    else:
        print(f"\n⚠️  CUDA not available, skipping speed benchmark") 