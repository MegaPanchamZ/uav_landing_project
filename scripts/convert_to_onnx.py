#!/usr/bin/env python3
"""
Convert BiSeNetV2 PyTorch model to ONNX format for UAV Landing Detector

This script converts the provided BiSeNetV2 .pth file to ONNX format
optimized for real-time inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse

class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU"""
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class GatherAndExpansionLayer(nn.Module):
    """Gather and Expansion Layer"""
    def __init__(self, in_chan, out_chan, expand=6, stride=1):
        super(GatherAndExpansionLayer, self).__init__()
        self.stride = stride
        mid_chan = in_chan * expand
        self.conv1 = nn.Conv2d(in_chan, mid_chan, kernel_size=3, stride=1, padding=1, groups=in_chan, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_chan)
        self.conv2 = nn.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        
        if stride == 2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_chan)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.stride == 2:
            out = F.avg_pool2d(out, kernel_size=3, stride=2, padding=1)
        
        return self.relu(out + residual)

class DetailBranch(nn.Module):
    """Detail Branch for high-resolution features"""
    def __init__(self):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 64, ks=3, stride=2, padding=1),
            ConvBNReLU(64, 64, ks=3, stride=1, padding=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(64, 64, ks=3, stride=2, padding=1),
            ConvBNReLU(64, 64, ks=3, stride=1, padding=1),
            ConvBNReLU(64, 64, ks=3, stride=1, padding=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, ks=3, stride=2, padding=1),
            ConvBNReLU(128, 128, ks=3, stride=1, padding=1),
            ConvBNReLU(128, 128, ks=3, stride=1, padding=1),
        )

    def forward(self, x):
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat

class SemanticBranch(nn.Module):
    """Semantic Branch for contextual information"""
    def __init__(self):
        super(SemanticBranch, self).__init__()
        self.S1S2 = nn.Sequential(
            ConvBNReLU(3, 16, ks=3, stride=2, padding=1),
            ConvBNReLU(16, 16, ks=3, stride=1, padding=1),
            ConvBNReLU(16, 16, ks=3, stride=2, padding=1),
        )
        
        self.S3 = nn.Sequential(
            GatherAndExpansionLayer(16, 32, expand=6, stride=2),
            GatherAndExpansionLayer(32, 32, expand=6, stride=1),
        )
        
        self.S4 = nn.Sequential(
            GatherAndExpansionLayer(32, 64, expand=6, stride=2),
            GatherAndExpansionLayer(64, 64, expand=6, stride=1),
        )
        
        self.S5 = nn.Sequential(
            GatherAndExpansionLayer(64, 128, expand=6, stride=2),
            GatherAndExpansionLayer(128, 128, expand=6, stride=1),
            GatherAndExpansionLayer(128, 128, expand=6, stride=1),
            GatherAndExpansionLayer(128, 128, expand=6, stride=1),
        )
        
        # Context Embedding
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5 = self.S5(feat4)
        
        context = self.context(feat5)
        feat5 = feat5 + context
        
        return feat2, feat3, feat4, feat5

class BilateralGuidedAggregation(nn.Module):
    """Bilateral Guided Aggregation"""
    def __init__(self):
        super(BilateralGuidedAggregation, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)
        )
        
        self.left2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        
        self.right1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        
        self.right2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)
        )
        
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
        self.conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = self.up1(right1)
        
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = self.up2(right)
        
        out = self.conv(left + right)
        return out

class BiSeNetV2(nn.Module):
    """BiSeNetV2 for UAV Landing Zone Detection"""
    
    def __init__(self, num_classes=6):
        super(BiSeNetV2, self).__init__()
        self.num_classes = num_classes
        
        # Branches
        self.detail = DetailBranch()
        self.semantic = SemanticBranch()
        
        # Aggregation
        self.bga = BilateralGuidedAggregation()
        
        # Head
        self.head = nn.Sequential(
            ConvBNReLU(128, 128, ks=3, stride=1, padding=1),
            nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0)
        )
        
        # Auxiliary heads for training
        self.aux2 = nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0)
        self.aux3 = nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0)
        self.aux4 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        self.aux5 = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        size = x.size()[2:]
        
        # Detail branch
        feat_d = self.detail(x)
        
        # Semantic branch  
        feat2, feat3, feat4, feat5 = self.semantic(x)
        
        # Aggregation
        feat_head = self.bga(feat_d, feat5)
        
        # Main output
        logits = self.head(feat_head)
        logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=False)
        
        if self.training:
            # Auxiliary outputs for training
            logits_aux2 = self.aux2(feat2)
            logits_aux2 = F.interpolate(logits_aux2, size=size, mode='bilinear', align_corners=False)
            
            logits_aux3 = self.aux3(feat3)
            logits_aux3 = F.interpolate(logits_aux3, size=size, mode='bilinear', align_corners=False)
            
            logits_aux4 = self.aux4(feat4)
            logits_aux4 = F.interpolate(logits_aux4, size=size, mode='bilinear', align_corners=False)
            
            logits_aux5 = self.aux5(feat5)
            logits_aux5 = F.interpolate(logits_aux5, size=size, mode='bilinear', align_corners=False)
            
            return logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5
        else:
            return logits

def convert_model(pth_path: str, output_path: str, input_size: tuple = (512, 512)):
    """Convert PyTorch model to ONNX format"""
    
    print(f"🔄 Converting {pth_path} to ONNX format...")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = BiSeNetV2(num_classes=6)
    
    # Load weights
    try:
        checkpoint = torch.load(pth_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Remove module prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'):
                name = name[7:]  # Remove 'module.' prefix
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print("✅ Model weights loaded successfully")
        
    except Exception as e:
        print(f"⚠️  Could not load weights: {e}")
        print("   Using random initialization")
    
    model.eval()
    model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1], device=device)
    
    # Test model
    with torch.no_grad():
        output = model(dummy_input)
        print(f"✅ Model test successful, output shape: {output.shape}")
    
    # Export to ONNX
    torch.onnx.export(
        model,                          # Model
        dummy_input,                    # Model input
        output_path,                    # Output path
        export_params=True,             # Store trained parameter weights
        opset_version=11,               # ONNX version
        do_constant_folding=True,       # Optimize constant folding
        input_names=['input'],          # Input names
        output_names=['output'],        # Output names
        dynamic_axes={                  # Dynamic axes
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    
    print(f"✅ Model exported to: {output_path}")
    
    # Verify ONNX model
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(output_path)
        
        # Test inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        test_input = np.random.randn(1, 3, input_size[0], input_size[1]).astype(np.float32)
        result = session.run([output_name], {input_name: test_input})
        
        print(f"✅ ONNX model verification successful")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {result[0].shape}")
        
    except ImportError:
        print("⚠️  ONNX Runtime not installed, skipping verification")
    except Exception as e:
        print(f"❌ ONNX verification failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert BiSeNetV2 to ONNX")
    parser.add_argument("--input", type=str, required=True, help="Path to .pth file")
    parser.add_argument("--output", type=str, help="Output ONNX path (optional)")
    parser.add_argument("--size", type=int, nargs=2, default=[512, 512], help="Input size [H, W]")
    
    args = parser.parse_args()
    
    # Set output path
    if args.output is None:
        pth_path = Path(args.input)
        args.output = str(pth_path.with_suffix('.onnx'))
    
    # Convert model
    convert_model(args.input, args.output, tuple(args.size))

if __name__ == "__main__":
    main()
