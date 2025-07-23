#!/usr/bin/env python3
"""
Convert ultra-fast trained model to ONNX
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class UltraFastBiSeNet(nn.Module):
    """Ultra-lightweight BiSeNet for fast training."""
    
    def __init__(self, num_classes=7):
        super().__init__()
        
        # Much smaller network for speed
        self.backbone = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Downsample 1 (128x128)
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Downsample 2 (64x64)
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Feature processing
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Simple upsampling path
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Conv2d(32, num_classes, 1)
    
    def forward(self, x):
        # Encode
        features = self.backbone(x)
        
        # Decode
        x = self.decoder(features)
        
        # Upsample to match input size
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        
        # Classify
        return self.classifier(x)

def convert_ultra_to_onnx():
    """Convert the ultra-fast model to ONNX"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Converting ultra-fast model on {device}")
    
    # Load stage 2 model (final fine-tuned)
    model = UltraFastBiSeNet(4)  # 4 landing classes
    model = model.to(device)
    
    stage2_path = Path("ultra_stage2_best.pth")
    if not stage2_path.exists():
        print("❌ Stage 2 model not found!")
        return
    
    checkpoint = torch.load(stage2_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(" Model loaded successfully")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    # Export to ONNX
    output_path = "models/ultra_fast_uav_landing.onnx"
    Path("models").mkdir(exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f" ONNX model saved: {output_path}")
    
    # Test ONNX model
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(output_path)
        
        # Test inference
        test_input = torch.randn(1, 3, 256, 256).numpy()
        result = session.run(None, {'input': test_input})
        
        print(f" ONNX model verified!")
        print(f"📊 Output shape: {result[0].shape}")
        
        # Speed test
        import time
        import numpy as np
        
        times = []
        for _ in range(100):
            start = time.time()
            _ = session.run(None, {'input': test_input})
            times.append((time.time() - start) * 1000)
        
        avg_time = np.mean(times)
        print(f" ONNX inference: {avg_time:.1f}ms ({1000/avg_time:.1f} FPS)")
        
    except ImportError:
        print("⚠️  onnxruntime not available for verification")

if __name__ == "__main__":
    convert_ultra_to_onnx()
