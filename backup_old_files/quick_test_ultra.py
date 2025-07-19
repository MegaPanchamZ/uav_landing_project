#!/usr/bin/env python3
"""
Quick test of ultra-fast trained model
"""
import torch
import numpy as np
from PIL import Image
import cv2
import time
from pathlib import Path

class UltraFastBiSeNet(torch.nn.Module):
    """Ultra-lightweight BiSeNet for fast training."""
    
    def __init__(self, num_classes=7):
        super().__init__()
        
        # Much smaller network for speed
        self.backbone = torch.nn.Sequential(
            # Initial conv
            torch.nn.Conv2d(3, 32, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            
            # Downsample 1 (128x128)
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            
            # Downsample 2 (64x64)
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            
            # Feature processing
            torch.nn.Conv2d(128, 128, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
        )
        
        # Simple upsampling path
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.Conv2d(64, 32, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        )
        
        self.classifier = torch.nn.Conv2d(32, num_classes, 1)
    
    def forward(self, x):
        # Encode
        features = self.backbone(x)
        
        # Decode
        x = self.decoder(features)
        
        # Upsample to match input size
        x = torch.nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        
        # Classify
        return self.classifier(x)

def test_ultra_model():
    """Test the ultra-fast model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Testing ultra-fast model on {device}")
    
    # Load model
    model = UltraFastBiSeNet(4)  # Landing classes
    model = model.to(device)
    model.eval()
    
    # Find stage 2 model
    stage2_path = Path("ultra_stage2_best.pth")
    if stage2_path.exists():
        print("✅ Found stage 2 model")
        checkpoint = torch.load(stage2_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print("❌ No stage 2 model found, using random weights")
        return
    
    # Test inference speed
    print("\n⚡ Speed Test:")
    test_input = torch.randn(1, 3, 256, 256).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(test_input)
    
    # Benchmark
    times = []
    for i in range(100):
        start = time.time()
        with torch.no_grad():
            output = model(test_input)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        times.append((time.time() - start) * 1000)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"🎯 Average inference: {avg_time:.1f}±{std_time:.1f}ms")
    print(f"🎯 Throughput: {1000/avg_time:.1f} FPS")
    
    # Test with real image if available
    test_images = list(Path("../datasets/UDD/UDD/UDD6/src/train/images").glob("*.jpg"))
    if test_images:
        print(f"\n🖼️  Testing with real image: {test_images[0].name}")
        
        # Load and preprocess image
        image = Image.open(test_images[0]).convert('RGB')
        orig_size = image.size
        image = image.resize((256, 256))
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Inference
        start = time.time()
        with torch.no_grad():
            output = model(image_tensor)
        inference_time = (time.time() - start) * 1000
        
        # Get prediction
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]
        
        # Map classes for visualization
        class_colors = {
            0: (0, 0, 0),       # Background - black
            1: (0, 255, 0),     # Safe landing - green  
            2: (255, 255, 0),   # Caution - yellow
            3: (255, 0, 0),     # No landing - red
        }
        
        # Create colored prediction
        pred_colored = np.zeros((256, 256, 3), dtype=np.uint8)
        for class_id, color in class_colors.items():
            pred_colored[pred == class_id] = color
        
        print(f"⚡ Real inference: {inference_time:.1f}ms")
        print(f"📊 Class distribution: {np.bincount(pred.flatten())}")
        
        # Save result
        result_path = "ultra_test_result.png"
        
        # Combine original and prediction
        orig_resized = np.array(image)
        combined = np.hstack([orig_resized, pred_colored])
        Image.fromarray(combined).save(result_path)
        print(f"💾 Result saved: {result_path}")
    
    print(f"\n🎉 Ultra-fast model test completed!")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📊 Model size: {sum(p.numel() * 4 for p in model.parameters()) / 1024**2:.1f} MB")

if __name__ == "__main__":
    test_ultra_model()
