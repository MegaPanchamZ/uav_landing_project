#!/usr/bin/env python3
"""
End-to-End UAV Landing System Demo with UDD6 Dataset

This demo showcases the complete neuro-symbolic UAV landing system:
- Neural network segmentation (BiSeNetV2)
- Symbolic reasoning (Scallop)
- TensorRT/CUDA/CPU acceleration
- Real UAV dataset processing
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from uav_landing_detector import UAVLandingDetector
from enhanced_uav_detector import EnhancedUAVDetector
from scallop_reasoning_engine import ScallopReasoningEngine

def find_best_udd6_image():
    """Find a good UDD6 validation image for demo"""
    udd6_path = Path("/home/mpz/development/playground/datasets/UDD/UDD/UDD6/val/src")
    
    # Look for interesting images (preferably DJI drone images)
    candidates = [
        "DJI_0031.JPG",      # Good aerial view
        "DJI_0291.JPG",      # Different perspective  
        "DJI_0373_heda.JPG", # Marked as special (heda)
        "DJI_0499_heda.JPG", # Another special one
        "000061.JPG",        # Standard numbered image
    ]
    
    for candidate in candidates:
        if (udd6_path / candidate).exists():
            return udd6_path / candidate
    
    # Fallback to first available image
    jpg_files = list(udd6_path.glob("*.JPG"))
    if jpg_files:
        return jpg_files[0]
    
    raise FileNotFoundError("No UDD6 validation images found!")

def load_and_preprocess_image(image_path):
    """Load and preprocess UDD6 image for UAV landing detection"""
    print(f"📸 Loading image: {image_path.name}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB for processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"   Original size: {image_rgb.shape}")
    print(f"   File size: {image_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    return image_rgb

def run_neural_analysis(image, detector):
    """Run neural network analysis"""
    print(f"\n🧠 Neural Network Analysis")
    print("=" * 30)
    
    start_time = time.time()
    
    # Process with UAV detector
    result = detector.process_frame(image, altitude=25.0)
    
    processing_time = time.time() - start_time
    
    print(f"   Status: {result.status}")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   Processing time: {processing_time:.3f}s")
    print(f"   FPS: {result.fps:.1f}")
    print(f"   Device: {detector.actual_device}")
    
    if result.target_pixel:
        print(f"   Target pixel: {result.target_pixel}")
        print(f"   Target world: {result.target_world}")
    
    return result

def run_neuro_symbolic_analysis(image, enhanced_detector):
    """Run neuro-symbolic analysis with Scallop reasoning"""
    print(f"\n🔗 Neuro-Symbolic Analysis") 
    print("=" * 32)
    
    start_time = time.time()
    
    # Process with enhanced detector (includes Scallop reasoning)
    result = enhanced_detector.process_frame(image, altitude=25.0)
    
    processing_time = time.time() - start_time
    
    print(f"   Status: {result.status}")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   Processing time: {processing_time:.3f}s") 
    print(f"   Scallop available: {enhanced_detector.scallop_available}")
    
    if hasattr(enhanced_detector, 'scallop_engine') and enhanced_detector.scallop_engine:
        print(f"   Context: {enhanced_detector.scallop_engine.context}")
    
    # Get reasoning explanation
    explanation = enhanced_detector.get_reasoning_explanation()
    print(f"   Reasoning engine: {explanation.get('reasoning_engine', 'N/A')}")
    print(f"   Use Scallop: {explanation.get('use_scallop', False)}")
    
    return result

def create_visualization(image, neural_result, neuro_symbolic_result, save_path=None):
    """Create comprehensive visualization of results"""
    print(f"\n📊 Creating Visualization")
    print("=" * 25)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("End-to-End UAV Landing System Demo - UDD6 Dataset", fontsize=16)
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original UAV Image (UDD6)")
    axes[0, 0].axis('off')
    
    # Neural network result
    axes[0, 1].imshow(image)
    if neural_result.target_pixel:
        # Draw target location
        circle = patches.Circle(neural_result.target_pixel, radius=20, 
                              color='red', fill=False, linewidth=3)
        axes[0, 1].add_patch(circle)
        axes[0, 1].plot(neural_result.target_pixel[0], neural_result.target_pixel[1], 
                       'rx', markersize=15, markeredgewidth=3)
    
    axes[0, 1].set_title(f"Neural Analysis\nStatus: {neural_result.status}\nConf: {neural_result.confidence:.3f}")
    axes[0, 1].axis('off')
    
    # Neuro-symbolic result
    axes[1, 0].imshow(image)
    if neuro_symbolic_result.target_pixel:
        # Draw target location with different color
        circle = patches.Circle(neuro_symbolic_result.target_pixel, radius=20,
                              color='green', fill=False, linewidth=3)
        axes[1, 0].add_patch(circle)
        axes[1, 0].plot(neuro_symbolic_result.target_pixel[0], neuro_symbolic_result.target_pixel[1],
                       'go', markersize=10, markeredgewidth=2)
    
    axes[1, 0].set_title(f"Neuro-Symbolic Analysis\nStatus: {neuro_symbolic_result.status}\nConf: {neuro_symbolic_result.confidence:.3f}")
    axes[1, 0].axis('off')
    
    # System performance comparison
    axes[1, 1].axis('off')
    performance_text = f"""
System Performance Summary:

Neural Network (BiSeNetV2):
• Status: {neural_result.status}
• Confidence: {neural_result.confidence:.3f}
• FPS: {neural_result.fps:.1f}

Neuro-Symbolic (+ Scallop):
• Status: {neuro_symbolic_result.status}  
• Confidence: {neuro_symbolic_result.confidence:.3f}
• Symbolic reasoning: Real Scallop v0.2.5

Hardware:
• Acceleration: TensorRT → CUDA → CPU
• Model: BiSeNetV2 ONNX
• Dataset: UDD6 (Urban Drone Dataset)

Legend:
🔴 Neural network target
🟢 Neuro-symbolic target
    """
    
    axes[1, 1].text(0.05, 0.95, performance_text, transform=axes[1, 1].transAxes,
                    verticalalignment='top', fontfamily='monospace', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Saved visualization: {save_path}")
    
    return fig

def main():
    """Main demo function"""
    print("🚁 UAV Landing System - End-to-End Demo")
    print("=" * 45)
    print("Dataset: UDD6 (Urban Drone Dataset)")
    print("System: Neuro-Symbolic UAV Landing Detection")
    print()
    
    try:
        # 1. Find and load UDD6 image
        image_path = find_best_udd6_image()
        image = load_and_preprocess_image(image_path)
        
        # 2. Initialize detectors
        print(f"\n🔧 Initializing Systems")
        print("=" * 25)
        
        print("Initializing neural network detector...")
        neural_detector = UAVLandingDetector(device='auto', input_resolution=(512, 512))
        
        print("Initializing neuro-symbolic detector...")
        enhanced_detector = EnhancedUAVDetector(device='auto', input_resolution=(512, 512))
        
        # 3. Run neural analysis
        neural_result = run_neural_analysis(image, neural_detector)
        
        # 4. Run neuro-symbolic analysis  
        neuro_symbolic_result = run_neuro_symbolic_analysis(image, enhanced_detector)
        
        # 5. Create visualization
        output_path = Path("uav_demo_results.png")
        fig = create_visualization(image, neural_result, neuro_symbolic_result, output_path)
        
        # 6. Final summary
        print(f"\n Demo Summary")
        print("=" * 15)
        print(f"   Image: {image_path.name}")
        print(f"   Neural result: {neural_result.status} ({neural_result.confidence:.3f})")
        print(f"   Neuro-symbolic result: {neuro_symbolic_result.status} ({neuro_symbolic_result.confidence:.3f})")
        print(f"   Visualization saved: {output_path}")
        print()
        print(" End-to-end demo completed successfully!")
        
        # Show the plot
        plt.show()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
