#!/usr/bin/env python3
"""
Test fine-tuned model and visualize semantic segmentation output - Simple Version
"""
import sys
import os
sys.path.append('src')

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from pathlib import Path

from uav_landing_detector import UAVLandingDetector

def create_realistic_test_images():
    """Create realistic test scenarios."""
    scenarios = []
    
    # Scenario 1: Clear landing field
    print("🌾 Creating test scenario 1: Clear landing field")
    field = np.zeros((480, 640, 3), dtype=np.uint8)
    field[:, :] = [100, 130, 90]  # Grass
    
    # Add landing pad
    cv2.rectangle(field, (250, 180), (390, 300), (180, 180, 160), -1)
    cv2.circle(field, (320, 240), 10, (255, 100, 100), -1)  # Target marker
    
    scenarios.append(("Clear Landing Field", field))
    
    # Scenario 2: Urban environment
    print("🏢 Creating test scenario 2: Urban with obstacles")
    urban = np.zeros((480, 640, 3), dtype=np.uint8)
    urban[:, :] = [90, 95, 100]  # Concrete base
    
    # Buildings
    cv2.rectangle(urban, (50, 50), (180, 200), (70, 70, 80), -1)
    cv2.rectangle(urban, (450, 100), (590, 220), (60, 60, 70), -1)
    
    # Road
    cv2.rectangle(urban, (200, 350), (440, 400), (50, 50, 60), -1)
    
    # Clear area for landing
    cv2.rectangle(urban, (280, 200), (380, 300), (120, 115, 105), -1)
    
    scenarios.append(("Urban Environment", urban))
    
    # Scenario 3: Mixed terrain
    print("🌲 Creating test scenario 3: Mixed natural terrain")
    mixed = np.zeros((480, 640, 3), dtype=np.uint8)
    mixed[:, :] = [95, 120, 85]  # Ground
    
    # Trees
    cv2.circle(mixed, (100, 150), 25, (40, 80, 40), -1)
    cv2.circle(mixed, (500, 120), 30, (35, 75, 35), -1)
    cv2.circle(mixed, (150, 350), 20, (45, 85, 45), -1)
    
    # Water
    cv2.ellipse(mixed, (450, 350), (70, 35), 0, 0, 360, (80, 120, 160), -1)
    
    # Sandy landing area
    cv2.ellipse(mixed, (320, 250), (50, 40), 0, 0, 360, (140, 130, 110), -1)
    
    scenarios.append(("Mixed Terrain", mixed))
    
    return scenarios

def simple_visualize_segmentation(image, segmentation_mask, title, save_path=None):
    """Simple segmentation visualization."""
    
    # Create color map for different classes
    colors = np.array([
        [0, 0, 0],       # Background - black
        [0, 255, 0],     # Suitable landing - green  
        [255, 255, 0],   # Marginal - yellow
        [255, 0, 0],     # Unsuitable - red
        [0, 0, 255],     # Obstacle - blue
        [255, 0, 255],   # Water - magenta
    ])
    
    # Create colored segmentation
    colored_seg = colors[segmentation_mask % len(colors)]
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Segmentation Analysis: {title}', fontsize=14, fontweight='bold')
    
    # Original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Segmentation
    axes[1].imshow(colored_seg)
    axes[1].set_title('Segmentation Result')
    axes[1].axis('off')
    
    # Overlay
    overlay = cv2.addWeighted(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 0.6,
        colored_seg.astype(np.uint8), 0.4, 0
    )
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
    
    plt.show()

def test_fine_tuned_segmentation():
    """Test the fine-tuned model with segmentation output visualization."""
    print("🚁 Testing Fine-tuned UAV Landing Model - Segmentation Output")
    print("=" * 70)
    
    # Check model exists
    model_path = "trained_models/ultra_fast_uav_landing.onnx"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    print(f"📁 Loading model: {model_path}")
    
    # Initialize detector
    detector = UAVLandingDetector(
        model_path=model_path,
        enable_visualization=True
    )
    
    # Create test scenarios
    scenarios = create_realistic_test_images()
    
    print(f"\n🧪 Testing {len(scenarios)} scenarios with segmentation analysis...")
    
    all_results = []
    
    for i, (scenario_name, test_image) in enumerate(scenarios, 1):
        print(f"\n{'='*50}")
        print(f" Scenario {i}: {scenario_name}")
        print(f"{'='*50}")
        
        # Process the image
        print("🔄 Running inference...")
        start_time = time.time()
        result = detector.process_frame(test_image, altitude=5.0)
        processing_time = (time.time() - start_time) * 1000
        
        # Print results
        print(f" Processing completed!")
        print(f"   ⏱️  Total time: {processing_time:.1f}ms")
        print(f"    Status: {result.status}")
        print(f"   📊 Confidence: {result.confidence:.3f}")
        print(f"   📈 FPS: {result.fps:.1f}")
        print(f"   🛬 Landing phase: {detector.landing_phase}")
        
        if result.target_pixel:
            print(f"   📍 Target pixel: {result.target_pixel}")
            print(f"   📏 Distance: {result.distance:.2f}m")
            print(f"   🎮 Commands: F={result.forward_velocity:.3f}m/s, R={result.right_velocity:.3f}m/s, D={result.descent_rate:.3f}m/s")
        
        # Get segmentation data
        seg_mask, raw_output, confidence_map = detector.get_segmentation_data()
        
        if seg_mask is not None:
            print(f"   🎨 Segmentation shape: {seg_mask.shape}")
            print(f"   🎨 Classes detected: {np.unique(seg_mask)}")
            print(f"   🎨 Class distribution:")
            
            for class_id in np.unique(seg_mask):
                pixel_count = np.sum(seg_mask == class_id)
                percentage = (pixel_count / seg_mask.size) * 100
                class_names = ["Background", "Suitable", "Marginal", "Unsuitable", "Obstacle", "Water"]
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                print(f"      Class {class_id} ({class_name}): {pixel_count} pixels ({percentage:.1f}%)")
            
            # Visualize
            save_path = f"segmentation_{scenario_name.lower().replace(' ', '_')}.png"
            simple_visualize_segmentation(test_image, seg_mask, scenario_name, save_path)
            
        else:
            print(f"   ❌ No segmentation data available (placeholder mode)")
        
        # Store results
        all_results.append({
            'scenario': scenario_name,
            'processing_time': processing_time,
            'status': result.status,
            'confidence': result.confidence,
            'has_target': result.target_pixel is not None,
            'segmentation_available': seg_mask is not None
        })
    
    # Final summary
    print(f"\n{'='*70}")
    print("📊 SEGMENTATION TEST SUMMARY")
    print(f"{'='*70}")
    
    total_time = sum(r['processing_time'] for r in all_results)
    avg_time = total_time / len(all_results)
    target_detection_rate = sum(1 for r in all_results if r['has_target']) / len(all_results) * 100
    
    print(f"🏆 Overall Performance:")
    print(f"   Total scenarios tested: {len(all_results)}")
    print(f"   Average processing time: {avg_time:.1f}ms")
    print(f"   Average FPS: {1000/avg_time:.1f}")
    print(f"   Target detection rate: {target_detection_rate:.0f}%")
    print(f"   Model size: {os.path.getsize(model_path) / 1024**2:.1f} MB")
    print(f"   Real-time capable: {' Yes' if avg_time < 33 else '❌ No'}")
    
    print(f"\n📊 Individual Results:")
    for result in all_results:
        status_icon = "✅" if result['has_target'] else "⚠️"
        seg_icon = "🎨" if result['segmentation_available'] else "❌"
        print(f"   {status_icon} {seg_icon} {result['scenario']:<20} | {result['processing_time']:6.1f}ms | {result['status']}")
    
    print(f"\n💾 Generated Files:")
    for i, (scenario_name, _) in enumerate(scenarios, 1):
        filename = f"segmentation_{scenario_name.lower().replace(' ', '_')}.png"
        if Path(filename).exists():
            print(f"   {i}. {filename}")
    
    print(f"\n🎉 Fine-tuned model segmentation testing complete!")
    return True

if __name__ == "__main__":
    success = test_fine_tuned_segmentation()
    if success:
        print("\n All tests completed successfully!")
    else:
        print("\n❌ Some tests failed!")
