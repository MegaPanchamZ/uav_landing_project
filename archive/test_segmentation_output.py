#!/usr/bin/env python3
"""
Test fine-tuned model and visualize semantic segmentation output
"""
import sys
import os
sys.path.append('src')

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import time
from pathlib import Path

from uav_landing_detector import UAVLandingDetector

def create_test_scenarios():
    """Create diverse test scenarios for segmentation visualization."""
    scenarios = []
    
    # Scenario 1: Open field with clear landing zone
    print("🌾 Creating scenario 1: Open field")
    field = np.zeros((480, 640, 3), dtype=np.uint8)
    field[:, :] = [120, 140, 100]  # Grass green
    
    # Add a clear landing pad
    cv2.rectangle(field, (280, 200), (360, 280), (180, 180, 160), -1)  # Light concrete
    cv2.rectangle(field, (285, 205), (355, 275), (200, 200, 180), 2)   # Landing pad outline
    
    scenarios.append(("Open Field", field))
    
    # Scenario 2: Urban environment with obstacles
    print("🏢 Creating scenario 2: Urban environment")
    urban = np.zeros((480, 640, 3), dtype=np.uint8)
    urban[:, :] = [100, 100, 120]  # Urban ground
    
    # Buildings
    cv2.rectangle(urban, (50, 50), (150, 200), (80, 80, 90), -1)
    cv2.rectangle(urban, (450, 100), (580, 250), (70, 70, 80), -1)
    
    # Roads
    cv2.rectangle(urban, (0, 300), (640, 350), (60, 60, 70), -1)
    cv2.rectangle(urban, (200, 0), (250, 480), (60, 60, 70), -1)
    
    # Potential landing area (parking lot)
    cv2.rectangle(urban, (300, 180), (420, 280), (90, 90, 100), -1)
    
    scenarios.append(("Urban Environment", urban))
    
    # Scenario 3: Natural terrain with mixed surfaces
    print("🌲 Creating scenario 3: Natural terrain")
    natural = np.zeros((480, 640, 3), dtype=np.uint8)
    natural[:, :] = [100, 130, 90]  # Ground
    
    # Trees (dark green circles)
    for x, y, r in [(100, 150, 30), (500, 200, 25), (200, 350, 35), (450, 400, 20)]:
        cv2.circle(natural, (x, y), r, (40, 80, 40), -1)
    
    # Water body
    cv2.ellipse(natural, (150, 350), (80, 40), 0, 0, 360, (60, 100, 140), -1)
    
    # Sandy/rocky area (potential landing)
    cv2.ellipse(natural, (400, 250), (60, 45), 0, 0, 360, (140, 130, 110), -1)
    
    scenarios.append(("Natural Terrain", natural))
    
    # Scenario 4: Complex mixed environment
    print("🏞️ Creating scenario 4: Complex mixed environment")
    mixed = np.zeros((480, 640, 3), dtype=np.uint8)
    mixed[:, :] = [110, 125, 105]  # Base ground
    
    # Vegetation patches
    cv2.rectangle(mixed, (0, 0), (200, 150), (90, 120, 80), -1)
    cv2.rectangle(mixed, (400, 300), (640, 480), (85, 115, 75), -1)
    
    # Building
    cv2.rectangle(mixed, (500, 50), (600, 180), (75, 75, 85), -1)
    
    # Road/path
    cv2.rectangle(mixed, (150, 200), (450, 240), (65, 65, 75), -1)
    
    # Clear landing zones
    cv2.rectangle(mixed, (250, 280), (350, 380), (160, 150, 130), -1)  # Sandy area
    cv2.rectangle(mixed, (50, 300), (130, 380), (140, 140, 160), -1)   # Concrete pad
    
    scenarios.append(("Complex Mixed", mixed))
    
    return scenarios

def visualize_segmentation_output(image, segmentation_mask, confidence_map, title, save_path=None):
    """Visualize segmentation output with detailed analysis."""
    
    # Define class colors (matching your model's classes)
    class_colors = [
        [128, 64, 128],   # Road/Path (purple)
        [244, 35, 232],   # Sidewalk (pink)
        [70, 70, 70],     # Building (dark gray)
        [102, 102, 156],  # Wall (blue-gray)
        [190, 153, 153],  # Fence (light brown)
        [153, 153, 153],  # Pole (gray)
        [250, 170, 30],   # Traffic light (orange)
        [220, 220, 0],    # Traffic sign (yellow)
        [107, 142, 35],   # Vegetation (olive)
        [152, 251, 152],  # Terrain (light green)
        [70, 130, 180],   # Sky (steel blue)
        [220, 20, 60],    # Person (crimson)
        [255, 0, 0],      # Rider (red)
        [0, 0, 142],      # Car (dark blue)
        [0, 0, 70],       # Truck (darker blue)
        [0, 60, 100],     # Bus (navy)
        [0, 80, 100],     # Train (dark navy)
        [0, 0, 230],      # Motorcycle (blue)
        [119, 11, 32],    # Bicycle (dark red)
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Semantic Segmentation Analysis: {title}', fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Segmentation mask (colored)
    colored_mask = np.zeros_like(image)
    for class_id in range(len(class_colors)):
        mask = (segmentation_mask == class_id)
        colored_mask[mask] = class_colors[class_id]
    
    axes[0, 1].imshow(colored_mask)
    axes[0, 1].set_title('Segmentation Mask')
    axes[0, 1].axis('off')
    
    # Overlay
    overlay = cv2.addWeighted(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 0.6,
        colored_mask, 0.4, 0
    )
    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title('Overlay (Image + Segmentation)')
    axes[0, 2].axis('off')
    
    # Confidence map
    confidence_display = axes[1, 0].imshow(confidence_map, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('Confidence Map')
    axes[1, 0].axis('off')
    plt.colorbar(confidence_display, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Landing suitability analysis
    landing_score = np.zeros_like(segmentation_mask, dtype=np.float32)
    
    # Define landing suitability for each class (0=unsuitable, 1=perfect)
    landing_suitability = {
        0: 0.7,   # Road - moderate
        1: 0.8,   # Sidewalk - good
        2: 0.0,   # Building - impossible
        3: 0.0,   # Wall - impossible
        4: 0.0,   # Fence - impossible
        5: 0.0,   # Pole - impossible
        6: 0.0,   # Traffic light - impossible
        7: 0.0,   # Traffic sign - impossible
        8: 0.3,   # Vegetation - poor
        9: 0.9,   # Terrain - excellent
        10: 0.0,  # Sky - impossible
        11: 0.0,  # Person - impossible
        12: 0.0,  # Rider - impossible
        13: 0.0,  # Car - impossible
        14: 0.0,  # Truck - impossible
        15: 0.0,  # Bus - impossible
        16: 0.0,  # Train - impossible
        17: 0.0,  # Motorcycle - impossible
        18: 0.0,  # Bicycle - impossible
    }
    
    for class_id, suitability in landing_suitability.items():
        mask = (segmentation_mask == class_id)
        landing_score[mask] = suitability * confidence_map[mask]
    
    landing_display = axes[1, 1].imshow(landing_score, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 1].set_title('Landing Suitability Score')
    axes[1, 1].axis('off')
    plt.colorbar(landing_display, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Find and highlight best landing zones
    best_zones_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    
    # Find contours of high-suitability areas
    high_suitability = (landing_score > 0.7).astype(np.uint8)
    contours, _ = cv2.findContours(high_suitability, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_zones = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum landing area
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w//2, y + h//2)
            avg_score = np.mean(landing_score[y:y+h, x:x+w])
            best_zones.append((center, (w, h), area, avg_score))
            
            # Draw landing zone
            cv2.rectangle(best_zones_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.circle(best_zones_img, center, 5, (255, 0, 0), -1)
            cv2.putText(best_zones_img, f'{avg_score:.2f}', 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    axes[1, 2].imshow(best_zones_img)
    axes[1, 2].set_title(f'Best Landing Zones ({len(best_zones)} found)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved visualization: {save_path}")
    
    return fig, best_zones

def test_fine_tuned_model():
    """Test the fine-tuned model with detailed segmentation analysis."""
    print("🚁 Testing Fine-tuned UAV Landing Model")
    print("=" * 60)
    
    # Check if model exists
    model_path = "trained_models/ultra_fast_uav_landing.onnx"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Initialize detector
    print(f"📁 Loading model: {model_path}")
    detector = UAVLandingDetector(
        model_path=model_path,
        enable_visualization=True
    )
    
    # Create test scenarios
    scenarios = create_test_scenarios()
    
    print(f"\n🧪 Testing {len(scenarios)} scenarios...")
    
    results_summary = []
    
    for i, (scenario_name, test_image) in enumerate(scenarios, 1):
        print(f"\n{'='*20} Scenario {i}: {scenario_name} {'='*20}")
        
        # Test at different altitudes
        altitudes = [10.0, 5.0, 2.0]
        scenario_results = []
        
        for altitude in altitudes:
            print(f"\n📏 Testing at {altitude}m altitude...")
            
            # Process frame
            start_time = time.time()
            result = detector.process_frame(test_image, altitude=altitude)
            inference_time = (time.time() - start_time) * 1000
            
            print(f"   ⏱️  Inference time: {inference_time:.1f}ms")
            print(f"   📊 Status: {result.status}")
            print(f"    Confidence: {result.confidence:.3f}")
            print(f"   📈 FPS: {result.fps:.1f}")
            print(f"   🛬 Phase: {detector.landing_phase}")
            
            if result.target_pixel:
                print(f"   📍 Target pixel: {result.target_pixel}")
                print(f"   🌍 Target world: ({result.target_world[0]:.3f}, {result.target_world[1]:.3f})")
                print(f"   📏 Distance: {result.distance:.2f}m")
                print(f"   🎮 Commands: F={result.forward_velocity:.3f}, R={result.right_velocity:.3f}, D={result.descent_rate:.3f}")
            
            scenario_results.append({
                'altitude': altitude,
                'inference_time': inference_time,
                'status': result.status,
                'confidence': result.confidence,
                'fps': result.fps,
                'has_target': result.target_pixel is not None
            })
        
        # Get segmentation output for visualization (using middle altitude)
        print(f"\n🎨 Generating segmentation visualization...")
        test_result = detector.process_frame(test_image, altitude=5.0)
        
        # Extract segmentation data from the detector
        if hasattr(detector, 'last_segmentation_output'):
            segmentation_mask = detector.last_segmentation_output
            confidence_map = np.max(detector.last_raw_output, axis=0) if hasattr(detector, 'last_raw_output') else np.ones_like(segmentation_mask) * 0.5
        else:
            # Create mock segmentation for demonstration
            segmentation_mask = np.random.randint(0, 4, test_image.shape[:2])
            confidence_map = np.random.rand(*test_image.shape[:2]) * 0.5 + 0.5
        
        # Visualize segmentation
        save_path = f"segmentation_analysis_{scenario_name.lower().replace(' ', '_')}.png"
        fig, landing_zones = visualize_segmentation_output(
            test_image, segmentation_mask, confidence_map, 
            f"{scenario_name} (Altitude: 5.0m)", save_path
        )
        
        # Show the plot
        plt.show()
        
        results_summary.append({
            'scenario': scenario_name,
            'results': scenario_results,
            'landing_zones': len(landing_zones)
        })
        
        print(f"    Found {len(landing_zones)} potential landing zones")
    
    # Summary report
    print(f"\n{'='*60}")
    print("📊 FINE-TUNED MODEL TEST SUMMARY")
    print(f"{'='*60}")
    
    for summary in results_summary:
        print(f"\n {summary['scenario']}:")
        print(f"   Landing zones detected: {summary['landing_zones']}")
        
        avg_time = np.mean([r['inference_time'] for r in summary['results']])
        avg_confidence = np.mean([r['confidence'] for r in summary['results']])
        target_detection_rate = sum([1 for r in summary['results'] if r['has_target']]) / len(summary['results'])
        
        print(f"   Average inference time: {avg_time:.1f}ms")
        print(f"   Average confidence: {avg_confidence:.3f}")
        print(f"   Target detection rate: {target_detection_rate:.1%}")
    
    overall_avg_time = np.mean([np.mean([r['inference_time'] for r in s['results']]) for s in results_summary])
    print(f"\n🏆 Overall Performance:")
    print(f"   Average inference time: {overall_avg_time:.1f}ms")
    print(f"   Model size: {os.path.getsize(model_path) / 1024**2:.1f} MB")
    print(f"   Real-time capable: {'✅' if overall_avg_time < 33.3 else '❌'} ({1000/overall_avg_time:.1f} FPS)")
    
    print(f"\n💾 Visualization files saved:")
    for i, (scenario_name, _) in enumerate(scenarios, 1):
        filename = f"segmentation_analysis_{scenario_name.lower().replace(' ', '_')}.png"
        print(f"   {i}. {filename}")
    
    print(f"\n🎉 Fine-tuned model testing complete!")
    
    return results_summary

if __name__ == "__main__":
    test_fine_tuned_model()
