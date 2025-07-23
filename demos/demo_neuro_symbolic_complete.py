#!/usr/bin/env python3
"""
Complete Neuro-Symbolic UAV Landing System Demo
==============================================

Demonstrates the full pipeline:
1. Natural 24-class semantic segmentation 
2. Landing safety interpretation
3. Scallop-based logical reasoning
4. Context-aware decision making

This shows how we solved the original training issues by working
WITH the dataset structure instead of against it.
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.neuro_symbolic_landing_system import NeuroSymbolicLandingSystem
from src.landing_safety_interpreter import LandingSafetyInterpreter

def create_test_scenarios():
    """Create test scenarios with different conditions."""
    return [
        {
            'name': 'Optimal Conditions',
            'battery': 0.9,
            'weather': ('clear', 0.1),
            'emergency': 'normal',
            'description': 'High battery, clear weather, normal operations'
        },
        {
            'name': 'Low Battery Warning',
            'battery': 0.2,
            'weather': ('cloudy', 0.4),
            'emergency': 'warning',
            'description': 'Low battery, cloudy conditions, need to land soon'
        },
        {
            'name': 'Emergency Landing',
            'battery': 0.05,
            'weather': ('storm', 0.8),
            'emergency': 'critical',
            'description': 'Critical battery, severe weather, emergency protocol'
        },
        {
            'name': 'Moderate Conditions',
            'battery': 0.6,
            'weather': ('light_rain', 0.5),
            'emergency': 'normal',
            'description': 'Moderate battery, light rain, standard approach'
        }
    ]

def load_sample_image():
    """Load a sample aerial image for testing."""
    # Try to load a real image from the dataset
    dataset_path = Path("H:/landing-system/datasets/Aerial_Semantic_Segmentation_Drone_Dataset/dataset/semantic_drone_dataset")
    image_dir = dataset_path / "original_images"
    
    if image_dir.exists():
        image_files = list(image_dir.glob("*.jpg"))
        if image_files:
            print(f"Loading real aerial image: {image_files[0].name}")
            image = cv2.imread(str(image_files[0]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
    
    # Fallback: create a synthetic aerial scene
    print("Creating synthetic aerial scene for demo")
    image = create_synthetic_aerial_scene()
    return image

def create_synthetic_aerial_scene():
    """Create a synthetic aerial scene with various landing surfaces."""
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Sky/background
    image[:, :] = [135, 206, 250]  # Light blue sky
    
    # Grass areas (safe landing)
    cv2.rectangle(image, (50, 50), (200, 200), (0, 100, 0), -1)
    cv2.rectangle(image, (300, 300), (450, 450), (0, 120, 0), -1)
    
    # Paved area (safe landing)
    cv2.rectangle(image, (220, 100), (400, 180), (120, 120, 120), -1)
    
    # Water (dangerous)
    cv2.ellipse(image, (100, 350), (80, 40), 0, 0, 360, (0, 0, 200), -1)
    
    # Trees (dangerous)
    cv2.circle(image, (300, 150), 30, (0, 80, 0), -1)
    cv2.circle(image, (350, 170), 25, (0, 70, 0), -1)
    
    # Building roof (dangerous)
    cv2.rectangle(image, (400, 50), (480, 120), (100, 100, 100), -1)
    
    # Car (dangerous)
    cv2.rectangle(image, (250, 250), (280, 280), (150, 0, 0), -1)
    
    # Add some noise for realism
    noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image

def visualize_results(image, analysis, scenario_name):
    """Create comprehensive visualization of the analysis results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Neuro-Symbolic UAV Landing Analysis - {scenario_name}', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Aerial Image')
    axes[0, 0].axis('off')
    
    # Safety visualization
    if 'safety_visualization' in analysis:
        axes[0, 1].imshow(analysis['safety_visualization'])
        axes[0, 1].set_title('Landing Safety Map\n(Green=Safe, Yellow=Caution, Red=Dangerous)')
        axes[0, 1].axis('off')
        
        # Add landing zones overlay
        if analysis['best_decision']:
            pos = analysis['best_decision']['position']
            decision = analysis['best_decision']['decision']
            color = {
                'LAND_IMMEDIATELY': 'lime',
                'LAND_WITH_CAUTION': 'yellow', 
                'HOVER_AND_ASSESS': 'orange',
                'FIND_ALTERNATIVE': 'red',
                'EMERGENCY_PROTOCOL': 'purple'
            }.get(decision, 'white')
            
            axes[0, 1].scatter(pos[0], pos[1], c=color, s=200, marker='X', 
                             edgecolors='black', linewidth=2)
            axes[0, 1].annotate(f'Best Landing Zone\n{decision}', 
                              xy=pos, xytext=(pos[0]+50, pos[1]-50),
                              arrowprops=dict(arrowstyle='->', color='black', lw=2),
                              bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                              fontsize=10, fontweight='bold')
    
    # Decision analysis
    ax = axes[0, 2]
    ax.axis('off')
    ax.set_title('Landing Decisions')
    
    if analysis['landing_decisions']:
        decision_text = "Top Landing Options:\n\n"
        for i, decision in enumerate(analysis['landing_decisions'][:5]):
            decision_text += f"{i+1}. {decision['decision']}\n"
            decision_text += f"   Position: {decision['position']}\n"
            decision_text += f"   Score: {decision['score']:.3f}\n"
            decision_text += f"   {decision['explanation']}\n\n"
    else:
        decision_text = "No suitable landing zones identified"
    
    ax.text(0.05, 0.95, decision_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    # Safety statistics
    ax = axes[1, 0]
    if 'safety_analysis' in analysis:
        safety_data = analysis['safety_analysis']['percentages']
        labels = list(safety_data.keys())
        sizes = list(safety_data.values())
        colors = ['green', 'yellow', 'red', 'gray']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                         startangle=90)
        ax.set_title('Safety Distribution')
        
        # Add overall assessment
        overall = analysis['safety_analysis']['overall_safety']
        recommendation = analysis['safety_analysis']['recommendation']
        ax.text(0, -1.5, f"Overall: {overall}\n{recommendation}", 
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
    
    # Context information
    ax = axes[1, 1]
    ax.axis('off')
    ax.set_title('Mission Context')
    
    context = analysis['context']
    context_text = f"""
Battery Level: {context['battery_level']:.1%}
Weather: {context['weather_condition'][0]} (severity: {context['weather_condition'][1]:.1f})
Emergency Status: {context['emergency_status']}

Extracted Facts:
• Semantic Regions: {len(analysis['extracted_facts']['semantic_regions'])}
• Safety Zones: {len(analysis['extracted_facts']['safety_levels'])}
• Obstacles Detected: {len(analysis['extracted_facts']['obstacles'])}

Neuro-Symbolic Pipeline:
✓ Neural: 24-class semantic segmentation
✓ Symbolic: Scallop logical reasoning
✓ Integration: Probabilistic facts → Rules → Decision
"""
    
    ax.text(0.05, 0.95, context_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    
    # Reasoning explanation
    ax = axes[1, 2]
    ax.axis('off')
    ax.set_title('Scallop Reasoning Process')
    
    reasoning_text = """
Scallop Logic Rules Applied:

1. Semantic Understanding:
   • good_surface(pos, conf) ← paved-area, grass, dirt
   • dangerous_element(pos, element) ← water, rocks, trees
   
2. Spatial Analysis:
   • large_safe_zone(pos) ← area > 200 pixels
   • clear_zone(pos) ← no obstacles nearby
   
3. Situational Awareness:
   • emergency_landing_needed() ← battery < 15%
   • optimal_conditions() ← battery > 50% & weather < 30%
   
4. Decision Logic:
   • LAND_IMMEDIATELY ← large safe zone & clear & optimal
   • LAND_WITH_CAUTION ← medium zone & no danger
   • HOVER_AND_ASSESS ← good surface & near danger
   
5. Ranking:
   • safety_score ← confidence × area / 300
   • best_decision ← max(safety_score)
"""
    
    ax.text(0.05, 0.95, reasoning_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    return fig

def run_complete_demo():
    """Run the complete neuro-symbolic demo."""
    print("🚁 Neuro-Symbolic UAV Landing System Demo")
    print("=" * 50)
    
    # Initialize the system
    print("Initializing neuro-symbolic system...")
    system = NeuroSymbolicLandingSystem(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test image
    print("Loading test aerial image...")
    test_image = load_sample_image()
    
    # Test scenarios
    scenarios = create_test_scenarios()
    
    results = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\n🔍 Testing Scenario {i+1}: {scenario['name']}")
        print(f"   {scenario['description']}")
        
        # Process with neuro-symbolic system
        analysis = system.process_image(
            test_image,
            battery_level=scenario['battery'],
            weather_condition=scenario['weather'],
            emergency_status=scenario['emergency']
        )
        
        # Display key results
        if analysis['best_decision']:
            decision = analysis['best_decision']
            print(f"   🎯 Decision: {decision['decision']}")
            print(f"   📍 Position: {decision['position']}")
            print(f"   ⭐ Score: {decision['score']:.3f}")
            print(f"   💡 Explanation: {decision['explanation']}")
        else:
            print("   ❌ No suitable landing zones found")
        
        overall_safety = analysis['safety_analysis']['overall_safety']
        print(f"   🛡️  Overall Safety: {overall_safety}")
        
        # Store results
        results.append({
            'scenario': scenario,
            'analysis': analysis
        })
        
        # Create visualization
        fig = visualize_results(test_image, analysis, scenario['name'])
        
        # Save visualization
        output_dir = Path("outputs/neuro_symbolic_demo")
        output_dir.mkdir(exist_ok=True)
        
        filename = f"scenario_{i+1}_{scenario['name'].lower().replace(' ', '_')}.png"
        filepath = output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"   📊 Visualization saved: {filepath}")
        
        plt.close(fig)  # Close to save memory
    
    # Generate summary comparison
    print("\n📊 Summary Comparison")
    print("-" * 50)
    print(f"{'Scenario':<20} {'Decision':<20} {'Score':<8} {'Safety':<12}")
    print("-" * 60)
    
    for result in results:
        scenario_name = result['scenario']['name']
        analysis = result['analysis']
        
        if analysis['best_decision']:
            decision = analysis['best_decision']['decision']
            score = analysis['best_decision']['score']
        else:
            decision = "NO_DECISION"
            score = 0.0
        
        safety = analysis['safety_analysis']['overall_safety']
        
        print(f"{scenario_name:<20} {decision:<20} {score:<8.3f} {safety:<12}")
    
    print(f"\n✅ Demo completed! Visualizations saved in 'outputs/neuro_symbolic_demo/'")
    print("\n🎯 Key Achievements:")
    print("• Solved class imbalance by working WITH dataset structure (24 natural classes)")
    print("• Integrated Scallop for explainable logical reasoning")
    print("• Context-aware decisions (battery, weather, emergency status)")
    print("• Probabilistic neural facts → Symbolic rules → Landing decisions")

if __name__ == "__main__":
    try:
        run_complete_demo()
    except KeyboardInterrupt:
        print("\n\n❌ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc() 