#!/usr/bin/env python3
"""
Create visualization of the Ultra-Fast BiSeNet architecture
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_model_architecture_visualization():
    """Create a detailed architecture diagram."""
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Colors for different components
    colors = {
        'input': '#E8F4FD',      # Light blue
        'conv': '#4CAF50',       # Green
        'backbone': '#FF9800',   # Orange  
        'decoder': '#9C27B0',    # Purple
        'classifier': '#F44336', # Red
        'output': '#607D8B'      # Blue grey
    }
    
    # Title
    ax.text(8, 11.5, 'Ultra-Fast BiSeNet Architecture', 
            fontsize=24, fontweight='bold', ha='center')
    ax.text(8, 11, 'UAV Landing Detection Model (333K parameters, 1.3MB)', 
            fontsize=14, ha='center', style='italic')
    
    # Input layer
    input_box = FancyBboxPatch((0.5, 9), 2, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 9.75, 'INPUT\nRGB Image\n3×256×256', 
            ha='center', va='center', fontweight='bold')
    
    # Backbone layers
    backbone_layers = [
        {'name': 'Conv2d(3→32)\nBatchNorm2d\nReLU', 'pos': (3.5, 9), 'size': (2, 1.5)},
        {'name': 'Conv2d(32→64)\nStride=2\nBatchNorm2d\nReLU', 'pos': (6.5, 9), 'size': (2, 1.5)},
        {'name': 'Conv2d(64→128)\nStride=2\nBatchNorm2d\nReLU', 'pos': (9.5, 9), 'size': (2, 1.5)},
        {'name': 'Conv2d(128→128)\nBatchNorm2d\nReLU', 'pos': (12.5, 9), 'size': (2, 1.5)},
    ]
    
    for i, layer in enumerate(backbone_layers):
        box = FancyBboxPatch(layer['pos'], layer['size'][0], layer['size'][1], 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors['backbone'], 
                            edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(layer['pos'][0] + layer['size'][0]/2, 
                layer['pos'][1] + layer['size'][1]/2, 
                layer['name'], ha='center', va='center', 
                fontsize=9, fontweight='bold')
    
    # Feature map sizes
    feature_sizes = ['256×256', '128×128', '64×64', '64×64']
    for i, size in enumerate(feature_sizes):
        ax.text(4.5 + i*3, 8.3, size, ha='center', va='center', 
                fontsize=8, style='italic', color='blue')
    
    # Decoder layers  
    decoder_layers = [
        {'name': 'Conv2d(128→64)\nBatchNorm2d\nReLU', 'pos': (9.5, 6), 'size': (2, 1.5)},
        {'name': 'Conv2d(64→32)\nBatchNorm2d\nReLU', 'pos': (6.5, 6), 'size': (2, 1.5)},
    ]
    
    for layer in decoder_layers:
        box = FancyBboxPatch(layer['pos'], layer['size'][0], layer['size'][1], 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors['decoder'], 
                            edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(layer['pos'][0] + layer['size'][0]/2, 
                layer['pos'][1] + layer['size'][1]/2, 
                layer['name'], ha='center', va='center', 
                fontsize=9, fontweight='bold', color='white')
    
    # Classifier
    classifier_box = FancyBboxPatch((3.5, 6), 2, 1.5, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=colors['classifier'], 
                                   edgecolor='black', linewidth=1)
    ax.add_patch(classifier_box)
    ax.text(4.5, 6.75, 'Conv2d(32→4)\nClassifier', 
            ha='center', va='center', fontweight='bold', color='white')
    
    # Interpolation
    interp_box = FancyBboxPatch((0.5, 6), 2, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['conv'], 
                               edgecolor='black', linewidth=1)
    ax.add_patch(interp_box)
    ax.text(1.5, 6.75, 'Bilinear\nInterpolation\n64×64 → 256×256', 
            ha='center', va='center', fontweight='bold', color='white')
    
    # Output
    output_box = FancyBboxPatch((0.5, 3), 2, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['output'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(1.5, 3.75, 'OUTPUT\nSegmentation\n4×256×256', 
            ha='center', va='center', fontweight='bold', color='white')
    
    # Draw connections
    connections = [
        # Forward pass
        ((2.5, 9.75), (3.5, 9.75)),  # Input → Conv1
        ((5.5, 9.75), (6.5, 9.75)),  # Conv1 → Conv2  
        ((8.5, 9.75), (9.5, 9.75)),  # Conv2 → Conv3
        ((11.5, 9.75), (12.5, 9.75)), # Conv3 → Conv4
        # Decoder path
        ((13.5, 9), (10.5, 7.5)),    # Conv4 → Decoder1
        ((9.5, 6.75), (8.5, 6.75)),  # Decoder1 → Decoder2
        ((6.5, 6.75), (5.5, 6.75)),  # Decoder2 → Classifier
        ((3.5, 6.75), (2.5, 6.75)),  # Classifier → Interpolation
        ((1.5, 6), (1.5, 4.5)),      # Interpolation → Output
    ]
    
    for start, end in connections:
        arrow = ConnectionPatch(start, end, "data", "data", 
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=20, fc="black", ec="black", linewidth=2)
        ax.add_patch(arrow)
    
    # Add performance metrics box
    metrics_box = FancyBboxPatch((11, 2), 4.5, 3.5, 
                                boxstyle="round,pad=0.2", 
                                facecolor='lightgray', 
                                edgecolor='black', linewidth=1)
    ax.add_patch(metrics_box)
    
    metrics_text = """🚀 PERFORMANCE METRICS

⚡ Inference Speed:
   • PyTorch: 1.0ms (1,022 FPS)
   • ONNX: 8.2ms (121 FPS)

📦 Model Size:
   • Parameters: 333,668
   • File Size: 1.3 MB
   • Memory: <2GB VRAM

🎯 Accuracy:
   • IoU Score: 59.0%
   • Training Time: 25 min"""
    
    ax.text(13.25, 3.75, metrics_text, ha='center', va='center', 
            fontsize=10, fontfamily='monospace')
    
    # Add class legend
    class_colors = ['black', 'green', 'gold', 'red']
    class_names = ['Background', 'Safe Landing', 'Caution', 'Danger']
    
    legend_elements = []
    for color, name in zip(class_colors, class_names):
        legend_elements.append(mpatches.Patch(color=color, label=name))
    
    ax.legend(handles=legend_elements, loc='upper left', 
             bbox_to_anchor=(0, 0.15), fontsize=10, title='Landing Classes')
    
    # Add architecture details
    arch_text = """ARCHITECTURE OPTIMIZATIONS:
• No bias in conv layers (memory efficient)
• Minimal channels (32→64→128→64→32)
• Single upsampling path
• 256×256 input (vs 512×512)
• Mixed precision training"""
    
    ax.text(4, 1, arch_text, ha='left', va='bottom', 
            fontsize=9, fontfamily='monospace', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    plt.tight_layout()
    plt.savefig('visualizations/model_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('visualizations/model_architecture.pdf', bbox_inches='tight')
    print("✅ Architecture diagram saved to visualizations/model_architecture.png")
    
    return fig

def create_training_pipeline_diagram():
    """Create training pipeline visualization."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Staged Fine-Tuning Pipeline', 
            fontsize=22, fontweight='bold', ha='center')
    
    # Stage boxes
    stages = [
        {'name': 'Stage 0\nBiSeNetV2\nPre-trained\n(Cityscapes)', 'pos': (1, 7), 'color': '#E3F2FD'},
        {'name': 'Stage 1\nDroneDeploy\nAerial Adaptation\n(7 classes)', 'pos': (5, 7), 'color': '#FFF3E0'},  
        {'name': 'Stage 2\nUDD6\nLanding Classes\n(4 classes)', 'pos': (9, 7), 'color': '#E8F5E8'},
    ]
    
    for i, stage in enumerate(stages):
        box = FancyBboxPatch(stage['pos'], 3, 2, 
                            boxstyle="round,pad=0.1", 
                            facecolor=stage['color'], 
                            edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(stage['pos'][0] + 1.5, stage['pos'][1] + 1, 
                stage['name'], ha='center', va='center', 
                fontsize=11, fontweight='bold')
        
        # Add arrows
        if i < len(stages) - 1:
            arrow = ConnectionPatch((stage['pos'][0] + 3, stage['pos'][1] + 1), 
                                  (stages[i+1]['pos'][0], stages[i+1]['pos'][1] + 1),
                                  "data", "data", arrowstyle="->", 
                                  shrinkA=5, shrinkB=5, mutation_scale=25, 
                                  fc="blue", ec="blue", linewidth=3)
            ax.add_patch(arrow)
    
    # Dataset info
    datasets = [
        {'name': 'ImageNet\nCityscapes\nGeneral Features', 'pos': (1, 4.5)},
        {'name': '55 Images\n44 Train / 11 Val\nAerial View', 'pos': (5, 4.5)},
        {'name': '106 Train / 35 Val\nLanding Detection\nReal UAV Data', 'pos': (9, 4.5)},
    ]
    
    for dataset in datasets:
        box = FancyBboxPatch(dataset['pos'], 3, 1.5, 
                            boxstyle="round,pad=0.1", 
                            facecolor='lightblue', 
                            edgecolor='navy', linewidth=1)
        ax.add_patch(box)
        ax.text(dataset['pos'][0] + 1.5, dataset['pos'][1] + 0.75, 
                dataset['name'], ha='center', va='center', 
                fontsize=9)
    
    # Results
    results = [
        {'text': 'Baseline\nSegmentation', 'pos': (1, 2)},
        {'text': 'Loss: 0.946\n~12 min', 'pos': (5, 2)},
        {'text': 'IoU: 59.0%\nLoss: 0.738\n~13 min', 'pos': (9, 2)},
    ]
    
    for result in results:
        box = FancyBboxPatch(result['pos'], 3, 1.5, 
                            boxstyle="round,pad=0.1", 
                            facecolor='lightgreen', 
                            edgecolor='darkgreen', linewidth=1)
        ax.add_patch(box)
        ax.text(result['pos'][0] + 1.5, result['pos'][1] + 0.75, 
                result['text'], ha='center', va='center', 
                fontsize=9, fontweight='bold')
    
    # Add optimization notes
    opt_text = """🚀 ULTRA-FAST OPTIMIZATIONS:
• Mixed Precision Training (CUDA AMP)
• 256×256 input size (vs 512×512)
• Lightweight architecture (333K params)
• Batch size 6 for 8GB GPU
• Persistent workers & pin memory
• Fewer epochs (6 + 8 vs 20+ each)"""
    
    ax.text(12.5, 5, opt_text, ha='center', va='center', 
            fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('visualizations/training_pipeline.png', dpi=300, bbox_inches='tight')
    print("✅ Training pipeline diagram saved to visualizations/training_pipeline.png")
    
    return fig

if __name__ == "__main__":
    # Create visualizations directory
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    print("🎨 Creating model architecture visualization...")
    arch_fig = create_model_architecture_visualization()
    
    print("🎨 Creating training pipeline visualization...")  
    pipeline_fig = create_training_pipeline_diagram()
    
    print("✅ All visualizations created successfully!")
    print("📁 Files saved in visualizations/")
    
    # Display if in interactive mode
    import sys
    if hasattr(sys, 'ps1'):
        plt.show()
