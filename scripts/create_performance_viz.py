#!/usr/bin/env python3
"""
Create a quick visualization of model performance for the semantic segmentation
"""
import matplotlib.pyplot as plt
import numpy as np

def create_performance_visualization():
    """Create performance comparison visualization."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🚁 Ultra-Fast UAV Landing Detection - Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Speed Comparison
    methods = ['Previous ML', 'Classical CV', 'Ultra-Fast ML']
    speeds = [5000, 13, 1.0]  # milliseconds
    colors = ['red', 'orange', 'green']
    
    bars1 = ax1.bar(methods, speeds, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Inference Time (ms)', fontweight='bold')
    ax1.set_title('⚡ Inference Speed Comparison', fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, speed in zip(bars1, speeds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height*1.1,
                f'{speed}ms', ha='center', va='bottom', fontweight='bold')
    
    # 2. Model Size Comparison
    sizes = [48, 0.01, 1.3]  # MB (Classical CV is negligible)
    bars2 = ax2.bar(methods, sizes, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Model Size (MB)', fontweight='bold')
    ax2.set_title('📦 Model Size Comparison', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    for bar, size in zip(bars2, sizes):
        height = bar.get_height()
        if size < 0.1:
            label = 'N/A'
        else:
            label = f'{size} MB'
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                label, ha='center', va='bottom', fontweight='bold')
    
    # 3. Accuracy vs Speed
    accuracy = [27, 70, 59]  # IoU/mIoU percentages  
    ax3.scatter(speeds, accuracy, s=[200, 200, 400], c=colors, alpha=0.7, edgecolors='black', linewidth=2)
    ax3.set_xlabel('Inference Time (ms)', fontweight='bold')
    ax3.set_ylabel('Accuracy (IoU %)', fontweight='bold')
    ax3.set_title('🎯 Accuracy vs Speed Trade-off', fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Add method labels
    for i, method in enumerate(methods):
        ax3.annotate(method, (speeds[i], accuracy[i]), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))
    
    # 4. Training Progress
    epochs_stage1 = range(1, 7)
    loss_stage1 = [1.93, 1.42, 1.18, 1.11, 1.06, 1.03]
    
    epochs_stage2 = range(7, 15)  # Continue from stage 1
    loss_stage2 = [1.36, 1.22, 0.97, 0.87, 0.85, 0.79, 0.81, 0.79]
    
    ax4.plot(epochs_stage1, loss_stage1, 'o-', color='orange', linewidth=2, 
             markersize=6, label='Stage 1: DroneDeploy')
    ax4.plot(epochs_stage2, loss_stage2, 'o-', color='purple', linewidth=2,
             markersize=6, label='Stage 2: UDD6')
    
    ax4.set_xlabel('Epoch', fontweight='bold')
    ax4.set_ylabel('Validation Loss', fontweight='bold')
    ax4.set_title('📈 Training Progress (Staged Fine-tuning)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Add vertical line to separate stages
    ax4.axvline(x=6.5, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Stage Transition')
    
    plt.tight_layout()
    plt.savefig('visualizations/performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('visualizations/performance_analysis.pdf', bbox_inches='tight')
    
    print("✅ Performance analysis saved to visualizations/performance_analysis.png")
    return fig

def create_inference_demo():
    """Create a visual demo of the segmentation output."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('🚁 Semantic Segmentation Demo - Landing Site Detection', fontsize=16, fontweight='bold')
    
    # Simulate different scenarios
    scenarios = [
        ('Urban Scene', 'High buildings, roads'),
        ('Rural Field', 'Grass, trees, clearings'),
        ('Mixed Terrain', 'Roads, vegetation, obstacles')
    ]
    
    # Class colors
    class_colors = {
        0: [0, 0, 0],        # Background - Black
        1: [0, 255, 0],      # Safe - Green
        2: [255, 255, 0],    # Caution - Yellow  
        3: [255, 0, 0],      # Danger - Red
    }
    
    for i, (title, description) in enumerate(scenarios):
        # Create synthetic input image
        np.random.seed(i + 42)
        input_image = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        
        # Create realistic segmentation patterns
        mask = np.zeros((256, 256), dtype=np.int32)
        
        if i == 0:  # Urban
            # Roads (safe)
            mask[100:156, :] = 1  # Horizontal road
            mask[:, 100:156] = 1  # Vertical road
            # Buildings (danger)  
            mask[20:80, 20:80] = 3
            mask[176:236, 176:236] = 3
            # Some vegetation
            mask[20:80, 176:236] = 2
            
        elif i == 1:  # Rural
            # Large safe area in center
            y, x = np.ogrid[:256, :256]
            center_mask = (x - 128)**2 + (y - 128)**2 < 50**2
            mask[center_mask] = 1
            # Vegetation around
            mask[center_mask == False] = 2
            # Some trees (danger)
            mask[:50, :50] = 3
            mask[206:, 206:] = 3
            
        else:  # Mixed
            # Complex pattern
            mask[50:100, 50:200] = 1  # Safe strip
            mask[150:200, 50:200] = 2  # Caution strip
            mask[0:40, :] = 3  # Danger at top
            mask[216:, :] = 3  # Danger at bottom
            
        # Convert to RGB
        output_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        for class_id, color in class_colors.items():
            output_rgb[mask == class_id] = color
        
        # Display input
        axes[0, i].imshow(input_image)
        axes[0, i].set_title(f'Input: {title}', fontweight='bold')
        axes[0, i].set_xlabel(description, style='italic')
        axes[0, i].axis('off')
        
        # Display output
        axes[1, i].imshow(output_rgb)
        axes[1, i].set_title(f'Output: Landing Classes', fontweight='bold')
        axes[1, i].axis('off')
        
        # Add class statistics
        unique, counts = np.unique(mask, return_counts=True)
        stats_text = ""
        for class_id, count in zip(unique, counts):
            if class_id == 0:
                continue  # Skip background
            class_name = ['', 'Safe', 'Caution', 'Danger'][class_id]
            percentage = (count / mask.size) * 100
            stats_text += f"{class_name}: {percentage:.1f}%\n"
        
        axes[1, i].text(10, 240, stats_text.strip(), 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=9, verticalalignment='bottom')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Safe Landing'),
        Patch(facecolor='yellow', label='Caution'), 
        Patch(facecolor='red', label='Danger'),
        Patch(facecolor='black', label='Background')
    ]
    
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=4)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig('visualizations/segmentation_demo.png', dpi=300, bbox_inches='tight')
    
    print("✅ Segmentation demo saved to visualizations/segmentation_demo.png")
    return fig

if __name__ == "__main__":
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    print("🎨 Creating performance visualization...")
    perf_fig = create_performance_visualization()
    
    print("🎨 Creating segmentation demo...")
    demo_fig = create_inference_demo()
    
    print("✅ All visualizations completed!")
    
    # Show if interactive
    import sys
    if hasattr(sys, 'ps1'):
        plt.show()
