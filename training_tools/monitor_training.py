#!/usr/bin/env python3
"""
Training Monitor - Check the progress of your fine-tuning

Run this script to see current training progress and results.
"""

import json
import time
import os
from pathlib import Path

def check_training_progress():
    """Check and display training progress."""
    
    output_dir = Path("./fine_tuned_models")
    history_file = output_dir / "training_history.json"
    best_model = output_dir / "best_model.pth"
    onnx_model = output_dir / "bisenetv2_uav_landing.onnx"
    
    print(" UAV Landing Detection Fine-Tuning Monitor")
    print("=" * 50)
    
    # Check if training has started
    if not output_dir.exists():
        print("❓ Training has not started yet.")
        print("   Run: python3 practical_fine_tuning.py --data_path ../../datasets/drone_deploy_dataset_intermediate/dataset-medium")
        return
        
    print(f"📂 Output directory: {output_dir}")
    
    # List files in output directory
    if output_dir.exists():
        files = list(output_dir.glob("*"))
        if files:
            print(f"📄 Files created:")
            for file in sorted(files):
                size = file.stat().st_size / (1024*1024) if file.is_file() else 0
                print(f"   {file.name} ({size:.1f} MB)")
        else:
            print("   No files created yet.")
            
    print()
    
    # Check training history
    if history_file.exists():
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
                
            current_epoch = len(history['train_losses'])
            best_miou = history['best_miou']
            latest_loss = history['train_losses'][-1] if history['train_losses'] else 0
            latest_miou = history['val_mious'][-1] if history['val_mious'] else 0
            
            print(f"📈 Training Progress:")
            print(f"   Current epoch: {current_epoch}")
            print(f"   Latest train loss: {latest_loss:.4f}")
            print(f"   Latest val mIoU: {latest_miou:.4f}")
            print(f"   Best val mIoU: {best_miou:.4f}")
            
            # Show trend
            if len(history['train_losses']) >= 2:
                loss_trend = "📉" if history['train_losses'][-1] < history['train_losses'][-2] else "📈"
                miou_trend = "📈" if history['val_mious'][-1] > history['val_mious'][-2] else "📉"
                print(f"   Loss trend: {loss_trend}")
                print(f"   mIoU trend: {miou_trend}")
                
        except Exception as e:
            print(f"❌ Error reading training history: {e}")
    else:
        print("⏳ Training history not available yet.")
        
    print()
    
    # Check model files
    if best_model.exists():
        print(f" Best model saved: {best_model}")
        model_size = best_model.stat().st_size / (1024*1024)
        print(f"   Model size: {model_size:.1f} MB")
    else:
        print("⏳ Best model not saved yet.")
        
    if onnx_model.exists():
        print(f" ONNX model exported: {onnx_model}")
        onnx_size = onnx_model.stat().st_size / (1024*1024)
        print(f"   ONNX size: {onnx_size:.1f} MB")
        print("   🚀 Ready for deployment!")
    else:
        print("⏳ ONNX model not exported yet.")
        
    print()
    
    # Show checkpoints
    checkpoints = list(output_dir.glob("checkpoint_epoch_*.pth"))
    if checkpoints:
        print(f"💾 Checkpoints saved: {len(checkpoints)}")
        for cp in sorted(checkpoints):
            print(f"   {cp.name}")
    else:
        print("💾 No checkpoints saved yet.")
        
def main():
    try:
        check_training_progress()
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped.")

if __name__ == "__main__":
    main()
