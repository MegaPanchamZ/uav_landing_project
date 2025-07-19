#!/usr/bin/env python3
"""
Quick Setup and Fine-Tuning Launcher for UAV Landing Detection

This script sets up the environment and launches fine-tuning with sensible defaults
for your DroneDeploy dataset.
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse

def install_requirements():
    """Install required packages."""
    requirements = [
        "torch>=1.9.0",
        "torchvision>=0.10.0", 
        "opencv-python",
        "numpy",
        "Pillow",
        "tqdm",
        "albumentations>=1.0.0",
        "matplotlib",
        "tensorboard"
    ]
    
    print("🔧 Installing required packages...")
    for req in requirements:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", req], 
                         check=True, capture_output=True)
            print(f"✅ Installed {req}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {req}: {e}")
            
def check_dataset_structure(data_path):
    """Check if the dataset has the expected structure."""
    data_path = Path(data_path)
    
    required_dirs = ["images", "labels"]
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not (data_path / dir_name).exists():
            missing_dirs.append(dir_name)
            
    if missing_dirs:
        print(f"❌ Missing directories in dataset: {missing_dirs}")
        print(f"Expected structure:")
        print(f"  {data_path}/")
        print(f"  ├── images/")
        print(f"  │   └── *.tif files")
        print(f"  └── labels/")
        print(f"      └── *.png files")
        return False
        
    # Count files
    image_count = len(list((data_path / "images").glob("*.tif")))
    label_count = len(list((data_path / "labels").glob("*.png")))
    
    print(f"✅ Dataset structure looks good!")
    print(f"   Found {image_count} images and {label_count} labels")
    
    if image_count == 0:
        print("❌ No image files found! Check the file extensions.")
        return False
        
    if label_count == 0:
        print("❌ No label files found! Check the file extensions.")
        return False
        
    return True

def find_pretrained_model():
    """Find pre-trained model in the project."""
    possible_paths = [
        "../model_pths/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes_20210902_015551-bcf10f09.pth",
        "../model_pths/bisenetv2_fcn_fp16_4x4_1024x1024_160k_cityscapes_20210902_045942-b979777b.pth",
        "../../model_pths/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes_20210902_015551-bcf10f09.pth",
        "../../model_pths/bisenetv2_fcn_fp16_4x4_1024x1024_160k_cityscapes_20210902_045942-b979777b.pth"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found pre-trained model: {path}")
            return path
            
    print("⚠️  No pre-trained model found. Training will start from scratch.")
    return None

def launch_training(args):
    """Launch the training process."""
    
    # Build command
    cmd = [
        sys.executable, "practical_fine_tuning.py",
        "--data_path", args.data_path,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--output_dir", args.output_dir
    ]
    
    # Add pre-trained model if found
    if args.pretrained_model:
        cmd.extend(["--pretrained_model", args.pretrained_model])
    else:
        pretrained_path = find_pretrained_model()
        if pretrained_path:
            cmd.extend(["--pretrained_model", pretrained_path])
    
    # Add device
    if args.device != "auto":
        cmd.extend(["--device", args.device])
        
    # Add ONNX export
    if args.export_onnx:
        cmd.append("--export_onnx")
        
    print(f"🚀 Launching training with command:")
    print(f"   {' '.join(cmd)}")
    print()
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Quick Setup and Training Launcher")
    parser.add_argument("--data_path", type=str, 
                       default="../datasets/drone_deploy_dataset_intermediate/dataset-medium",
                       help="Path to dataset directory")
    parser.add_argument("--pretrained_model", type=str,
                       help="Path to pre-trained model (auto-detected if not provided)")
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_models",
                       help="Output directory for models")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size (reduce if you have memory issues)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--skip_install", action="store_true",
                       help="Skip package installation")
    parser.add_argument("--export_onnx", action="store_true", 
                       help="Export to ONNX after training")
    parser.add_argument("--quick_test", action="store_true",
                       help="Quick test with 5 epochs")
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick_test:
        args.epochs = 5
        args.batch_size = 2
        print("🧪 Quick test mode: 5 epochs, batch size 2")
        
    print("🎯 UAV Landing Detection Fine-Tuning Setup")
    print("=" * 50)
    
    # Install requirements
    if not args.skip_install:
        install_requirements()
    else:
        print("⏭️  Skipping package installation")
        
    print()
    
    # Check dataset
    print("📂 Checking dataset structure...")
    if not check_dataset_structure(args.data_path):
        print("\n❌ Dataset check failed. Please fix the issues above.")
        sys.exit(1)
        
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("practical_fine_tuning.py"):
        print("❌ practical_fine_tuning.py not found!")
        print("   Please run this script from the training_tools/ directory")
        sys.exit(1)
        
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Launch training
    print("🚀 Starting fine-tuning process...")
    launch_training(args)
    
    print("\n🎉 Setup and training completed!")

if __name__ == "__main__":
    main()
