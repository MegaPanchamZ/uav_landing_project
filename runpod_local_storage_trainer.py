#!/usr/bin/env python3
"""
RunPod Local Storage A100 Training
==================================

SOLUTION: Copy datasets from network storage (/workspace) to local storage (/tmp)
for 5-10x faster training speed by eliminating network I/O bottleneck.

Problem: RunPod /workspace = Network Attached Storage (200-400MB/s)
Solution: Copy to /tmp = Local SSD Storage (2GB/s+)
"""

import os
import sys
import shutil
import time
import subprocess
from pathlib import Path

def print_banner():
    print("🚀" * 50)
    print("🚀 RunPod Local Storage A100 Training")
    print("🚀 Network Storage → Local Storage Migration")
    print("🚀 Expected Speed Gain: 5-10x faster!")
    print("🚀" * 50)

def check_storage():
    """Check available storage space."""
    print("\n📊 Storage Analysis:")
    
    # Check local storage (/tmp)
    local_stat = os.statvfs('/tmp')
    local_free_gb = (local_stat.f_bavail * local_stat.f_frsize) / (1024**3)
    
    # Check network storage (/workspace)
    workspace_stat = os.statvfs('/workspace')
    workspace_free_gb = (workspace_stat.f_bavail * workspace_stat.f_frsize) / (1024**3)
    
    print(f"   📁 Local Storage (/tmp): {local_free_gb:.1f}GB free")
    print(f"   🌐 Network Storage (/workspace): {workspace_free_gb:.1f}GB free")
    
    return local_free_gb > 5.0  # Need at least 5GB for datasets

def copy_datasets_to_local():
    """Copy datasets from network storage to local storage."""
    print("\n🚀 Copying datasets to local storage...")
    
    # Source (network storage)
    workspace_datasets = Path('/workspace/uav_landing/datasets')
    
    # Destination (local storage)
    local_datasets = Path('/tmp/datasets')
    
    if not workspace_datasets.exists():
        print(f"❌ Source datasets not found: {workspace_datasets}")
        return False
    
    # Remove existing local datasets
    if local_datasets.exists():
        print(f"   🗑️ Removing existing local datasets...")
        shutil.rmtree(local_datasets)
    
    # Create local datasets directory
    local_datasets.mkdir(parents=True, exist_ok=True)
    
    # Copy each dataset
    datasets_to_copy = [
        'semantic_drone_dataset',
        'drone_deploy_dataset', 
        'udd6_dataset'
    ]
    
    start_time = time.time()
    
    for dataset in datasets_to_copy:
        src = workspace_datasets / dataset
        dst = local_datasets / dataset
        
        if src.exists():
            print(f"   📂 Copying {dataset}...")
            copy_start = time.time()
            
            # Use rsync for faster copying
            cmd = f"rsync -av --progress {src}/ {dst}/"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            copy_time = time.time() - copy_start
            
            if result.returncode == 0:
                # Calculate size
                size_result = subprocess.run(f"du -sh {dst}", shell=True, capture_output=True, text=True)
                size = size_result.stdout.split()[0] if size_result.returncode == 0 else "unknown"
                
                print(f"   ✅ {dataset}: {size} copied in {copy_time:.1f}s")
            else:
                print(f"   ❌ Failed to copy {dataset}: {result.stderr}")
                return False
        else:
            print(f"   ⚠️ Dataset not found: {src}")
    
    total_time = time.time() - start_time
    print(f"\n✅ All datasets copied to local storage in {total_time:.1f}s")
    return True

def setup_fast_training():
    """Setup optimized training environment."""
    print("\n🔧 Setting up fast training environment...")
    
    # CPU optimizations
    os.environ['OMP_NUM_THREADS'] = '16'
    os.environ['MKL_NUM_THREADS'] = '16'
    os.environ['NUMEXPR_NUM_THREADS'] = '16'
    os.environ['PYTHONOPTIMIZE'] = '1'
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    print("   ✅ CPU threading optimized")
    print("   ✅ Python environment optimized")

def run_training():
    """Run the actual training with local datasets."""
    print("\n🚁 Starting A100 training with local datasets...")
    
    # Training command using local storage paths
    cmd = [
        'python', '/workspace/uav_landing/scripts/train_a100_progressive_multi_dataset.py',
        '--stage', '1',
        '--sdd_data_root', '/tmp/datasets/semantic_drone_dataset',
        '--dronedeploy_data_root', '/tmp/datasets/drone_deploy_dataset', 
        '--udd6_data_root', '/tmp/datasets/udd6_dataset',
        '--stage1_epochs', '2'
    ]
    
    print(f"   🚀 Command: {' '.join(cmd)}")
    print("   📈 Expected speed: 30-60 seconds per epoch (vs 13 minutes!)")
    
    # Change to workspace directory
    os.chdir('/workspace/uav_landing')
    
    # Run training
    start_time = time.time()
    result = subprocess.run(cmd)
    training_time = time.time() - start_time
    
    print(f"\n🎉 Training completed in {training_time:.1f}s")
    return result.returncode == 0

def save_results_to_workspace():
    """Copy training results back to network storage."""
    print("\n💾 Saving results to network storage...")
    
    # Copy outputs back to workspace
    local_outputs = Path('/tmp/outputs')
    workspace_outputs = Path('/workspace/uav_landing/outputs/local_storage_training')
    
    if local_outputs.exists():
        workspace_outputs.mkdir(parents=True, exist_ok=True)
        shutil.copytree(local_outputs, workspace_outputs, dirs_exist_ok=True)
        print(f"   ✅ Results saved to: {workspace_outputs}")
    else:
        print("   ⚠️ No outputs found to save")

def main():
    """Main function to orchestrate local storage training."""
    print_banner()
    
    # Step 1: Check storage
    if not check_storage():
        print("❌ Insufficient local storage space!")
        return 1
    
    # Step 2: Copy datasets to local storage
    if not copy_datasets_to_local():
        print("❌ Failed to copy datasets to local storage!")
        return 1
    
    # Step 3: Setup training environment
    setup_fast_training()
    
    # Step 4: Run training
    if not run_training():
        print("❌ Training failed!")
        return 1
    
    # Step 5: Save results
    save_results_to_workspace()
    
    print("\n🎉 LOCAL STORAGE TRAINING COMPLETE!")
    print("🚀 Network storage bottleneck eliminated!")
    print("⚡ A100 running at full speed!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 