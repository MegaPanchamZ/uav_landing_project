#!/usr/bin/env python3
"""
Quick A100 Performance Fix
==========================
Directly modify training parameters for optimal A100 performance
"""

import re

def fix_a100_performance():
    """Apply quick fixes for A100 performance."""
    
    file_path = "/workspace/uav_landing/scripts/train_a100_progressive_multi_dataset.py"
    
    print("🔧 Applying A100 performance fixes...")
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix 1: Reduce batch size from 256 to 32
    content = re.sub(
        r'batch_size: int = 256.*', 
        'batch_size: int = 32,  # Optimal for A100 performance',
        content
    )
    
    # Fix 2: Increase workers to 24 (from 2)
    content = re.sub(
        r'num_workers=2.*pin_memory=True.*drop_last=True.*',
        'num_workers=24, pin_memory=True, drop_last=True,  # Use 24 of 32 cores!',
        content
    )
    
    # Fix 3: Increase CPU threads
    content = re.sub(
        r"os\.environ\['OMP_NUM_THREADS'\] = '8'",
        "os.environ['OMP_NUM_THREADS'] = '16'  # Use more cores",
        content
    )
    
    content = re.sub(
        r"torch\.set_num_threads\(8\)",
        "torch.set_num_threads(16)  # Match environment",
        content
    )
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ A100 performance fixes applied!")
    print("   • Batch size: 256 → 32")
    print("   • Workers: 2 → 24")  
    print("   • CPU threads: 8 → 16")
    
    return True

if __name__ == "__main__":
    fix_a100_performance() 