#!/bin/bash

# =============================================================================
# Kaggle Datasets Download Script for UAV Landing Training
# =============================================================================
# Downloads all required datasets for the enhanced training pipeline
# Run this after setting up kaggle.json credentials

set -e  # Exit on any error

echo "📥 Downloading UAV Landing Training Datasets..."

# Ensure we're in the right directory and environment
cd ~/uav_landing_system
source venv/bin/activate

# Verify Kaggle API is set up
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Error: kaggle.json not found!"
    echo "Please download your Kaggle API token from https://www.kaggle.com/settings"
    echo "Place it in ~/.kaggle/kaggle.json and run:"
    echo "chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# Verify permissions
chmod 600 ~/.kaggle/kaggle.json

# Create datasets directory
mkdir -p datasets
cd datasets

echo "🔍 Verifying Kaggle API connection..."
kaggle --version

# =============================================================================
# 1. Primary Dataset: Semantic Drone Dataset
# =============================================================================
echo ""
echo "📦 1/3 Downloading Semantic Drone Dataset (Primary)..."
echo "Size: ~2.8GB - 400 high-resolution images (6000x4000)"
echo "Classes: 24 semantic classes for aerial segmentation"

if [ ! -d "semantic_drone_dataset" ]; then
    kaggle datasets download -d bulentsiyah/semantic-drone-dataset
    
    echo "📂 Extracting Semantic Drone Dataset..."
    unzip -q semantic-drone-dataset.zip -d semantic_drone_dataset/
    rm semantic-drone-dataset.zip
    
    echo "✅ Semantic Drone Dataset downloaded and extracted"
else
    echo "⏭️  Semantic Drone Dataset already exists"
fi

# =============================================================================
# 2. Secondary Dataset: UDD6 Dataset (if available on Kaggle)
# =============================================================================
echo ""
echo "📦 2/3 Downloading Urban Drone Dataset (UDD6)..."
echo "Size: ~500MB - Urban aerial imagery"
echo "Classes: 6 urban classes for domain adaptation"

# Try to download UDD6 dataset (adjust dataset name if different)
if [ ! -d "udd6_dataset" ]; then
    # Search for UDD6 or similar urban drone datasets
    echo "🔍 Searching for Urban Drone datasets..."
    kaggle datasets list -s "urban drone dataset" --max-size 1000000000
    
    # Attempt download (adjust the dataset path as needed)
    echo "📝 Note: UDD6 may need manual download from original source"
    echo "Original source: https://sites.google.com/view/zhyzheng/udd"
    mkdir -p udd6_dataset
    echo "⚠️  UDD6 dataset may require manual download"
else
    echo "⏭️  UDD6 dataset directory already exists"
fi

# =============================================================================
# 3. Tertiary Dataset: DroneDeploy Dataset
# =============================================================================
echo ""
echo "📦 3/3 Downloading DroneDeploy Dataset..."
echo "Size: ~200MB - High-resolution aerial imagery"
echo "Classes: 6 classes optimized for landing detection"

if [ ! -d "dronedeploy_dataset" ]; then
    # Search for DroneDeploy dataset
    echo "🔍 Searching for DroneDeploy datasets..."
    kaggle datasets list -s "dronedeploy" --max-size 500000000
    
    # Try common DroneDeploy dataset names
    DRONEDEPLOY_FOUND=false
    
    # Try different potential dataset names
    for dataset in "dronedeploy/segmentation-dataset" "dronedeploy/drone-dataset" "bulentsiyah/dronedeploy-dataset"
    do
        echo "🔍 Trying to download: $dataset"
        if kaggle datasets download "$dataset" 2>/dev/null; then
            echo "📂 Extracting DroneDeploy dataset..."
            mkdir -p dronedeploy_dataset
            unzip -q "*.zip" -d dronedeploy_dataset/ 2>/dev/null || true
            rm -f *.zip
            DRONEDEPLOY_FOUND=true
            echo "✅ DroneDeploy dataset downloaded"
            break
        fi
    done
    
    if [ "$DRONEDEPLOY_FOUND" = false ]; then
        mkdir -p dronedeploy_dataset
        echo "⚠️  DroneDeploy dataset not found on Kaggle"
        echo "📝 May require manual download from original source"
    fi
else
    echo "⏭️  DroneDeploy dataset directory already exists"
fi

# =============================================================================
# 4. Dataset Organization and Verification
# =============================================================================
echo ""
echo "📊 Organizing dataset structure..."

# Create organized directory structure
mkdir -p {processed,raw,splits}

# Move datasets to organized structure
if [ -d "semantic_drone_dataset" ]; then
    mv semantic_drone_dataset raw/
fi

if [ -d "udd6_dataset" ]; then
    mv udd6_dataset raw/
fi

if [ -d "dronedeploy_dataset" ]; then
    mv dronedeploy_dataset raw/
fi

# =============================================================================
# 5. Create Dataset Info Summary
# =============================================================================
echo "📋 Creating dataset summary..."
cat > dataset_info.txt << EOF
UAV Landing Detection Datasets
==============================

Downloaded on: $(date)
Total datasets: 3

1. Semantic Drone Dataset (Primary)
   - Location: raw/semantic_drone_dataset/
   - Images: 400 high-resolution (6000x4000)
   - Classes: 24 semantic classes
   - Purpose: Primary semantic learning
   - Status: $([ -d "raw/semantic_drone_dataset" ] && echo "✅ Downloaded" || echo "❌ Missing")

2. Urban Drone Dataset (UDD6) (Secondary)
   - Location: raw/udd6_dataset/
   - Images: ~200 urban aerial images
   - Classes: 6 urban classes
   - Purpose: Domain adaptation
   - Status: $([ -d "raw/udd6_dataset" ] && echo "✅ Downloaded" || echo "⚠️  Manual download needed")

3. DroneDeploy Dataset (Tertiary)
   - Location: raw/dronedeploy_dataset/
   - Images: ~50 high-resolution images
   - Classes: 6 landing-specific classes
   - Purpose: Landing specialization
   - Status: $([ -d "raw/dronedeploy_dataset" ] && echo "✅ Downloaded" || echo "⚠️  Manual download needed")

Training Pipeline:
Stage 1: Semantic Drone Dataset (foundation)
Stage 2: Domain adaptation with UDD6
Stage 3: Landing specialization with DroneDeploy
Stage 4: Multi-dataset refinement

Next Steps:
1. Run data preprocessing: python preprocess_datasets.py
2. Start training: python train_a100.py
3. Monitor progress: ./monitor_training.sh
EOF

# =============================================================================
# 6. Dataset Statistics
# =============================================================================
echo ""
echo "📈 Generating dataset statistics..."

python << 'EOF'
import os
from pathlib import Path

def count_files(directory, extensions=['.jpg', '.jpeg', '.png', '.tiff']):
    if not os.path.exists(directory):
        return 0
    
    count = 0
    for ext in extensions:
        count += len(list(Path(directory).rglob(f'*{ext}')) + 
                    list(Path(directory).rglob(f'*{ext.upper()}')))
    return count

def get_dir_size(directory):
    if not os.path.exists(directory):
        return 0
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024**3)  # Convert to GB

print("\n📊 Dataset Statistics:")
print("=" * 50)

datasets = {
    "Semantic Drone": "raw/semantic_drone_dataset",
    "UDD6": "raw/udd6_dataset", 
    "DroneDeploy": "raw/dronedeploy_dataset"
}

total_images = 0
total_size = 0

for name, path in datasets.items():
    images = count_files(path)
    size = get_dir_size(path)
    status = "✅" if images > 0 else "❌"
    
    print(f"{status} {name}:")
    print(f"   📁 Path: {path}")
    print(f"   🖼️  Images: {images}")
    print(f"   💾 Size: {size:.2f} GB")
    print()
    
    total_images += images
    total_size += size

print(f"📊 Total Images: {total_images}")
print(f"💾 Total Size: {total_size:.2f} GB")
print(f"🎯 Ready for training: {'✅ Yes' if total_images > 100 else '❌ Need more data'}")
EOF

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "🎉 Dataset Download Complete!"
echo "============================="
echo ""
echo "📁 Dataset Structure:"
echo "   datasets/"
echo "   ├── raw/"
echo "   │   ├── semantic_drone_dataset/"
echo "   │   ├── udd6_dataset/"
echo "   │   └── dronedeploy_dataset/"
echo "   ├── processed/"
echo "   └── dataset_info.txt"
echo ""
echo "📋 Next Steps:"
echo "1. Review dataset_info.txt for download status"
echo "2. If any datasets failed, download manually:"
echo "   - UDD6: https://sites.google.com/view/zhyzheng/udd"
echo "   - DroneDeploy: Search Kaggle or original source"
echo "3. Run preprocessing: python preprocess_datasets.py"
echo "4. Start training: python train_a100.py"
echo ""
echo "✅ Ready for UAV landing detection training!" 