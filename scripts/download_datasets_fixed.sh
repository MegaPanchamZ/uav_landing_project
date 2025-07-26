#!/bin/bash
# Fixed Dataset Download Script for A100 Remote Machine
# ====================================================
#
# Downloads all three datasets with proper structure handling

set -e  # Exit on any error

echo "🚁 UAV Landing Dataset Download Script (Fixed)"
echo "=============================================="

# Create datasets directory
DATASETS_DIR="./datasets"
mkdir -p "$DATASETS_DIR"
cd "$DATASETS_DIR"

echo "📂 Working directory: $(pwd)"

# Check dependencies
if ! command -v kaggle &> /dev/null; then
    echo "📦 Installing Kaggle CLI..."
    pip install kaggle
fi

if ! command -v gdown &> /dev/null; then
    echo "📦 Installing gdown for Google Drive downloads..."
    pip install gdown
fi

echo ""
echo "🔐 Kaggle Authentication Check..."
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Kaggle credentials not found!"
    echo "Please set up Kaggle credentials:"
    echo "1. Go to https://www.kaggle.com/settings"
    echo "2. Create new API token"
    echo "3. Download kaggle.json"
    echo "4. Place it at ~/.kaggle/kaggle.json"
    echo "5. Run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
else
    echo "✅ Kaggle credentials found"
fi

echo ""
echo "📥 Dataset 1: Semantic Drone Dataset (Kaggle)"
echo "============================================="
SDD_DIR="semantic_drone_dataset"
if [ -d "$SDD_DIR" ] && [ -d "$SDD_DIR/original_images" ]; then
    echo "✅ $SDD_DIR already exists and is properly structured"
else
    echo "📥 Downloading Semantic Drone Dataset..."
    
    # Clean up any previous attempts
    rm -rf semantic_drone_temp/ semantic-drone-dataset.zip
    
    kaggle datasets download -d bulentsiyah/semantic-drone-dataset
    
    echo "📦 Extracting Semantic Drone Dataset..."
    unzip -q semantic-drone-dataset.zip -d semantic_drone_temp/
    
    # Handle the actual structure: semantic_drone_temp/dataset/semantic_drone_dataset/
    if [ -d "semantic_drone_temp/dataset/semantic_drone_dataset" ]; then
        mv semantic_drone_temp/dataset/semantic_drone_dataset "$SDD_DIR"
        echo "✅ Moved from nested structure"
    elif [ -d "semantic_drone_temp/semantic_drone_dataset" ]; then
        mv semantic_drone_temp/semantic_drone_dataset "$SDD_DIR"
        echo "✅ Moved from flat structure"
    else
        # Fallback: look for any directory with the right contents
        for dir in semantic_drone_temp/*/; do
            if [ -d "$dir/original_images" ] && [ -d "$dir/label_images_semantic" ]; then
                mv "$dir" "$SDD_DIR"
                echo "✅ Found and moved correct directory: $dir"
                break
            fi
        done
    fi
    
    # Cleanup
    rm -rf semantic_drone_temp/
    rm semantic-drone-dataset.zip
    
    if [ -d "$SDD_DIR/original_images" ]; then
        echo "✅ Semantic Drone Dataset downloaded successfully"
    else
        echo "❌ Failed to properly extract Semantic Drone Dataset"
        exit 1
    fi
fi

echo ""
echo "📥 Dataset 2: DroneDeploy Dataset (Kaggle)"
echo "=========================================="
DD_DIR="drone_deploy_dataset"
if [ -d "$DD_DIR" ] && [ "$(find $DD_DIR -name '*.tif' -o -name '*.jpg' -o -name '*.png' | wc -l)" -gt 10 ]; then
    echo "✅ $DD_DIR already exists with image files"
else
    echo "📥 Downloading DroneDeploy Dataset..."
    
    # Clean up any previous attempts
    rm -rf drone_deploy_temp/ drone-deploy-medium-dataset.zip
    
    kaggle datasets download -d mightyrains/drone-deploy-medium-dataset
    
    echo "📦 Extracting DroneDeploy Dataset..."
    unzip -q drone-deploy-medium-dataset.zip -d drone_deploy_temp/
    
    # Move contents to target directory
    mkdir -p "$DD_DIR"
    mv drone_deploy_temp/* "$DD_DIR/" 2>/dev/null || true
    
    # Cleanup
    rm -rf drone_deploy_temp/
    rm drone-deploy-medium-dataset.zip
    
    echo "✅ DroneDeploy Dataset downloaded"
fi

echo ""
echo "📥 Dataset 3: Urban Drone Dataset UDD6 (Google Drive)"
echo "==================================================="
UDD6_DIR="udd6_dataset"
if [ -d "$UDD6_DIR" ] && [ "$(find $UDD6_DIR -type f | wc -l)" -gt 10 ]; then
    echo "✅ $UDD6_DIR already exists with files"
else
    echo "📥 Downloading UDD6 Dataset from Google Drive..."
    
    # Clean up any previous attempts
    rm -rf udd6_temp/ udd6_dataset.zip
    
    # UDD6 Google Drive link
    UDD6_GDRIVE_ID="1BNL8HNFRiNjSzdcQJo-uXiejZJ6DgunY"
    
    gdown --id "$UDD6_GDRIVE_ID" --output udd6_dataset.zip
    
    echo "📦 Extracting UDD6 Dataset..."
    unzip -q udd6_dataset.zip -d udd6_temp/
    
    # Move contents to target directory
    mkdir -p "$UDD6_DIR"
    mv udd6_temp/* "$UDD6_DIR/" 2>/dev/null || true
    
    # Cleanup
    rm -rf udd6_temp/
    rm udd6_dataset.zip
    
    echo "✅ UDD6 Dataset downloaded"
fi

echo ""
echo "🔍 Dataset Verification"
echo "======================"

# Verify Semantic Drone Dataset
echo "📊 Semantic Drone Dataset:"
if [ -d "$SDD_DIR/original_images" ]; then
    img_count=$(find "$SDD_DIR/original_images" -name "*.jpg" | wc -l)
    echo "   ✅ Original images: $img_count files"
else
    echo "   ❌ Original images directory not found"
fi

if [ -d "$SDD_DIR/label_images_semantic" ]; then
    label_count=$(find "$SDD_DIR/label_images_semantic" -name "*.png" | wc -l)
    echo "   ✅ Semantic labels: $label_count files"
else
    echo "   ❌ Semantic labels directory not found"
fi

# Verify DroneDeploy Dataset
echo ""
echo "📊 DroneDeploy Dataset:"
if [ -d "$DD_DIR" ]; then
    dd_files=$(find "$DD_DIR" -type f | wc -l)
    echo "   ✅ Total files: $dd_files"
    
    # Check for common structures
    for subdir in "images" "labels" "dataset-medium"; do
        if [ -d "$DD_DIR/$subdir" ]; then
            subdir_files=$(find "$DD_DIR/$subdir" -type f | wc -l)
            echo "   ✅ $subdir/: $subdir_files files"
        fi
    done
else
    echo "   ❌ DroneDeploy directory not found"
fi

# Verify UDD6 Dataset
echo ""
echo "📊 UDD6 Dataset:"
if [ -d "$UDD6_DIR" ]; then
    udd6_files=$(find "$UDD6_DIR" -type f | wc -l)
    echo "   ✅ Total files: $udd6_files"
    
    # Check for common structures
    for subdir in "train" "val" "test" "src" "gt" "images" "labels"; do
        if [ -d "$UDD6_DIR/$subdir" ]; then
            subdir_files=$(find "$UDD6_DIR/$subdir" -type f | wc -l)
            echo "   ✅ $subdir/: $subdir_files files"
        fi
    done
else
    echo "   ❌ UDD6 directory not found"
fi

echo ""
echo "💾 Disk Usage Summary"
echo "===================="
echo "📁 Dataset sizes:"
du -sh "$SDD_DIR" 2>/dev/null || echo "   SDD: Not found"
du -sh "$DD_DIR" 2>/dev/null || echo "   DroneDeploy: Not found"
du -sh "$UDD6_DIR" 2>/dev/null || echo "   UDD6: Not found"

total_size=$(du -sh . | cut -f1)
echo "📊 Total dataset size: $total_size"

echo ""
echo "🎉 Dataset Download Complete!"
echo "=============================="

echo "📋 Next Steps:"
echo "1. Test dataset loaders:"
echo "   python -c \"from datasets.semantic_drone_dataset import SemanticDroneDataset; print('SDD OK')\""
echo "   python -c \"from datasets.dronedeploy_1024_dataset import DroneDeploy1024Dataset; print('DD OK')\""
echo "   python -c \"from datasets.udd6_dataset import UDD6Dataset; print('UDD6 OK')\""
echo ""
echo "2. Start progressive training:"
echo "   python scripts/train_a100_progressive_multi_dataset.py --stage 1 --sdd_data_root ./datasets/$SDD_DIR --use_wandb"
echo "   python scripts/train_a100_progressive_multi_dataset.py --stage 2 --dronedeploy_data_root ./datasets/$DD_DIR --use_wandb"
echo "   python scripts/train_a100_progressive_multi_dataset.py --stage 3 --udd6_data_root ./datasets/$UDD6_DIR --use_wandb"

echo ""
echo "✅ All datasets ready for progressive training! 🚁" 