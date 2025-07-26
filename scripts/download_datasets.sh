#!/bin/bash
# Dataset Download Script for A100 Remote Machine
# ===============================================
#
# Downloads all three datasets needed for progressive training:
# 1. Semantic Drone Dataset (Kaggle)
# 2. DroneDeploy Dataset (Kaggle) 
# 3. Urban Drone Dataset UDD6 (Google Drive)

set -e  # Exit on any error

echo "🚁 UAV Landing Dataset Download Script"
echo "======================================"

# Create datasets directory
DATASETS_DIR="./datasets"
mkdir -p "$DATASETS_DIR"
cd "$DATASETS_DIR"

echo "📂 Working directory: $(pwd)"

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "📦 Installing Kaggle CLI..."
    pip install kaggle
fi

# Check if gdown is installed for Google Drive
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
if [ -d "$SDD_DIR" ]; then
    echo "⚠️  $SDD_DIR already exists, skipping download"
else
    echo "📥 Downloading Semantic Drone Dataset..."
    kaggle datasets download -d bulentsiyah/semantic-drone-dataset
    
    echo "📦 Extracting Semantic Drone Dataset..."
    unzip -q semantic-drone-dataset.zip -d semantic_drone_temp/
    mv semantic_drone_temp/* "$SDD_DIR/"
    rm -rf semantic_drone_temp/
    rm semantic-drone-dataset.zip
    
    echo "✅ Semantic Drone Dataset downloaded to: $SDD_DIR"
    echo "   Structure check:"
    ls -la "$SDD_DIR"
fi

echo ""
echo "📥 Dataset 2: DroneDeploy Dataset (Kaggle)"
echo "=========================================="
DD_DIR="drone_deploy_dataset"
if [ -d "$DD_DIR" ]; then
    echo "⚠️  $DD_DIR already exists, skipping download"
else
    echo "📥 Downloading DroneDeploy Dataset..."
    kaggle datasets download -d mightyrains/drone-deploy-medium-dataset
    
    echo "📦 Extracting DroneDeploy Dataset..."
    unzip -q drone-deploy-medium-dataset.zip -d drone_deploy_temp/
    mv drone_deploy_temp/* "$DD_DIR/"
    rm -rf drone_deploy_temp/
    rm drone-deploy-medium-dataset.zip
    
    echo "✅ DroneDeploy Dataset downloaded to: $DD_DIR"
    echo "   Structure check:"
    ls -la "$DD_DIR"
fi

echo ""
echo "📥 Dataset 3: Urban Drone Dataset UDD6 (Google Drive)"
echo "==================================================="
UDD6_DIR="udd6_dataset"
if [ -d "$UDD6_DIR" ]; then
    echo "⚠️  $UDD6_DIR already exists, skipping download"
else
    echo "📥 Downloading UDD6 Dataset from Google Drive..."
    
    # UDD6 Google Drive link (from the provided info)
    UDD6_GDRIVE_ID="1BNL8HNFRiNjSzdcQJo-uXiejZJ6DgunY"
    
    gdown --id "$UDD6_GDRIVE_ID" --output udd6_dataset.zip
    
    echo "📦 Extracting UDD6 Dataset..."
    unzip -q udd6_dataset.zip -d udd6_temp/
    mv udd6_temp/* "$UDD6_DIR/"
    rm -rf udd6_temp/
    rm udd6_dataset.zip
    
    echo "✅ UDD6 Dataset downloaded to: $UDD6_DIR"
    echo "   Structure check:"
    ls -la "$UDD6_DIR"
fi

echo ""
echo "🔍 Dataset Verification"
echo "======================"

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

echo ""
echo "📊 DroneDeploy Dataset:"
if [ -d "$DD_DIR" ]; then
    dd_files=$(find "$DD_DIR" -type f | wc -l)
    echo "   ✅ Total files: $dd_files"
    
    # Check for images and labels directories
    if [ -d "$DD_DIR/images" ]; then
        dd_images=$(find "$DD_DIR/images" -name "*.tif" -o -name "*.jpg" -o -name "*.png" | wc -l)
        echo "   ✅ Images: $dd_images files"
    fi
    
    if [ -d "$DD_DIR/labels" ]; then
        dd_labels=$(find "$DD_DIR/labels" -name "*.png" -o -name "*.tif" | wc -l)
        echo "   ✅ Labels: $dd_labels files"
    fi
else
    echo "   ❌ DroneDeploy directory not found"
fi

echo ""
echo "📊 UDD6 Dataset:"
if [ -d "$UDD6_DIR" ]; then
    udd6_files=$(find "$UDD6_DIR" -type f | wc -l)
    echo "   ✅ Total files: $udd6_files"
    
    # Look for various possible structures
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
echo "1. Verify dataset structures with dataset loaders"
echo "2. Run progressive training stages:"
echo "   Stage 1: python train_a100_progressive_multi_dataset.py --stage 1 --sdd_data_root ./datasets/$SDD_DIR"
echo "   Stage 2: python train_a100_progressive_multi_dataset.py --stage 2 --dronedeploy_data_root ./datasets/$DD_DIR"  
echo "   Stage 3: python train_a100_progressive_multi_dataset.py --stage 3 --udd6_data_root ./datasets/$UDD6_DIR"

echo ""
echo "💡 Tips:"
echo "- Use tmux/screen for long training sessions"
echo "- Monitor GPU usage with nvidia-smi"
echo "- Check wandb for training progress"
echo "- Large images will be automatically converted to chips during training"

echo "✅ All done! Ready for progressive training on A100 🚁" 