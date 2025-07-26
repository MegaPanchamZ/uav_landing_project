#!/bin/bash
echo "🚁 Setting up UAV Landing Training Environment"
echo "============================================="

# Install requirements
echo "📦 Installing Python requirements..."
pip install -r requirements.txt

# Make scripts executable
chmod +x scripts/*.sh

# Create output directories
mkdir -p outputs/a100_progressive
mkdir -p datasets

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Set up Kaggle credentials: ~/.kaggle/kaggle.json"
echo "2. Run: bash scripts/download_datasets.sh"
echo "3. Start training: python scripts/train_a100_progressive_multi_dataset.py --stage 1 ..."
