#!/bin/bash
# Install Scallop for UAV Landing Project

set -e

echo "🚀 Installing Scallop for UAV Landing Project..."

# Check if Rust is installed
if ! command -v rustc &> /dev/null; then
    echo "📦 Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
    rustup default nightly
else
    echo "✅ Rust already installed"
fi

# Create a temporary directory for installation
TEMP_DIR=$(mktemp -d)
cd $TEMP_DIR

echo "📥 Downloading Scallop..."

# Download Scallop source
git clone https://github.com/scallop-lang/scallop.git
cd scallop

echo "🔨 Building Scallop..."

# Build Scallop components
make install-scli
make install-sclc
make install-sclrepl

echo "🐍 Installing Python bindings..."

# Try to install the Python package
cd etc/scallopy

# Build Python package from source
python setup.py build
python setup.py install

echo "✅ Scallop installation completed!"

# Clean up
cd /
rm -rf $TEMP_DIR

echo "🧪 Testing Scallop installation..."
scli --version || echo "⚠️  scli not found in PATH, you may need to add ~/.cargo/bin to your PATH"

echo "🎉 Scallop setup complete!"
echo "💡 If scli is not found, run: export PATH=\"\$HOME/.cargo/bin:\$PATH\""
