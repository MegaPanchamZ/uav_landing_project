# Setup Scripts

This directory contains environment setup and configuration scripts for the UAV Landing System.

## Scripts

- **`setup.sh`** - Main system setup script
  - Installs Python dependencies
  - Configures virtual environment
  - Sets up basic system requirements

- **`setup_gpu.sh`** - GPU acceleration setup
  - Configures CUDA environment
  - Sets up GPU drivers and libraries
  - Prepares system for GPU-accelerated inference

- **`setup_tensorrt.sh`** - TensorRT optimization setup
  - Installs TensorRT runtime
  - Configures optimization libraries
  - Sets up high-performance inference

## Usage

Run these scripts in order for complete system setup:

```bash
# Basic system setup
bash setup/setup.sh

# GPU acceleration (optional, if you have NVIDIA GPU)
bash setup/setup_gpu.sh

# TensorRT optimization (optional, for maximum performance)
bash setup/setup_tensorrt.sh
```

## Requirements

- Ubuntu/Debian-based system
- Python 3.8+
- NVIDIA GPU (for GPU/TensorRT scripts)
- Appropriate CUDA drivers

## Notes

- Run scripts with appropriate permissions
- Check logs for any installation issues
- Some scripts may require sudo access
- GPU scripts require compatible NVIDIA hardware
