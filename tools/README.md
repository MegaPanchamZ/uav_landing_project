# Tools

This directory contains utility tools for system maintenance, testing, and performance analysis.

## Scripts

- **`verify_integration.py`** - System integration verification
  - Tests all critical imports work correctly
  - Verifies Scallop consolidation success
  - Confirms system readiness for deployment
  - Provides comprehensive integration status report

- **`benchmark_gpu.py`** - GPU performance benchmarking
  - Compares TensorRT vs CUDA vs CPU performance
  - Measures processing times and FPS
  - Tests different device configurations
  - Helps optimize system performance

## Usage

From the project root directory:

```bash
# Verify system integration
python tools/verify_integration.py

# Benchmark GPU performance
python tools/benchmark_gpu.py
```

These tools are essential for:
- System validation after changes
- Performance optimization 
- Troubleshooting integration issues
- Hardware configuration testing

All tools include proper error handling and detailed status reporting.
