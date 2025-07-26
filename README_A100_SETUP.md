# A100 GPU Pod Setup for UAV Landing Detection Training

Complete setup guide for training the UAV Landing Detection system on your A100 GPU pod instance.

## 🚀 Quick Start

1. **SSH into your A100 pod**
2. **Run setup script**
3. **Download datasets**
4. **Start training**
5. **Monitor progress**
6. **Sync results**

## 📋 Prerequisites

- A100 GPU pod instance (40GB+ VRAM recommended)
- SSH access to the pod
- Kaggle account with API token
- Local machine with `rsync` and `ssh` for syncing

## 🛠️ Step-by-Step Setup

### Step 1: Initial Pod Setup

SSH into your A100 pod and run the setup script:

```bash
# SSH into your pod
ssh root@your_pod_ip

# Download and run setup script
wget https://your-repo/setup_a100_pod.sh
chmod +x setup_a100_pod.sh
./setup_a100_pod.sh
```

This script will:
- Install system dependencies (Python, CUDA tools, monitoring utilities)
- Create Python virtual environment with A100 optimizations
- Install PyTorch with CUDA 12.1 support
- Install training dependencies (OpenCV, Albumentations, W&B, etc.)
- Setup GPU monitoring tools
- Create directory structure

### Step 2: Upload Kaggle Credentials

Get your Kaggle API token:
1. Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
2. Click "Create New API Token"
3. Download `kaggle.json`

Upload to your pod:
```bash
# From your local machine
scp kaggle.json root@your_pod_ip:~/.kaggle/kaggle.json

# Or upload manually and set permissions
ssh root@your_pod_ip
# Upload kaggle.json to ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Step 3: Download Datasets

```bash
cd ~/uav_landing_system
source venv/bin/activate
./download_datasets.sh
```

This downloads:
- **Semantic Drone Dataset** (~2.8GB): 400 high-res images, 24 classes
- **Urban Drone Dataset (UDD6)**: Urban aerial imagery
- **DroneDeploy Dataset**: High-resolution landing-specific data

### Step 4: Preprocess Datasets

```bash
python preprocess_datasets.py
```

This will:
- Map 24 semantic classes to 4 landing classes
- Resize images to 512×512
- Create train/validation splits
- Generate dataset statistics and visualizations
- Create optimized data structure for training

### Step 5: Start Training

```bash
# Create training session (runs in background)
./start_training_session.sh

# Attach to training session
tmux attach -t uav_training

# Start training (inside tmux session)
python train_a100.py
```

Training features:
- **A100 optimized**: TF32, large batch sizes, mixed precision
- **Advanced monitoring**: W&B integration, comprehensive logging
- **Checkpoint management**: Automatic saving, best model tracking
- **Professional pipeline**: Safety-aware loss, uncertainty quantification

### Step 6: Monitor Training

In a separate terminal/tab:

```bash
# GPU monitoring dashboard
./monitor_training.sh

# Or individual commands
nvidia-smi watch -n 2
htop
```

Monitor training progress:
- **Local monitoring**: GPU utilization, memory usage, temperatures
- **W&B dashboard**: Training curves, metrics, system stats
- **Checkpoint files**: Automatic saving every 10 epochs

### Step 7: Sync Results

From your local machine:

```bash
# Download sync script
wget https://your-repo/sync_results.sh
chmod +x sync_results.sh

# Sync training results
./sync_results.sh your_pod_ip root ./a100_results
```

This syncs:
- Trained models and checkpoints
- Training logs and metrics
- W&B experiment data
- Analysis and visualization scripts

## 📊 Training Configuration

### A100 Optimized Settings

```python
# Optimized for A100 40GB
batch_size = 32          # Large batch for 40GB memory
epochs = 150             # Professional training duration  
learning_rate = 1e-3     # Optimal for large batches
mixed_precision = True   # A100 TF32 optimizations
```

### Model Options

- **Enhanced BiSeNetV2**: 6M+ parameters, fast inference
- **Enhanced DeepLabV3+**: Higher accuracy, more memory

### Dataset Pipeline

1. **Stage 1**: Semantic Drone Dataset (foundation learning)
2. **Stage 2**: Domain adaptation with UDD6 
3. **Stage 3**: Landing specialization with DroneDeploy
4. **Stage 4**: Multi-dataset refinement

## 🔧 Troubleshooting

### Common Issues

**GPU Not Detected**
```bash
# Check NVIDIA drivers
nvidia-smi

# Verify CUDA installation  
nvcc --version

# Test PyTorch GPU support
python -c "import torch; print(torch.cuda.is_available())"
```

**Out of Memory**
```bash
# Reduce batch size in train_a100.py
batch_size = 16  # Instead of 32

# Enable gradient accumulation
accumulation_steps = 2
```

**Kaggle Download Fails**
```bash
# Verify API token
kaggle datasets list

# Check permissions
ls -la ~/.kaggle/kaggle.json
# Should show: -rw------- (600 permissions)

# Manual download if needed
kaggle datasets download -d bulentsiyah/semantic-drone-dataset
```

**Training Disconnects**
```bash
# Training runs in tmux - reconnect anytime
tmux attach -t uav_training

# Check if training is still running
ps aux | grep python

# View latest logs
tail -f logs/training.log
```

### Performance Optimization

**Maximize A100 Usage**
```python
# Enable TF32 (automatic in our setup)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use persistent workers
persistent_workers = True
num_workers = 8
```

**Memory Management**
```python
# Mixed precision training (enabled by default)
use_amp = True

# Gradient clipping
gradient_clipping = 1.0

# Memory cleanup
torch.cuda.empty_cache()
```

## 📈 Expected Performance

### Training Times (A100 40GB)

- **Semantic Drone Dataset**: ~2-3 hours (400 images, 150 epochs)
- **Multi-dataset training**: ~4-6 hours total
- **Preprocessing**: ~15-30 minutes

### Model Performance

- **Training mIoU**: 85-90%
- **Validation mIoU**: 80-85%
- **Inference speed**: 60-120 FPS (A100)
- **Model size**: ~25MB (ONNX export)

### Hardware Utilization

- **GPU Memory**: 25-35GB / 40GB (optimal)
- **GPU Utilization**: 85-95%
- **Training efficiency**: ~1000 samples/second

## 🔗 Useful Commands

### Session Management
```bash
# Create training session
./start_training_session.sh

# List sessions
tmux list-sessions

# Attach to session
tmux attach -t uav_training

# Detach from session (Ctrl+B, then D)
```

### Monitoring
```bash
# GPU status
nvidia-smi

# Continuous monitoring
watch -n 2 nvidia-smi

# System resources
htop
df -h

# Training logs
tail -f logs/training.log
```

### Data Management
```bash
# Check dataset sizes
du -sh datasets/*/

# View preprocessing results
cat datasets/processed/preprocessing_summary.json | jq

# Check training splits
wc -l datasets/splits/*.txt
```

### Model Management
```bash
# List checkpoints
ls -lh checkpoints/

# Check model size
ls -lh outputs/final_model.pth

# Load model (Python)
python -c "import torch; model=torch.load('checkpoints/best_checkpoint.pth'); print('Model loaded')"
```

## 📋 File Structure

```
~/uav_landing_system/
├── datasets/
│   ├── raw/                    # Downloaded datasets
│   ├── processed/              # Preprocessed data  
│   ├── splits/                 # Train/val splits
│   └── cache/                  # Cached data
├── checkpoints/                # Model checkpoints
├── outputs/                    # Final models
├── logs/                       # Training logs
├── wandb/                      # W&B experiment data
├── results/                    # Analysis results
├── setup_a100_pod.sh          # Setup script
├── download_datasets.sh        # Dataset download
├── preprocess_datasets.py      # Data preprocessing
├── train_a100.py              # Main training script
├── monitor_training.sh         # GPU monitoring
└── sync_results.sh            # Results sync
```

## 🎯 Next Steps After Training

1. **Analyze Results**: Run analysis scripts on synced data
2. **Model Evaluation**: Test on validation data
3. **ONNX Export**: Convert for production deployment
4. **Integration**: Use with neurosymbolic system
5. **Deployment**: Deploy to edge devices

## 🆘 Support

If you encounter issues:

1. **Check logs**: Training logs contain detailed error information
2. **GPU monitoring**: Ensure GPU is being utilized properly  
3. **Memory usage**: Monitor for out-of-memory errors
4. **Dataset issues**: Verify dataset downloads completed
5. **Network issues**: Check connection for W&B logging

## 📝 Configuration Files

### Training Configuration
```python
# Key settings in train_a100.py
class A100TrainingConfig:
    batch_size = 32
    epochs = 150
    learning_rate = 1e-3
    model_name = 'enhanced_bisenetv2'
    use_amp = True
    use_wandb = True
```

### Preprocessing Configuration  
```python
# Key settings in preprocess_datasets.py
class PreprocessingConfig:
    target_size = (512, 512)
    train_val_split = 0.8
    landing_classes = {
        0: "safe_landing",
        1: "unsafe_structure", 
        2: "unsafe_vehicle",
        3: "vegetation"
    }
```

---

## ✅ Quick Checklist

- [ ] SSH access to A100 pod working
- [ ] Setup script completed successfully
- [ ] Kaggle credentials uploaded and configured
- [ ] Datasets downloaded (check with `du -sh datasets/raw/*/`)
- [ ] Preprocessing completed (check `datasets/processed/`)
- [ ] Training started in tmux session
- [ ] Monitoring setup working
- [ ] W&B logging functional (check wandb.ai)
- [ ] Sync script tested from local machine

**Ready for production-grade UAV landing detection training! 🚁** 