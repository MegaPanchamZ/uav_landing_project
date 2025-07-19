# 🚁 UAV Landing System - Neuro-Symbolic Intelligence for Safe Autonomous Landing

A **production-ready UAV landing detection system** that combines fine-tuned deep learning with rule-based reasoning for intelligent, traceable, and safe autonomous landing decisions.

## 🌟 Key Features

- **🧠 Neuro-Symbolic Intelligence**: Deep learning + rule-based reasoning for robust decision making
- **🔍 Full Traceability**: Complete decision path logging with explainable AI 
- **⚡ Real-Time Processing**: 7-127 FPS with ONNX optimization
- **🎯 Plug & Play Interface**: Simple 3-line integration into any UAV system
- **🛡️ Safety-First Design**: Risk assessment and automatic abort mechanisms
- **📊 Real-World Validated**: Tested on actual UAV imagery from UDD dataset
- **🔧 Production Ready**: Error handling, logging, and performance monitoring

## 🚀 Quick Start (3 Lines of Code!)

```python
from uav_landing_system import process_image_for_landing
import cv2

# 1. Load your UAV image
image = cv2.imread("your_uav_frame.jpg")

# 2. Process for landing detection  
result = process_image_for_landing(image, altitude=5.0, enable_tracing=True)

# 3. Get intelligent landing decision
print(f"Decision: {result.status} | Confidence: {result.confidence:.3f} | Explanation: {result.decision_explanation}")
```

**Output**: `Decision: TARGET_ACQUIRED | Confidence: 0.847 | Explanation: High-quality landing zone detected with excellent safety margins`

## 📖 Complete Usage Guide

🎯 **For detailed plug & play instructions, examples, and configuration options, see**: [`USAGE_GUIDE.md`](USAGE_GUIDE.md)

The usage guide covers:
- **Installation & Setup**: One-command installation
- **Basic Usage**: Simple examples for immediate use
- **Advanced Features**: Custom models, real-time video, batch processing  
- **Neuro-Symbolic Traceability**: Full decision explanation and risk assessment
- **Production Deployment**: Error handling and monitoring
- **Configuration**: Custom neural/symbolic weights and safety thresholds

## 🧠 How It Works

### Neuro-Symbolic Architecture
```
UAV Image → Neural Network → Symbolic Reasoning → Decision Fusion → Safe Landing Decision
    ↓            (40% weight)      (60% weight)         ↓              ↓
256×256 RGB   Segmentation     Safety Rules      Weighted Score   TARGET_ACQUIRED
              Confidence       Risk Assessment   + Explanation    + Coordinates
```

### Intelligence Layers
1. **Neural Component**: Fine-tuned BiSeNetV2 for semantic segmentation
2. **Symbolic Component**: Rule-based safety analysis and risk assessment  
3. **Decision Fusion**: Weighted integration with safety overrides
4. **Traceability**: Complete decision path recording for explainability

## 🔬 Model Performance

- **Model**: BiSeNetV2 with custom UAV landing head (1.3MB ONNX)
- **Training**: Multi-stage fine-tuning (CITYSCAPES → UDD → DRONEDEPLOY)
- **Input**: 256×256 RGB images
- **Output**: 4-class semantic segmentation + neuro-symbolic analysis
- **Speed**: 7-127 FPS (depending on hardware)
- **Validation**: Real UDD dataset imagery (2160×4096 resolution)

## 🧪 Testing & Validation

Run comprehensive tests:
```bash
# Quick functionality test
python tests/quick_test.py

# Real model test with ONNX
python tests/test_real_model.py

# Neuro-symbolic reasoning with real UDD data
python tests/integration/test_udd_neuro_symbolic.py
```

**Validated Performance**:
- ✅ Real-world UAV imagery processing (UDD dataset)  
- ✅ Neuro-symbolic risk assessment and safety recommendations
- ✅ Processing time: 330-530ms on high-resolution images
- ✅ Realistic confidence calibration for production use

## 📁 Project Structure

```
uav_landing_project/
├── uav_landing_system.py      # 🎯 Main plug & play interface
├── uav_landing_detector.py    # 🧠 Core neuro-symbolic detector  
├── models/                    # 🤖 Fine-tuned ONNX models
├── tests/                     # 🧪 Comprehensive test suite
│   ├── integration/           # Real dataset validation
│   └── quick_test.py          # Basic functionality tests
├── training_tools/            # 🛠️ Fine-tuning pipelines
├── USAGE_GUIDE.md            # 📖 Complete plug & play guide
└── requirements.txt           # 📦 Dependencies
```

## 🎯 Production-Ready Features

- **🔌 Plug & Play**: Simple integration with existing UAV systems
- **📊 Performance Monitoring**: Processing time tracking and FPS monitoring  
- **🔍 Full Traceability**: Decision paths, risk assessments, and recommendations
- **⚠️ Safety Systems**: Risk level assessment and automatic abort mechanisms
- **🛠️ Error Handling**: Robust exception handling and recovery mechanisms
- **📝 Comprehensive Logging**: Configurable logging levels for development and production

## 🚀 Ready for Deployment

This system is **production-ready** and has been validated on real UAV imagery. Whether you're building:
- **Research UAVs**: Full traceability and detailed analysis
- **Racing Drones**: High-speed processing with minimal overhead
- **Commercial Systems**: Production-grade error handling and monitoring
- **Educational Projects**: Easy plug & play interface with comprehensive documentation

**Start landing safely in 3 lines of code!** 🚁🎯

## 📚 Documentation

- [`USAGE_GUIDE.md`](USAGE_GUIDE.md) - Complete plug & play guide with examples
- `docs/TRAINING.md` - Model fine-tuning process  
- `docs/API.md` - Full API reference
- `docs/ARCHITECTURE.md` - System design details

## 🔗 Quick Links

- **🎯 [Get Started → USAGE_GUIDE.md](USAGE_GUIDE.md)**
- **🧪 [Run Tests → tests/](tests/)**
- **🤖 [Model Files → models/](models/)**
- **🛠️ [Training Tools → training_tools/](training_tools/)**

---

*Built with ❤️ for safe autonomous UAV operations*