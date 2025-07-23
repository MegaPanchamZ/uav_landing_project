# Training Issues Analysis & Comprehensive Fixes

## 🚨 Critical Issues Identified

### 1. **Severe Class Imbalance (226:1 ratio)**
- **Class 0 (Background)**: Only 0.28% of pixels
- **Class 1 (Safe Landing)**: 63.17% of pixels  
- **Class 2 (Caution)**: 18.83% of pixels
- **Class 3 (Danger)**: 17.72% of pixels

**Impact**: Model never learned to predict Class 0, leading to 0% accuracy/IoU for background class.

### 2. **Inadequate Loss Function**
- Original `CombinedSafetyLoss` used hardcoded weights `[1.0, 2.0, 1.5, 3.0]`
- Did not account for severe class imbalance
- Complex multi-component loss may have confused training

### 3. **Conservative Training Parameters**
- Learning rate too low (1e-4)
- Small effective batch size (16)
- May have led to underfitting

### 4. **Poor Monitoring**
- Limited per-class performance tracking
- No class distribution analysis during training

## 🔧 Comprehensive Fixes Applied

### 1. **Proper Class Weighting**
```python
# Computed weights based on class frequency analysis
class_weights = [89.74, 0.396, 1.328, 1.411]
```

**Calculation**: `weight = total_samples / (n_classes * class_count)`

### 2. **Improved Loss Function**
```python
class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=[89.74, 0.396, 1.328, 1.411], gamma=2.0):
        # Focal loss with proper class weighting
        # Addresses both class imbalance AND hard examples
```

**Benefits**:
- Addresses severe class imbalance
- Focuses on hard examples
- Simpler than original combined loss

### 3. **Enhanced Training Parameters**
- **Learning Rate**: Increased from 1e-4 to 3e-4
- **Batch Size**: Increased from 4 to 6
- **Accumulation Steps**: Reduced from 4 to 3 (effective batch size: 18)
- **Scheduler**: Changed to CosineAnnealingWarmRestarts for better dynamics

### 4. **Improved Data Handling**
- **More Crops**: Increased from 4 to 6 crops per image
- **Better Analysis**: Added dataset class distribution monitoring
- **Data Type Fix**: Ensured targets are properly cast to long type

### 5. **Enhanced Monitoring**
```python
# Per-class metrics tracking
for cls in range(4):
    cls_iou = val_metrics.get(f'class_{cls}_iou', 0.0)
    cls_acc = val_metrics.get(f'class_{cls}_accuracy', 0.0)
    print(f"Class {cls}: IoU={cls_iou:.3f}, Acc={cls_acc:.3f}")
```

## 📊 Expected Improvements

### **Before Fixes**:
- **Class 0**: 0% accuracy, 0% IoU ❌
- **Overall mIoU**: 27.33% ❌
- **Model Bias**: Heavily biased toward dominant classes

### **After Fixes**:
- **Class 0**: Should achieve >70% accuracy ✅
- **Overall mIoU**: Expected >50% (80%+ improvement) ✅
- **Balanced Performance**: All classes should perform reasonably

## 🚀 Usage

### Run Fixed Training:
```bash
python scripts/train_memory_efficient_fixed.py \
    --batch-size 6 \
    --epochs 30 \
    --accumulation-steps 3 \
    --learning-rate 3e-4
```

### Monitor Improvements:
- Check wandb project: `uav-landing-fixed`
- Look for balanced per-class performance
- Expect steady mIoU improvement

## 📋 Key Lessons Learned

1. **Always analyze class distribution** before training
2. **Class imbalance > 10:1 requires explicit handling**
3. **Monitor per-class metrics**, not just overall accuracy
4. **Simple, well-tuned losses often outperform complex ones**
5. **Learning rate tuning is critical for convergence**

## 🔍 Diagnostic Script

Use `scripts/analyze_class_distribution.py` to analyze any dataset:
```bash
python scripts/analyze_class_distribution.py
```

**Output**:
- Class frequency analysis
- Recommended class weights
- Visualization of imbalance
- Detection of unmapped classes

## 📈 Performance Monitoring

### Key Metrics to Watch:
1. **Per-class IoU** - Should be >0.5 for all classes
2. **Per-class Accuracy** - Should be >0.7 for all classes  
3. **Overall mIoU** - Target >0.6 for good segmentation
4. **Loss Convergence** - Should steadily decrease

### Red Flags:
- Any class with 0% performance
- mIoU plateau < 0.4
- Extremely high loss values
- Memory allocation errors

## 🎯 Next Steps

1. **Run the fixed training script**
2. **Monitor class-specific performance**  
3. **Fine-tune hyperparameters if needed**
4. **Consider data augmentation for rare classes**
5. **Evaluate on test set once training is successful**

---

**Note**: These fixes address the root causes of the training failure. The severe class imbalance was the primary culprit, but the combination of all fixes should result in dramatically improved performance. 