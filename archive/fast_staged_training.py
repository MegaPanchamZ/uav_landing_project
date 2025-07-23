#!/usr/bin/env python3
"""
Fast Staged Fine-Tuning Script

Stage 1: BiSeNetV2 -> DroneDeploy (7 classes)
Stage 2: Stage1 Model -> UDD6 (4 landing classes)

Optimized for speed and efficiency with the limited datasets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
import time
import json
from tqdm import tqdm
import numpy as np

from staged_training import (
    DroneDeployDataset, UDDDataset, SimpleBiSeNetV2, 
    create_transforms, STAGE1_CLASSES, STAGE2_CLASSES
)

class FastTrainer:
    """Fast trainer for staged fine-tuning."""
    
    def __init__(self, device='auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
    def train_stage1(self, dataset_path, epochs=15, batch_size=4, lr=1e-4):
        """Stage 1: DroneDeploy intermediate fine-tuning."""
        
        print(f"\n🌍 STAGE 1: DroneDeploy Fine-Tuning")
        print("=" * 45)
        
        # Create datasets
        train_transform, val_transform = create_transforms()
        train_dataset = DroneDeployDataset(dataset_path, "train", train_transform)
        val_dataset = DroneDeployDataset(dataset_path, "val", val_transform)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=2, pin_memory=True)
        
        # Create model
        model = SimpleBiSeNetV2(num_classes=len(STAGE1_CLASSES))
        model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        # Training loop
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            model.train()
            train_loss = 0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            
            for batch in train_pbar:
                images = batch['image'].to(self.device, non_blocking=True)
                masks = batch['mask'].to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': loss.item()})
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    images = batch['image'].to(self.device, non_blocking=True)
                    masks = batch['mask'].to(self.device, non_blocking=True)
                    
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == masks).sum().item()
                    val_total += masks.numel()
            
            # Update metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            scheduler.step()
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1:2d}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.3f}, "
                  f"Time: {epoch_time:.1f}s")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 'stage1_best_model.pth')
                print(f"💾 Saved best Stage 1 model (val_loss: {avg_val_loss:.4f})")
        
        # Save final model and history
        torch.save(model.state_dict(), 'stage1_final_model.pth')
        with open('stage1_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f" Stage 1 complete! Best val loss: {best_val_loss:.4f}")
        return model, history
    
    def train_stage2(self, dataset_path, stage1_model_path, epochs=20, batch_size=4, lr=1e-5):
        """Stage 2: UDD6 task-specific fine-tuning."""
        
        print(f"\n🚁 STAGE 2: UDD6 Task-Specific Fine-Tuning") 
        print("=" * 50)
        
        # Create datasets
        train_transform, val_transform = create_transforms()
        train_dataset = UDDDataset(dataset_path, "train", train_transform)
        val_dataset = UDDDataset(dataset_path, "val", val_transform)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        if len(train_dataset) == 0:
            print("❌ No training samples found!")
            return None, None
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
        
        # Load Stage 1 model and adapt for Stage 2
        model = SimpleBiSeNetV2(num_classes=len(STAGE1_CLASSES))  # Original classes
        model.load_state_dict(torch.load(stage1_model_path, map_location='cpu'))
        print(f" Loaded Stage 1 model from {stage1_model_path}")
        
        # Replace classifier for Stage 2 classes
        model.classifier = nn.Conv2d(256, len(STAGE2_CLASSES), 1)
        model.to(self.device)
        
        # Use lower learning rate for fine-tuning
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_iou': []}
        
        # Training loop
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            model.train()
            train_loss = 0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            
            for batch in train_pbar:
                images = batch['image'].to(self.device, non_blocking=True)
                masks = batch['mask'].to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': loss.item()})
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            iou_sum = 0
            iou_count = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    images = batch['image'].to(self.device, non_blocking=True)
                    masks = batch['mask'].to(self.device, non_blocking=True)
                    
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    
                    # Calculate accuracy and IoU
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == masks).sum().item()
                    val_total += masks.numel()
                    
                    # Calculate mean IoU for landing classes
                    for class_id in [1, 2, 3]:  # Skip ignore class
                        pred_mask = (preds == class_id)
                        true_mask = (masks == class_id)
                        intersection = (pred_mask & true_mask).sum().float()
                        union = (pred_mask | true_mask).sum().float()
                        if union > 0:
                            iou_sum += intersection / union
                            iou_count += 1
            
            # Update metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            mean_iou = iou_sum / iou_count if iou_count > 0 else 0
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            history['val_iou'].append(mean_iou.item() if isinstance(mean_iou, torch.Tensor) else mean_iou)
            
            scheduler.step()
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1:2d}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.3f}, "
                  f"mIoU: {mean_iou:.3f}, Time: {epoch_time:.1f}s")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 'stage2_best_model.pth')
                print(f"💾 Saved best Stage 2 model (val_loss: {avg_val_loss:.4f})")
        
        # Save final model and history
        torch.save(model.state_dict(), 'stage2_final_model.pth')
        with open('stage2_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f" Stage 2 complete! Best val loss: {best_val_loss:.4f}")
        return model, history
    
    def export_to_onnx(self, model_path, output_path="uav_landing_model.onnx"):
        """Export the final model to ONNX format."""
        
        print(f"\n📦 Exporting to ONNX: {output_path}")
        
        # Load model
        model = SimpleBiSeNetV2(num_classes=len(STAGE2_CLASSES))
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # Export
        dummy_input = torch.randn(1, 3, 512, 512)
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f" Model exported to {output_path}")
        
        # Test ONNX model
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(output_path)
            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: dummy_input.numpy()})
            print(f" ONNX model verified: {output[0].shape}")
        except ImportError:
            print("⚠️  ONNX Runtime not available for verification")

def main():
    """Main training pipeline."""
    
    print("🛩️  Fast Staged Fine-Tuning Pipeline")
    print("=" * 45)
    
    trainer = FastTrainer()
    
    # Stage 1: DroneDeploy
    drone_deploy_path = "../datasets/drone_deploy_dataset_intermediate/dataset-medium"
    if Path(drone_deploy_path).exists():
        print("Starting Stage 1...")
        stage1_model, stage1_history = trainer.train_stage1(
            drone_deploy_path, epochs=10, batch_size=2, lr=1e-4
        )
        print("Stage 1 completed!")
    else:
        print(f"❌ DroneDeploy dataset not found: {drone_deploy_path}")
        return
    
    # Stage 2: UDD6
    udd_path = "../datasets/UDD/UDD/UDD6"
    if Path(udd_path).exists() and Path("stage1_best_model.pth").exists():
        print("Starting Stage 2...")
        stage2_model, stage2_history = trainer.train_stage2(
            udd_path, "stage1_best_model.pth", epochs=15, batch_size=2, lr=5e-5
        )
        
        if stage2_model is not None:
            # Export final model
            trainer.export_to_onnx("stage2_best_model.pth", "uav_landing_detector.onnx")
            print(" Staged training completed successfully!")
        else:
            print("❌ Stage 2 training failed")
    else:
        print(f"❌ UDD dataset not found or Stage 1 model missing")

if __name__ == "__main__":
    main()
