#!/usr/bin/env python3
"""
Enhanced UAV Landing Model Training Script
==========================================

Production-ready training script using the enhanced pipeline:
- Multi-dataset integration (Semantic Drone + UDD + DroneDeploy)
- Proper capacity models (6M+ parameters)
- Safety-aware loss functions
- Uncertainty quantification
- Professional evaluation metrics
- Cross-domain validation

This addresses all critical inadequacies identified in the original approach.
"""

import argparse
import json
import sys
from pathlib import Path
import torch

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from training.enhanced_training_pipeline import EnhancedTrainingPipeline, create_training_config


def main():
    """Main training function with comprehensive configuration."""
    
    parser = argparse.ArgumentParser(
        description="Enhanced UAV Landing Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with enhanced pipeline
  python train_enhanced_model.py --semantic-drone-path ../datasets/semantic_drone_dataset

  # High-quality training with all datasets
  python train_enhanced_model.py \\
    --semantic-drone-path ../datasets/semantic_drone_dataset \\
    --udd-path ../datasets/UDD/UDD/UDD6 \\
    --drone-deploy-path ../datasets/drone_deploy_dataset_intermediate/dataset-medium \\
    --model-type deeplabv3plus \\
    --training-mode high_quality

  # Fast training for development
  python train_enhanced_model.py \\
    --semantic-drone-path ../datasets/semantic_drone_dataset \\
    --training-mode fast \\
    --epochs 30
        """
    )
    
    # Dataset paths
    parser.add_argument(
        '--semantic-drone-path', 
        type=str,
        required=True,
        help='Path to Semantic Drone Dataset (primary dataset)'
    )
    parser.add_argument(
        '--udd-path', 
        type=str,
        help='Path to UDD6 dataset (optional secondary dataset)'
    )
    parser.add_argument(
        '--drone-deploy-path', 
        type=str,
        help='Path to DroneDeploy dataset (optional tertiary dataset)'
    )
    
    # Model configuration
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['enhanced_bisenetv2', 'deeplabv3plus'],
        default='enhanced_bisenetv2',
        help='Model architecture to use'
    )
    parser.add_argument(
        '--backbone',
        type=str,
        choices=['resnet50', 'resnet101', 'efficientnet-b2'],
        default='resnet50',
        help='Backbone architecture'
    )
    parser.add_argument(
        '--pretrained-model',
        type=str,
        help='Path to pretrained model weights'
    )
    
    # Training configuration
    parser.add_argument(
        '--training-mode',
        type=str,
        choices=['fast', 'comprehensive', 'high_quality'],
        default='comprehensive',
        help='Training mode preset'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (overrides mode preset)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Training batch size (overrides mode preset)'
    )
    parser.add_argument(
        '--input-resolution',
        type=int,
        nargs=2,
        metavar=('HEIGHT', 'WIDTH'),
        help='Input image resolution (height width)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Learning rate for head layers'
    )
    
    # Loss configuration
    parser.add_argument(
        '--loss-type',
        type=str,
        choices=['combined_safety', 'focal', 'cross_entropy'],
        default='combined_safety',
        help='Loss function type'
    )
    parser.add_argument(
        '--safety-weights',
        type=float,
        nargs=4,
        metavar=('BG', 'SAFE', 'CAUTION', 'DANGER'),
        default=[1.0, 2.0, 1.5, 3.0],
        help='Safety weights for each class'
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/enhanced_training',
        help='Output directory for models and logs'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        help='Experiment name for logging'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    
    # Hardware configuration
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='Training device'
    )
    
    # Validation and testing
    parser.add_argument(
        '--cross-domain-validation',
        action='store_true',
        help='Enable cross-domain validation testing'
    )
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save prediction visualizations'
    )
    
    args = parser.parse_args()
    
    # Validate dataset paths
    if not Path(args.semantic_drone_path).exists():
        print(f"❌ Semantic Drone Dataset not found: {args.semantic_drone_path}")
        return 1
    
    # Create training configuration
    config = create_training_config(
        model_type=args.model_type,
        training_mode=args.training_mode
    )
    
    # Update config with command line arguments
    config['semantic_drone_path'] = args.semantic_drone_path
    if args.udd_path:
        config['udd_path'] = args.udd_path
    if args.drone_deploy_path:
        config['drone_deploy_path'] = args.drone_deploy_path
    
    # Model configuration updates
    config['model']['type'] = args.model_type
    config['model']['kwargs']['backbone'] = args.backbone
    
    # Handle pretrained model selection
    if args.pretrained_model:
        config['model']['pretrained_path'] = args.pretrained_model
    else:
        # Auto-detect best available BiSeNetV2 model from model_pths
        config['model']['pretrained_path'] = "auto_detect_cityscapes_bisenetv2"
    
    # Training configuration updates
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.input_resolution:
        config['input_resolution'] = args.input_resolution
    if args.learning_rate:
        config['optimizer']['head_lr'] = args.learning_rate
        config['optimizer']['backbone_lr'] = args.learning_rate * 0.1
    
    # Loss configuration updates
    config['loss']['type'] = args.loss_type
    config['loss']['safety_weights'] = args.safety_weights
    
    # Hardware configuration
    config['num_workers'] = args.num_workers
    if args.device != 'auto':
        config['device'] = args.device
    
    # Print configuration summary
    print("🚀 Enhanced UAV Landing Training Configuration")
    print("=" * 60)
    print(f"📊 Datasets:")
    print(f"   Primary: {args.semantic_drone_path}")
    if args.udd_path:
        print(f"   Secondary: {args.udd_path}")
    if args.drone_deploy_path:
        print(f"   Tertiary: {args.drone_deploy_path}")
    
    print(f"\n🏗️ Model: {args.model_type}")
    print(f"   Backbone: {args.backbone}")
    print(f"   Input Resolution: {config['input_resolution']}")
    print(f"   Uncertainty Estimation: {config['model']['uncertainty_estimation']}")
    
    print(f"\n🎯 Training:")
    print(f"   Mode: {args.training_mode}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Loss: {config['loss']['type']}")
    print(f"   Safety Weights: {config['loss']['safety_weights']}")
    
    print(f"\n💾 Output: {args.output_dir}")
    
    # Initialize enhanced training pipeline
    pipeline = EnhancedTrainingPipeline(
        config=config,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb
    )
    
    try:
        # Create datasets
        print("\n📊 Creating datasets...")
        train_loader, val_loader, test_loader = pipeline.create_datasets()
        
        if not train_loader:
            print("❌ No training data available!")
            return 1
        
        if not val_loader:
            print("⚠️ No validation data available - using test set")
            val_loader = test_loader
        
        print(f"✅ Training samples: {len(train_loader.dataset)}")
        if val_loader:
            print(f"✅ Validation samples: {len(val_loader.dataset)}")
        if test_loader:
            print(f"✅ Test samples: {len(test_loader.dataset)}")
        
        # Start training
        print("\n🚀 Starting enhanced training...")
        model = pipeline.train(train_loader, val_loader)
        
        # Final evaluation on test set
        if test_loader and test_loader != val_loader:
            print("\n🧪 Final evaluation on test set...")
            criterion = pipeline.create_loss_function()
            test_metrics = pipeline.validate_epoch(model, test_loader, criterion)
            
            print("📊 Test Results:")
            print(f"   Test Loss: {test_metrics['loss']:.4f}")
            print(f"   Test mIoU: {test_metrics.get('miou', 0.0):.4f}")
            print(f"   Safety Score: {test_metrics.get('safety_score', 0.0):.4f}")
            
            # Save test results
            test_results_path = Path(args.output_dir) / 'test_results.json'
            with open(test_results_path, 'w') as f:
                json.dump(test_metrics, f, indent=2)
        
        # Convert best model to ONNX
        print("\n🔄 Converting to ONNX...")
        try:
            best_model_path = Path(args.output_dir) / 'best_model.pth'
            if best_model_path.exists():
                # Load best model
                checkpoint = torch.load(best_model_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Export to ONNX
                model.eval()
                dummy_input = torch.randn(1, 3, *config['input_resolution'])
                onnx_path = Path(args.output_dir) / 'enhanced_uav_landing_model.onnx'
                
                torch.onnx.export(
                    model,
                    dummy_input,
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['image'],
                    output_names=['predictions', 'uncertainty'] if config['model']['uncertainty_estimation'] else ['predictions'],
                    dynamic_axes={
                        'image': {0: 'batch_size'},
                        'predictions': {0: 'batch_size'},
                        'uncertainty': {0: 'batch_size'} if config['model']['uncertainty_estimation'] else {}
                    }
                )
                
                print(f"✅ ONNX model saved: {onnx_path}")
        except Exception as e:
            print(f"⚠️ ONNX export failed: {e}")
        
        print("\n🎉 Enhanced training completed successfully!")
        print(f"📁 Results saved in: {args.output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 