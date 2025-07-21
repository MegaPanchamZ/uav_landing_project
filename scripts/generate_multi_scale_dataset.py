#!/usr/bin/env python3
"""
Multi-Scale Dataset Generation Script
====================================

Generate a massively expanded training dataset from the Semantic Drone Dataset
by leveraging its high resolution (6000×4000) to create multiple training samples
at different scales and zoom levels.

Expected Output:
- 400 base images → 10,000+ training samples (25x multiplier)
- Multiple altitude simulations (5m to 50m flight height)
- Quality-filtered patches with landing-relevant content
- Professional train/val/test splits with metadata
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from datasets.multi_scale_dataset_generator import (
    MultiScaleDatasetGenerator, 
    MultiScaleDataset,
    create_multi_scale_transforms
)


def main():
    """Main function for multi-scale dataset generation."""
    
    parser = argparse.ArgumentParser(
        description="Generate Multi-Scale Dataset for UAV Landing Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic generation with default settings
  python generate_multi_scale_dataset.py \\
    --base-dataset ../datasets/semantic_drone_dataset \\
    --output ../datasets/multi_scale_semantic_drone

  # High-overlap generation for maximum samples
  python generate_multi_scale_dataset.py \\
    --base-dataset ../datasets/semantic_drone_dataset \\
    --output ../datasets/multi_scale_semantic_drone \\
    --overlap-ratio 0.5 \\
    --min-landing-pixels 500

  # Custom patch sizes for specific training needs
  python generate_multi_scale_dataset.py \\
    --base-dataset ../datasets/semantic_drone_dataset \\
    --output ../datasets/multi_scale_semantic_drone \\
    --patch-sizes 384 512 768 1024
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--base-dataset',
        type=str,
        required=True,
        help='Path to Semantic Drone Dataset directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for generated multi-scale dataset'
    )
    
    # Dataset generation parameters
    parser.add_argument(
        '--patch-sizes',
        type=int,
        nargs='+',
        default=[512, 768, 1024],
        help='Patch sizes to generate (square patches)'
    )
    parser.add_argument(
        '--overlap-ratio',
        type=float,
        default=0.25,
        help='Overlap ratio between adjacent patches (0.0-0.8)'
    )
    parser.add_argument(
        '--min-landing-pixels',
        type=int,
        default=1000,
        help='Minimum landing-relevant pixels per patch'
    )
    parser.add_argument(
        '--quality-threshold',
        type=float,
        default=0.3,
        help='Quality threshold for patch selection'
    )
    
    # Split configuration
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.7,
        help='Fraction of data for training'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.15,
        help='Fraction of data for validation'
    )
    
    # Altitude simulation
    parser.add_argument(
        '--altitude-ranges',
        type=str,
        help='JSON string defining altitude ranges for each patch size'
    )
    
    # Processing options
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show estimated output without generating dataset'
    )
    parser.add_argument(
        '--test-loading',
        action='store_true',
        help='Test loading generated dataset after creation'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    base_path = Path(args.base_dataset)
    if not base_path.exists():
        print(f"❌ Base dataset not found: {base_path}")
        return 1
    
    if not (base_path / "original_images").exists():
        print(f"❌ Original images directory not found: {base_path / 'original_images'}")
        return 1
    
    if not (base_path / "label_images_semantic").exists():
        print(f"❌ Label images directory not found: {base_path / 'label_images_semantic'}")
        return 1
    
    # Convert patch sizes to tuples
    patch_sizes = [(size, size) for size in args.patch_sizes]
    
    # Parse altitude ranges if provided
    altitude_ranges = None
    if args.altitude_ranges:
        import json
        try:
            altitude_data = json.loads(args.altitude_ranges)
            altitude_ranges = {eval(k): v for k, v in altitude_data.items()}
        except Exception as e:
            print(f"❌ Invalid altitude ranges JSON: {e}")
            return 1
    
    # Display configuration
    print("🚀 Multi-Scale Dataset Generation Configuration")
    print("=" * 60)
    print(f"📊 Input:")
    print(f"   Base dataset: {base_path}")
    print(f"   Expected base images: 400")
    
    print(f"\n🔍 Generation Parameters:")
    print(f"   Patch sizes: {args.patch_sizes}")
    print(f"   Overlap ratio: {args.overlap_ratio}")
    print(f"   Min landing pixels: {args.min_landing_pixels}")
    print(f"   Quality threshold: {args.quality_threshold}")
    
    print(f"\n📈 Split Configuration:")
    print(f"   Training: {args.train_split*100:.1f}%")
    print(f"   Validation: {args.val_split*100:.1f}%")
    print(f"   Test: {(1-args.train_split-args.val_split)*100:.1f}%")
    
    print(f"\n💾 Output:")
    print(f"   Output directory: {args.output}")
    
    # Initialize generator
    generator = MultiScaleDatasetGenerator(
        base_dataset_path=str(base_path),
        output_path=args.output,
        patch_sizes=patch_sizes,
        altitude_ranges=altitude_ranges,
        overlap_ratio=args.overlap_ratio,
        min_landing_pixels=args.min_landing_pixels,
        quality_threshold=args.quality_threshold
    )
    
    # Show estimated output
    estimated_samples = generator.estimate_sample_count()
    print(f"\n🎯 Estimated Output:")
    print(f"   Total samples: {estimated_samples:,}")
    print(f"   Dataset multiplier: {estimated_samples/400:.1f}x")
    print(f"   Training samples: {int(estimated_samples * args.train_split):,}")
    print(f"   Validation samples: {int(estimated_samples * args.val_split):,}")
    print(f"   Test samples: {int(estimated_samples * (1-args.train_split-args.val_split)):,}")
    
    # Calculate storage requirements
    avg_patch_size_mb = 0.5  # Approximate MB per patch (image + label)
    total_storage_gb = (estimated_samples * avg_patch_size_mb) / 1024
    print(f"   Estimated storage: {total_storage_gb:.1f} GB")
    
    if args.dry_run:
        print("\n🔍 Dry run completed - no dataset generated.")
        return 0
    
    # Confirm generation
    print(f"\n⚠️  This will generate {estimated_samples:,} samples (~{total_storage_gb:.1f} GB).")
    response = input("Continue? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Generation cancelled.")
        return 0
    
    try:
        # Generate dataset
        print("\n🚀 Starting dataset generation...")
        generator.generate_dataset(
            train_split=args.train_split,
            val_split=args.val_split
        )
        
        print("\n✅ Dataset generation completed successfully!")
        
        # Test loading if requested
        if args.test_loading:
            print("\n🧪 Testing dataset loading...")
            
            try:
                # Create transforms
                train_transform, val_transform = create_multi_scale_transforms()
                
                # Test loading each split
                for split in ['train', 'val', 'test']:
                    dataset = MultiScaleDataset(
                        args.output,
                        split=split,
                        transform=val_transform
                    )
                    
                    if len(dataset) > 0:
                        # Test loading first sample
                        sample = dataset[0]
                        print(f"   ✅ {split}: {len(dataset):,} samples, shape: {sample['image'].shape}")
                    else:
                        print(f"   ⚠️ {split}: No samples found")
                
                print("✅ Dataset loading test passed!")
                
            except Exception as e:
                print(f"❌ Dataset loading test failed: {e}")
                return 1
        
        # Print final summary
        print(f"\n🎉 Multi-Scale Dataset Generation Complete!")
        print(f"📁 Dataset saved to: {args.output}")
        print(f"📊 Use with enhanced training pipeline:")
        print(f"   python train_enhanced_model.py --semantic-drone-path {args.output}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Generation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 