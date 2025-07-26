#!/usr/bin/env python3
"""
UAV Landing System - Inference Script
====================================

Standalone inference script for quick model testing and real-time prediction.
Supports single images, batch processing, video streams, and webcam input.

Features:
- Real-time inference with performance monitoring
- Batch processing with progress tracking
- Video/webcam processing with frame-by-frame analysis
- Confidence thresholding and uncertainty visualization
- Export results in multiple formats (images, video, JSON)

Usage:
    # Single image
    python inference.py --image test.jpg --model outputs/stage3_best.pth

    # Batch processing
    python inference.py --input_dir images/ --output_dir results/

    # Video processing
    python inference.py --video input.mp4 --output_video output.mp4

    # Webcam (real-time)
    python inference.py --webcam --display
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# Import our components
from models.mobilenetv3_edge_model import create_edge_model
from datasets.semantic_drone_dataset import create_semantic_drone_transforms


class UAVLandingInference:
    """High-performance inference engine for UAV landing detection."""
    
    def __init__(
        self, 
        model_path: str,
        device: str = 'auto',
        confidence_threshold: float = 0.5,
        input_size: Tuple[int, int] = (512, 512)
    ):
        """Initialize inference engine."""
        
        # Auto-detect device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        
        # Load model
        print(f"🧠 Loading model from: {model_path}")
        self.model = self._load_model(model_path)
        
        # Setup preprocessing
        self.transform = create_semantic_drone_transforms(
            input_size=input_size,
            is_training=False
        )
        
        # Class definitions
        self.class_names = ['ground', 'vegetation', 'obstacle', 'water', 'vehicle', 'other']
        self.class_colors = np.array([
            [128, 64, 128],    # ground - purple
            [107, 142, 35],    # vegetation - olive
            [70, 70, 70],      # obstacle - dark gray
            [0, 0, 142],       # water - dark blue
            [0, 0, 70],        # vehicle - dark red
            [102, 102, 156]    # other - light purple
        ])
        
        # Safety mapping
        self.safety_levels = {
            0: 'safe',      # ground
            1: 'caution',   # vegetation
            2: 'danger',    # obstacle
            3: 'danger',    # water
            4: 'danger',    # vehicle
            5: 'unknown'    # other
        }
        
        # Performance tracking
        self.inference_times = []
        self.processed_frames = 0
        
        print(f"✅ Inference engine ready")
        print(f"   Device: {self.device}")
        print(f"   Input size: {self.input_size}")
        print(f"   Confidence threshold: {self.confidence_threshold}")
    
    def _load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        model = create_edge_model(
            model_type='enhanced',
            num_classes=6,
            use_uncertainty=True,
            pretrained=False
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Enable inference optimizations
        if hasattr(torch, 'compile'):
            model = torch.compile(model)
        
        print(f"   Model loaded successfully")
        print(f"   Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"   Checkpoint mIoU: {checkpoint.get('metrics', {}).get('miou', 'N/A')}")
        
        return model
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess input image for inference."""
        
        # Ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR and convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if hasattr(self.transform, 'replay'):
            # Handle albumentations transform
            transformed = self.transform(image=image)
            tensor = transformed['image']
        else:
            # Handle torchvision transform
            tensor = self.transform(image)
        
        # Add batch dimension and move to device
        return tensor.unsqueeze(0).to(self.device)
    
    def predict_single(self, image: np.ndarray, return_confidence: bool = True) -> Dict:
        """Predict on a single image."""
        
        start_time = time.time()
        
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            if isinstance(outputs, dict):
                predictions = outputs['main']
                uncertainty = outputs.get('uncertainty', None)
            else:
                predictions = outputs
                uncertainty = None
        
        # Post-process
        pred_probs = F.softmax(predictions, dim=1)
        pred_classes = predictions.argmax(dim=1)
        
        # Convert to numpy
        pred_classes_np = pred_classes[0].cpu().numpy()
        pred_probs_np = pred_probs[0].cpu().numpy()
        
        # Confidence map
        confidence_map = None
        if uncertainty is not None and return_confidence:
            confidence_map = (1.0 - torch.sigmoid(uncertainty[0])).cpu().numpy()
        
        # Performance tracking
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.processed_frames += 1
        
        # Compute statistics
        stats = self._compute_stats(pred_classes_np, pred_probs_np, confidence_map)
        
        return {
            'prediction': pred_classes_np,
            'probabilities': pred_probs_np,
            'confidence_map': confidence_map,
            'stats': stats,
            'inference_time': inference_time,
            'timestamp': time.time()
        }
    
    def _compute_stats(
        self, 
        prediction: np.ndarray, 
        probabilities: np.ndarray, 
        confidence_map: Optional[np.ndarray]
    ) -> Dict:
        """Compute prediction statistics."""
        
        total_pixels = prediction.size
        stats = {}
        
        # Per-class statistics
        for class_id, class_name in enumerate(self.class_names):
            class_mask = prediction == class_id
            class_pixels = class_mask.sum()
            percentage = (class_pixels / total_pixels) * 100
            
            # Average confidence for this class
            avg_confidence = None
            if confidence_map is not None:
                class_confidences = confidence_map[class_mask]
                avg_confidence = float(class_confidences.mean()) if len(class_confidences) > 0 else 0.0
            
            stats[class_name] = {
                'pixels': int(class_pixels),
                'percentage': float(percentage),
                'avg_confidence': avg_confidence,
                'safety_level': self.safety_levels[class_id]
            }
        
        # Overall safety assessment
        safe_percentage = sum(
            s['percentage'] for s in stats.values() 
            if s['safety_level'] == 'safe'
        )
        danger_percentage = sum(
            s['percentage'] for s in stats.values() 
            if s['safety_level'] == 'danger'
        )
        
        # Safety score (simple heuristic)
        safety_score = (safe_percentage - danger_percentage) / 100.0
        safety_score = max(0.0, min(1.0, safety_score))
        
        if safety_score > 0.6:
            overall_safety = 'safe'
        elif safety_score > 0.3:
            overall_safety = 'caution'
        else:
            overall_safety = 'danger'
        
        stats['overall'] = {
            'safety_score': float(safety_score),
            'safety_level': overall_safety,
            'safe_percentage': float(safe_percentage),
            'danger_percentage': float(danger_percentage)
        }
        
        return stats
    
    def create_visualization(
        self, 
        original_image: np.ndarray, 
        prediction: np.ndarray,
        confidence_map: Optional[np.ndarray] = None,
        stats: Optional[Dict] = None,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """Create visualization overlay."""
        
        # Resize prediction to match original image
        if original_image.shape[:2] != prediction.shape:
            prediction_resized = cv2.resize(
                prediction.astype(np.uint8), 
                (original_image.shape[1], original_image.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
        else:
            prediction_resized = prediction
        
        # Create colored mask
        colored_mask = np.zeros((*prediction_resized.shape, 3), dtype=np.uint8)
        for class_id in range(6):
            mask = prediction_resized == class_id
            colored_mask[mask] = self.class_colors[class_id]
        
        # Blend with original image
        alpha = 0.6
        overlay = cv2.addWeighted(original_image, 1 - alpha, colored_mask, alpha, 0)
        
        # Add text overlay with stats
        if stats is not None:
            self._add_text_overlay(overlay, stats)
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, overlay)
        
        return overlay
    
    def _add_text_overlay(self, image: np.ndarray, stats: Dict):
        """Add text overlay with statistics."""
        
        # Overall safety info
        overall = stats['overall']
        safety_text = f"Safety: {overall['safety_level'].upper()} ({overall['safety_score']:.2f})"
        
        # Safety color
        if overall['safety_level'] == 'safe':
            color = (0, 255, 0)  # Green
        elif overall['safety_level'] == 'caution':
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Background rectangle
        text_size = cv2.getTextSize(safety_text, font, font_scale, thickness)[0]
        cv2.rectangle(image, (10, 10), (text_size[0] + 20, text_size[1] + 20), (0, 0, 0), -1)
        
        # Text
        cv2.putText(image, safety_text, (15, 30), font, font_scale, color, thickness)
        
        # Top classes
        y_offset = 60
        for class_name, class_stats in stats.items():
            if class_name == 'overall' or class_stats['percentage'] < 5:
                continue
            
            class_text = f"{class_name}: {class_stats['percentage']:.1f}%"
            cv2.putText(image, class_text, (15, y_offset), font, 0.5, (255, 255, 255), 1)
            y_offset += 25
    
    def process_image(self, input_path: str, output_path: Optional[str] = None) -> Dict:
        """Process a single image file."""
        
        print(f"📷 Processing image: {input_path}")
        
        # Load image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        # Predict
        result = self.predict_single(image)
        
        # Create visualization
        if output_path:
            visualization = self.create_visualization(
                image, result['prediction'], result['confidence_map'], result['stats']
            )
            cv2.imwrite(output_path, visualization)
            print(f"💾 Saved result: {output_path}")
        
        # Print summary
        stats = result['stats']
        overall = stats['overall']
        print(f"   Safety Level: {overall['safety_level'].upper()}")
        print(f"   Safety Score: {overall['safety_score']:.3f}")
        print(f"   Inference Time: {result['inference_time']:.3f}s")
        
        return result
    
    def process_batch(
        self, 
        input_dir: str, 
        output_dir: Optional[str] = None,
        file_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
    ) -> List[Dict]:
        """Process a batch of images."""
        
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Find all image files
        image_files = []
        for ext in file_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            raise ValueError(f"No image files found in: {input_dir}")
        
        print(f"📁 Processing {len(image_files)} images from: {input_dir}")
        
        # Create output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Process images
        results = []
        
        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                # Output path
                output_file = None
                if output_dir:
                    output_file = output_path / f"{image_file.stem}_result{image_file.suffix}"
                
                # Process
                result = self.process_image(str(image_file), str(output_file) if output_file else None)
                result['input_file'] = str(image_file)
                result['output_file'] = str(output_file) if output_file else None
                
                results.append(result)
                
            except Exception as e:
                print(f"❌ Failed to process {image_file}: {e}")
                continue
        
        # Summary
        avg_inference_time = np.mean([r['inference_time'] for r in results])
        print(f"\n✅ Batch processing complete!")
        print(f"   Processed: {len(results)}/{len(image_files)} images")
        print(f"   Average inference time: {avg_inference_time:.3f}s")
        print(f"   Total time: {sum(r['inference_time'] for r in results):.2f}s")
        
        return results
    
    def process_video(
        self, 
        input_path: str, 
        output_path: Optional[str] = None,
        frame_skip: int = 1,
        display: bool = False
    ) -> Dict:
        """Process video file frame by frame."""
        
        print(f"🎬 Processing video: {input_path}")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"   Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps // frame_skip, (width, height))
        
        # Process frames
        frame_count = 0
        processed_count = 0
        results = []
        
        pbar = tqdm(total=total_frames // frame_skip, desc="Processing video")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames
                if frame_count % frame_skip != 0:
                    continue
                
                # Process frame
                result = self.predict_single(frame, return_confidence=False)
                
                # Create visualization
                visualization = self.create_visualization(
                    frame, result['prediction'], stats=result['stats']
                )
                
                # Write to output video
                if writer:
                    writer.write(visualization)
                
                # Display
                if display:
                    cv2.imshow('UAV Landing Detection', visualization)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Store result
                result['frame_number'] = frame_count
                results.append(result)
                processed_count += 1
                
                pbar.update(1)
        
        finally:
            pbar.close()
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Summary
        avg_inference_time = np.mean([r['inference_time'] for r in results])
        processing_fps = processed_count / sum(r['inference_time'] for r in results)
        
        print(f"\n✅ Video processing complete!")
        print(f"   Processed: {processed_count} frames")
        print(f"   Average inference time: {avg_inference_time:.3f}s")
        print(f"   Processing FPS: {processing_fps:.1f}")
        if output_path:
            print(f"   Output saved: {output_path}")
        
        return {
            'total_frames': total_frames,
            'processed_frames': processed_count,
            'avg_inference_time': avg_inference_time,
            'processing_fps': processing_fps,
            'results': results
        }
    
    def process_webcam(self, display: bool = True, save_frames: bool = False) -> None:
        """Process webcam input in real-time."""
        
        print(f"📹 Starting webcam processing (press 'q' to quit)")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Could not open webcam")
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                result = self.predict_single(frame, return_confidence=False)
                
                # Create visualization
                visualization = self.create_visualization(
                    frame, result['prediction'], stats=result['stats']
                )
                
                # Add FPS info
                if len(self.inference_times) > 10:
                    recent_fps = 1.0 / np.mean(self.inference_times[-10:])
                    cv2.putText(visualization, f"FPS: {recent_fps:.1f}", 
                              (visualization.shape[1] - 120, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display
                if display:
                    cv2.imshow('UAV Landing Detection - Webcam', visualization)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s') and save_frames:
                        # Save current frame
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"webcam_frame_{timestamp}.jpg"
                        cv2.imwrite(filename, visualization)
                        print(f"💾 Saved frame: {filename}")
                
                frame_count += 1
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
        
        print(f"\n✅ Webcam processing ended")
        print(f"   Processed frames: {frame_count}")
        if len(self.inference_times) > 0:
            avg_fps = 1.0 / np.mean(self.inference_times)
            print(f"   Average FPS: {avg_fps:.1f}")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        
        if not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        
        return {
            'total_frames': self.processed_frames,
            'total_time': float(times.sum()),
            'avg_inference_time': float(times.mean()),
            'min_inference_time': float(times.min()),
            'max_inference_time': float(times.max()),
            'std_inference_time': float(times.std()),
            'avg_fps': float(1.0 / times.mean()),
            'min_fps': float(1.0 / times.max()),
            'max_fps': float(1.0 / times.min())
        }


def main():
    parser = argparse.ArgumentParser(description='UAV Landing System - Inference')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='outputs/stage3_best.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Confidence threshold for predictions')
    parser.add_argument('--input_size', type=int, nargs=2, default=[512, 512],
                        help='Input image size (height width)')
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Single image file')
    input_group.add_argument('--input_dir', type=str, help='Directory of images')
    input_group.add_argument('--video', type=str, help='Video file')
    input_group.add_argument('--webcam', action='store_true', help='Use webcam input')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output file/directory')
    parser.add_argument('--output_video', type=str, help='Output video file')
    parser.add_argument('--save_json', action='store_true', help='Save results as JSON')
    parser.add_argument('--display', action='store_true', help='Display results')
    
    # Processing options
    parser.add_argument('--frame_skip', type=int, default=1,
                        help='Process every Nth frame (for video)')
    parser.add_argument('--save_frames', action='store_true',
                        help='Save individual frames (webcam mode)')
    
    args = parser.parse_args()
    
    print("🚁 UAV Landing System - Inference Engine")
    print("=======================================")
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"❌ Model checkpoint not found: {args.model}")
        print("   Please train a model first using train.py")
        return
    
    try:
        # Initialize inference engine
        inference_engine = UAVLandingInference(
            model_path=args.model,
            device=args.device,
            confidence_threshold=args.confidence_threshold,
            input_size=tuple(args.input_size)
        )
        
        # Process based on input type
        results = None
        
        if args.image:
            # Single image
            output_path = args.output or f"{Path(args.image).stem}_result{Path(args.image).suffix}"
            results = inference_engine.process_image(args.image, output_path)
            
        elif args.input_dir:
            # Batch processing
            output_dir = args.output or f"{args.input_dir.rstrip('/')}_results"
            results = inference_engine.process_batch(args.input_dir, output_dir)
            
        elif args.video:
            # Video processing
            output_path = args.output_video or args.output
            if output_path is None:
                output_path = f"{Path(args.video).stem}_result{Path(args.video).suffix}"
            
            results = inference_engine.process_video(
                args.video, output_path, args.frame_skip, args.display
            )
            
        elif args.webcam:
            # Webcam processing
            inference_engine.process_webcam(args.display, args.save_frames)
        
        # Save JSON results
        if args.save_json and results:
            json_path = args.output or 'inference_results.json'
            if not json_path.endswith('.json'):
                json_path += '.json'
            
            # Make results JSON serializable
            if isinstance(results, list):
                for result in results:
                    if 'prediction' in result:
                        result['prediction'] = result['prediction'].tolist()
                    if 'probabilities' in result:
                        result['probabilities'] = result['probabilities'].tolist()
                    if 'confidence_map' in result and result['confidence_map'] is not None:
                        result['confidence_map'] = result['confidence_map'].tolist()
            elif isinstance(results, dict):
                if 'prediction' in results:
                    results['prediction'] = results['prediction'].tolist()
                if 'probabilities' in results:
                    results['probabilities'] = results['probabilities'].tolist()
                if 'confidence_map' in results and results['confidence_map'] is not None:
                    results['confidence_map'] = results['confidence_map'].tolist()
            
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved: {json_path}")
        
        # Performance summary
        perf_stats = inference_engine.get_performance_stats()
        if perf_stats:
            print(f"\n📊 Performance Summary:")
            print(f"   Processed frames: {perf_stats['total_frames']}")
            print(f"   Average FPS: {perf_stats['avg_fps']:.1f}")
            print(f"   Average inference time: {perf_stats['avg_inference_time']:.3f}s")
            print(f"   Total processing time: {perf_stats['total_time']:.2f}s")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 