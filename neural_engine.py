# neural_engine.py
"""
Neural Engine for UAV Landing Zone Detection
Handles the BiSeNetV2 model inference using ONNX Runtime.
"""

import numpy as np
import cv2
import onnxruntime as ort
import config
from typing import Optional
import time


class NeuralEngine:
    """
    Wrapper class for the BiSeNetV2 segmentation model.
    Handles model loading, preprocessing, inference, and postprocessing.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the neural engine.
        
        Args:
            model_path: Path to the ONNX model file. If None, uses config.MODEL_PATH
        """
        self.model_path = model_path or config.MODEL_PATH
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        
        # Performance tracking
        self.inference_times = []
        
        self._load_model()
    
    def _load_model(self):
        """Load the ONNX model and initialize the inference session."""
        try:
            # Configure ONNX Runtime providers (prefer CUDA if available)
            providers = ['CPUExecutionProvider']  # Fallback to CPU
            
            # Check for CUDA availability
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print("Using CUDA for neural network inference")
            else:
                print("CUDA not available, using CPU for neural network inference")
            
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Get input and output information
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            
            print(f"Model loaded successfully from {self.model_path}")
            print(f"Input shape: {self.input_shape}")
            print(f"Input name: {self.input_name}")
            print(f"Output name: {self.output_name}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to placeholder mode for development")
            self.session = None
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess the input frame for the neural network.
        
        Args:
            frame: Input BGR frame from camera
            
        Returns:
            Preprocessed tensor ready for inference
        """
        # Resize to model input resolution
        resized = cv2.resize(frame, config.INPUT_RESOLUTION)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] range
        normalized = rgb_frame.astype(np.float32) / 255.0
        
        # Standard ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        # Convert HWC to CHW format and add batch dimension
        tensor = normalized.transpose(2, 0, 1)  # HWC -> CHW
        tensor = np.expand_dims(tensor, axis=0)  # Add batch dimension
        
        return tensor
    
    def _postprocess_output(self, raw_output: np.ndarray) -> np.ndarray:
        """
        Postprocess the neural network output.
        
        Args:
            raw_output: Raw network output (logits or probabilities)
            
        Returns:
            Segmentation map with class IDs
        """
        # Handle different output formats
        if raw_output.ndim == 4:  # Batch dimension present
            raw_output = raw_output[0]  # Remove batch dimension
        
        if raw_output.ndim == 3:  # Multi-class logits/probabilities
            # Convert to class IDs by taking argmax
            seg_map = np.argmax(raw_output, axis=0)
        else:
            # Already a class map
            seg_map = raw_output
        
        # Ensure output is the right data type and size
        seg_map = seg_map.astype(np.uint8)
        
        return seg_map
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame and return segmentation map.
        
        Args:
            frame: Input BGR frame from camera
            
        Returns:
            Segmentation map with class IDs (H, W) array
        """
        start_time = time.time()
        
        if self.session is None:
            # Placeholder for development/testing
            return self._generate_placeholder_segmentation()
        
        try:
            # Preprocess the frame
            input_tensor = self._preprocess_frame(frame)
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            raw_output = outputs[0]
            
            # Postprocess the output
            seg_map = self._postprocess_output(raw_output)
            
            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Keep only recent measurements
            if len(self.inference_times) > config.PERFORMANCE_WINDOW:
                self.inference_times = self.inference_times[-config.PERFORMANCE_WINDOW:]
            
            return seg_map
            
        except Exception as e:
            print(f"Error during neural network inference: {e}")
            return self._generate_placeholder_segmentation()
    
    def _generate_placeholder_segmentation(self) -> np.ndarray:
        """
        Generate a placeholder segmentation for testing purposes.
        Creates a synthetic scene with safe zones and obstacles.
        """
        # Create base segmentation map
        seg_map = np.zeros(config.INPUT_RESOLUTION, dtype=np.uint8)
        
        # Add some safe flat surfaces (class 1)
        cv2.rectangle(seg_map, (100, 100), (300, 300), config.SAFE_LANDING_CLASS_ID, -1)
        cv2.rectangle(seg_map, (350, 200), (450, 350), config.SAFE_LANDING_CLASS_ID, -1)
        
        # Add some obstacles (class 4)
        cv2.circle(seg_map, (200, 400), 50, config.HIGH_OBSTACLE_CLASS_ID, -1)
        cv2.rectangle(seg_map, (400, 400), (500, 500), config.HIGH_OBSTACLE_CLASS_ID, -1)
        
        # Add some uneven surfaces (class 2)
        cv2.rectangle(seg_map, (50, 400), (150, 500), config.CLASS_NAME_TO_ID["unsafe_uneven_surface"], -1)
        
        # Add some low obstacles (class 3)
        cv2.rectangle(seg_map, (300, 50), (320, 80), config.LOW_OBSTACLE_CLASS_ID, -1)
        
        return seg_map
    
    def get_performance_stats(self) -> dict:
        """
        Get performance statistics for the neural engine.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.inference_times:
            return {"avg_inference_time": 0, "fps": 0, "samples": 0}
        
        avg_time = np.mean(self.inference_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            "avg_inference_time": avg_time,
            "fps": fps,
            "samples": len(self.inference_times),
            "min_time": np.min(self.inference_times),
            "max_time": np.max(self.inference_times)
        }
    
    def reset_performance_stats(self):
        """Reset performance tracking statistics."""
        self.inference_times = []
