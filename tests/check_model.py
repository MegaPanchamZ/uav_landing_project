#!/usr/bin/env python3
"""Check ONNX model input/output specifications"""

import onnxruntime as ort
import numpy as np

def check_model_specs():
    model_path = "trained_models/ultra_fast_uav_landing.onnx"
    
    print("🔍 Analyzing ONNX Model Specifications")
    print("=" * 50)
    
    try:
        session = ort.InferenceSession(model_path)
        
        print(f"📁 Model: {model_path}")
        print(f"📊 Providers: {session.get_providers()}")
        
        # Input specs
        print(f"\n📥 Input Specifications:")
        for input in session.get_inputs():
            print(f"   Name: {input.name}")
            print(f"   Shape: {input.shape}")
            print(f"   Type: {input.type}")
        
        # Output specs
        print(f"\n📤 Output Specifications:")
        for output in session.get_outputs():
            print(f"   Name: {output.name}")
            print(f"   Shape: {output.shape}")
            print(f"   Type: {output.type}")
        
        # Test inference with correct dimensions
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        print(f"\n🧪 Testing inference with shape {input_shape}...")
        
        # Handle dynamic dimensions
        test_shape = []
        for dim in input_shape:
            if isinstance(dim, str) or dim == -1:
                test_shape.append(1)  # Batch size
            else:
                test_shape.append(dim)
        
        test_input = np.random.rand(*test_shape).astype(np.float32)
        print(f"   Created test input: {test_input.shape}")
        
        output = session.run(None, {input_name: test_input})
        print(f"    Inference successful!")
        print(f"   Output shape: {output[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    check_model_specs()
