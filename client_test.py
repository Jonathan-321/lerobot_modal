#!/usr/bin/env python3
"""Test client for SmolVLA Modal deployment"""

import modal
import numpy as np
import time
import cv2
import base64
from PIL import Image
from io import BytesIO

def test_smolvla_inference():
    """Test the Modal SmolVLA inference"""
    print("🚀 Starting SmolVLA Modal Client Tests")
    print("=" * 50)
    
    try:
        # Get the deployed function
        f = modal.Function.lookup("lerobot-smolvla-inference", "run_inference")
        
        # Test 1: Basic inference with numpy array
        print("🤖 Testing SmolVLA inference...")
        
        # Create test inputs
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_state = [0.0, 0.1, -0.1, 0.0, 0.2, -0.2, 0.5]  # 7 joint positions
        
        print(f"   Image shape: {test_image.shape}")
        print(f"   State shape: {len(test_state)}")
        
        # Call remote inference
        print("📡 Calling remote inference...")
        start_time = time.time()
        
        result = f.remote(test_image.tolist(), test_state)  # Convert to list for serialization
        
        inference_time = time.time() - start_time
        
        if result['success']:
            print(f"✅ Inference successful in {inference_time:.3f}s")
            print(f"   Action: {result['action']}")
            print(f"   Device: {result.get('device', 'unknown')}")
        else:
            print(f"❌ Inference failed: {result.get('error', 'Unknown error')}")
            return False
        
        # Test 2: Base64 image input
        print("\n📸 Testing with base64 image...")
        
        # Convert image to base64
        pil_image = Image.fromarray(test_image)
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        result2 = f.remote(img_base64, test_state)
        
        if result2['success']:
            print("✅ Base64 inference successful")
            print(f"   Action: {result2['action']}")
        else:
            print(f"❌ Base64 inference failed: {result2.get('error', 'Unknown error')}")
        
        # Test 3: Performance test
        print("\n⚡ Running performance test...")
        num_calls = 5
        times = []
        
        for i in range(num_calls):
            start = time.time()
            result = f.remote(test_image.tolist(), test_state)
            times.append(time.time() - start)
            print(f"   Call {i+1}: {times[-1]:.3f}s")
        
        avg_time = np.mean(times)
        print(f"\n📊 Average inference time: {avg_time:.3f}s")
        print(f"   Min: {np.min(times):.3f}s, Max: {np.max(times):.3f}s")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_continuous_control():
    """Test continuous control simulation"""
    print("\n🎮 Testing continuous control loop...")
    
    try:
        f = modal.Function.lookup("lerobot-smolvla-inference", "run_inference")
        
        # Simulate camera capture
        print("   Simulating 10 control steps...")
        
        for step in range(10):
            # Generate varying test data
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            # Simulate changing robot state
            test_state = [
                0.1 * np.sin(step * 0.1),
                0.1 * np.cos(step * 0.1),
                0.0,
                0.0,
                0.1 * np.sin(step * 0.2),
                0.0,
                0.5 + 0.1 * np.sin(step * 0.3)  # Gripper
            ]
            
            result = f.remote(test_image.tolist(), test_state)
            
            if result['success']:
                action = result['action']
                print(f"   Step {step+1}: Action=[{action[0]:.3f}, {action[1]:.3f}, ..., {action[6]:.3f}]")
            else:
                print(f"   Step {step+1}: Failed - {result.get('error')}")
                
            time.sleep(0.1)  # Simulate control frequency
            
        print("✅ Continuous control test complete")
        
    except Exception as e:
        print(f"❌ Continuous control test failed: {e}")

if __name__ == "__main__":
    # Run tests
    success = test_smolvla_inference()
    
    if success:
        test_continuous_control()
    else:
        print("\n❌ Tests failed")
        print("\n📝 Troubleshooting:")
        print("   1. Make sure you've deployed modal_runner.py first")
        print("   2. Check that the Modal app name matches")
        print("   3. Verify your Modal authentication")