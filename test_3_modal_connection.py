#!/usr/bin/env python3
"""
Test 3: Modal Connection Test
Tests connection to deployed SmolVLA model
"""

import modal
import numpy as np
import time
import json

def test_modal_apps():
    """Test which Modal apps are available."""
    print("🔍 Checking available Modal apps...")
    
    # Common app names to try
    app_names = [
        "smolvla-so101-deployment", 
        "lerobot-smolvla",
        "smolvla-deployment",
        "lerobot"
    ]
    
    available_apps = []
    
    for app_name in app_names:
        try:
            app = modal.App.lookup(app_name)
            print(f"✅ Found app: {app_name}")
            available_apps.append(app_name)
        except Exception as e:
            print(f"❌ App '{app_name}' not found: {str(e)[:100]}...")
    
    return available_apps

def test_modal_function(app_name):
    """Test Modal function calls."""
    print(f"\n🧪 Testing functions in app: {app_name}")
    
    # Common function names to try
    function_names = [
        "get_action_direct",
        "run_inference", 
        "inference",
        "get_action",
        "predict"
    ]
    
    for func_name in function_names:
        try:
            print(f"  🔍 Looking for function: {func_name}")
            func = modal.Function.lookup(app_name, func_name)
            print(f"  ✅ Found function: {func_name}")
            
            # Test the function
            success = test_inference_function(func, func_name)
            if success:
                return func, func_name
                
        except Exception as e:
            print(f"  ❌ Function '{func_name}' not found: {str(e)[:100]}...")
    
    return None, None

def test_inference_function(func, func_name):
    """Test inference function with dummy data."""
    print(f"    🧪 Testing {func_name} with dummy data...")
    
    try:
        # Create dummy data
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        dummy_state = [0.0] * 7  # 7-DOF robot state
        instruction = "Pick up the red object"
        
        print(f"    📊 Input shapes:")
        print(f"      Image: {dummy_image.shape}")
        print(f"      State: {len(dummy_state)} values")
        print(f"      Instruction: '{instruction}'")
        
        # Time the inference
        start_time = time.time()
        
        # Try different function signatures
        try:
            # Most common signature
            result = func.remote(
                dummy_image.tolist(),
                dummy_state,
                instruction
            )
        except Exception as e1:
            try:
                # Alternative signature (just image + instruction)
                result = func.remote(
                    dummy_image.tolist(),
                    instruction
                )
            except Exception as e2:
                try:
                    # Alternative signature (different format)
                    result = func.remote({
                        'image': dummy_image.tolist(),
                        'robot_state': dummy_state,
                        'instruction': instruction
                    })
                except Exception as e3:
                    print(f"    ❌ All signatures failed:")
                    print(f"      Signature 1: {str(e1)[:100]}...")
                    print(f"      Signature 2: {str(e2)[:100]}...")
                    print(f"      Signature 3: {str(e3)[:100]}...")
                    return False
        
        inference_time = time.time() - start_time
        
        print(f"    ✅ Inference successful!")
        print(f"    ⏱️  Time: {inference_time:.2f}s")
        print(f"    📋 Result type: {type(result)}")
        
        # Parse result
        if isinstance(result, dict):
            print(f"    📊 Result keys: {list(result.keys())}")
            if 'action' in result:
                action = result['action']
                if isinstance(action, list) and len(action) >= 3:
                    print(f"    🎯 Action (first 3): [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}]")
                else:
                    print(f"    🎯 Action: {action}")
            if 'success' in result:
                print(f"    ✅ Success: {result['success']}")
        else:
            print(f"    📋 Result: {str(result)[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Inference failed: {e}")
        return False

def run_performance_test(func, func_name, num_tests=5):
    """Run performance test with multiple inferences."""
    print(f"\n🏃‍♂️ Running performance test ({num_tests} inferences)...")
    
    times = []
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_state = [0.0] * 7
    instruction = "Pick up the object"
    
    for i in range(num_tests):
        print(f"  Test {i+1}/{num_tests}...", end=" ")
        
        start_time = time.time()
        try:
            result = func.remote(
                dummy_image.tolist(),
                dummy_state,
                instruction
            )
            inference_time = time.time() - start_time
            times.append(inference_time)
            print(f"✅ {inference_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            continue
    
    if times:
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"\n📊 Performance Summary:")
        print(f"  Average: {avg_time:.2f}s")
        print(f"  Min: {min_time:.2f}s") 
        print(f"  Max: {max_time:.2f}s")
        print(f"  Theoretical FPS: {1/avg_time:.1f}")
        
        return avg_time
    
    return None

def main():
    print("🔌 Modal Connection Test")
    print("=" * 40)
    
    try:
        # Test available apps
        apps = test_modal_apps()
        
        if not apps:
            print("\n❌ No Modal apps found!")
            print("\nTroubleshooting:")
            print("1. Run: modal app list")
            print("2. Deploy your app: modal deploy modal_runner.py")
            return
        
        print(f"\n✅ Found {len(apps)} app(s): {apps}")
        
        # Test each app
        working_func = None
        working_app = None
        
        for app_name in apps:
            func, func_name = test_modal_function(app_name)
            if func:
                working_func = func
                working_app = app_name
                print(f"\n🎉 Successfully connected to {app_name}.{func_name}")
                break
        
        if working_func:
            # Run performance test
            avg_time = run_performance_test(working_func, func_name)
            
            if avg_time:
                print(f"\n🎯 Ready for real-time robot control!")
                print(f"Expected inference rate: {1/avg_time:.1f} FPS")
                print(f"Recommended camera FPS: {min(30, int(1/avg_time))}")
        else:
            print("\n❌ No working functions found in any app")
            
    except Exception as e:
        print(f"\n❌ Error connecting to Modal: {e}")
        print("\nTroubleshooting:")
        print("1. Check Modal login: modal token set --token-id YOUR_TOKEN")
        print("2. List apps: modal app list")
        print("3. Check app logs: modal app logs APP_NAME")

if __name__ == "__main__":
    main()
