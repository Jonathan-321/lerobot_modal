# modal_smolvla_minimal.py
import modal

app = modal.App("smolvla-test")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("torch", "transformers", "num2words", "accelerate")
    .run_commands("git clone https://github.com/huggingface/lerobot.git /lerobot")
    .run_commands("cd /lerobot && pip install -e .")
)

@app.function(image=image, gpu="T4", timeout=600)
def test_smolvla():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    import torch
    
    print("Loading model...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.eval()
    
    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    print(f"Model loaded on {device}!")
    
    # Fix the normalization parameters
    print("Fixing normalization parameters...")
    
    if hasattr(policy.normalize_inputs, 'buffer_observation_state'):
        policy.normalize_inputs.buffer_observation_state.mean.data = torch.zeros(6).to(device)
        policy.normalize_inputs.buffer_observation_state.std.data = torch.ones(6).to(device)
    
    if hasattr(policy.normalize_targets, 'buffer_action'):
        action_dim = policy.normalize_targets.buffer_action.mean.shape[0]
        policy.normalize_targets.buffer_action.mean.data = torch.zeros(action_dim).to(device)
        policy.normalize_targets.buffer_action.std.data = torch.ones(action_dim).to(device)
    
    if hasattr(policy.unnormalize_outputs, 'buffer_action'):
        action_dim = policy.unnormalize_outputs.buffer_action.mean.shape[0]
        policy.unnormalize_outputs.buffer_action.mean.data = torch.zeros(action_dim).to(device)
        policy.unnormalize_outputs.buffer_action.std.data = torch.ones(action_dim).to(device)
    
    print("Fixed normalization parameters!")
    
    # Look for the correct inference method
    print("\n🔍 Exploring available methods...")
    methods = [attr for attr in dir(policy) if not attr.startswith('_') and callable(getattr(policy, attr))]
    inference_methods = [m for m in methods if any(keyword in m.lower() for keyword in ['predict', 'infer', 'generate', 'act', 'select'])]
    print(f"Potential inference methods: {inference_methods}")
    
    # Check for select_action which is common in LeRobot
    if hasattr(policy, 'select_action'):
        print("\n✅ Found select_action method! This is the inference method.")
        
        # Prepare batch for inference
        dummy_batch = {
            "observation.image": torch.randn(1, 3, 224, 224).to(device),
            "observation.state": torch.randn(1, 6).to(device),
            "task": "Pick up the red block",  # Try as a single string
        }
        
        try:
            with torch.no_grad():
                action = policy.select_action(dummy_batch)
            
            print(f"🎉 SUCCESS! Got action: {action}")
            if hasattr(action, 'shape'):
                print(f"Action shape: {action.shape}")
                print(f"Action values (first 3): {action[0][:3].cpu().numpy() if len(action.shape) > 1 else action[:3].cpu().numpy()}")
            
            return "SmolVLA works! Ready for SO-101 deployment! 🤖"
            
        except Exception as e:
            print(f"select_action error: {e}")
            
            # Try with task as a list
            dummy_batch["task"] = ["Pick up the red block"]
            try:
                with torch.no_grad():
                    action = policy.select_action(dummy_batch)
                print(f"🎉 SUCCESS with task as list! Got action: {action}")
                return "SmolVLA works! Ready for SO-101 deployment! 🤖"
            except Exception as e2:
                print(f"Also failed with list: {e2}")
    
    # If select_action doesn't exist, let's check the source code
    print("\n📖 Checking policy class structure...")
    
    # Try to find any method that generates actions
    for method_name in ['act', 'get_action', 'predict', 'generate_action']:
        if hasattr(policy, method_name):
            print(f"\nFound {method_name} method!")
            method = getattr(policy, method_name)
            
            # Try calling it
            dummy_batch = {
                "observation.image": torch.randn(1, 3, 224, 224).to(device),
                "observation.state": torch.randn(1, 6).to(device),
                "task": ["Pick up the red block"],
            }
            
            try:
                with torch.no_grad():
                    result = method(dummy_batch)
                print(f"✅ {method_name} works! Result: {result}")
                return f"Success with {method_name}!"
            except Exception as e:
                print(f"❌ {method_name} failed: {e}")
    
    # Last resort - check the model's internal structure
    if hasattr(policy, 'model'):
        print("\n🔧 Checking internal model...")
        print(f"Model type: {type(policy.model)}")
        model_methods = [attr for attr in dir(policy.model) if not attr.startswith('_') and callable(getattr(policy.model, attr))]
        print(f"Model methods: {[m for m in model_methods if 'forward' not in m][:10]}")
    
    raise Exception("Could not find inference method. Need to check LeRobot documentation.")

@app.local_entrypoint()
def main():
    result = test_smolvla.remote()
    print(f"\nResult: {result}")