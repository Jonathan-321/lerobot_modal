# modal_inference_only.py
import modal

app = modal.App("smolvla-inference-production")

# Build the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "libgomp1")
    .pip_install(
        "torch==2.1.0",
        "torchvision==0.16.0",
        "transformers==4.51.3",  # Specific version that works
        "accelerate",
        "opencv-python-headless",
        "numpy",
        "huggingface_hub",
        "safetensors",
        "einops",
        "av",
        "imageio",
        "pillow",
        "omegaconf",
        "hydra-core",
    )
    .run_commands([
        "git clone https://github.com/huggingface/lerobot.git /lerobot",
        "cd /lerobot && pip install -e .",
        "cd /lerobot && pip install -e '.[smolvla]'"
    ])
)

@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    volumes={"/cache": modal.Volume.from_name("model-cache", create_if_missing=True)},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def predict_action(
    images,  # Can be single image array or dict of camera_name -> image_array
    state,   # Robot state vector (typically 6D for SO-100 robots)
    task="Pick up the object",
    model_id="kaku/my_smolvla_pickplace"
):
    """
    Run SmolVLA inference for robotic action prediction.
    
    Args:
        images: Either a single numpy array (H,W,3) or dict of camera_name -> array
                Single images will be duplicated for all expected cameras
        state: Robot state vector (list or array). Will be padded/truncated to match model
        task: Natural language instruction for the robot
        model_id: HuggingFace model ID of fine-tuned SmolVLA
        
    Returns:
        List of action values (typically 6D for joint positions/velocities)
    """
    import torch
    import numpy as np
    import os
    import sys
    sys.path.insert(0, '/lerobot')
    
    from huggingface_hub import login
    
    if "HF_TOKEN" in os.environ:
        login(token=os.environ["HF_TOKEN"])
    
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    
    print(f"Loading SmolVLA model: {model_id}")
    
    # Load the model
    policy = SmolVLAPolicy.from_pretrained(
        model_id,
        cache_dir="/cache"
    )
    
    policy.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    
    # Get expected camera configuration
    expected_cameras = {}
    for key, feature in policy.config.input_features.items():
        if key.startswith("observation.images."):
            expected_cameras[key] = feature.shape
    
    print(f"Model expects {len(expected_cameras)} cameras: {list(expected_cameras.keys())}")
    
    # Prepare batch
    batch = {}
    
    # Handle images
    if isinstance(images, dict):
        # Multiple cameras provided
        for camera_key in expected_cameras:
            camera_name = camera_key.split('.')[-1]  # e.g., "top" from "observation.images.top"
            
            # Try different naming conventions
            image_array = None
            for possible_key in [camera_key, camera_name, f"observation.images.{camera_name}"]:
                if possible_key in images:
                    image_array = images[possible_key]
                    break
            
            if image_array is None:
                # Use first available image as fallback
                image_array = list(images.values())[0]
                print(f"Warning: {camera_key} not found, using fallback image")
            
            # Convert to tensor
            image_tensor = torch.tensor(image_array).float() / 255.0
            if image_tensor.dim() == 3 and image_tensor.shape[2] == 3:
                image_tensor = image_tensor.permute(2, 0, 1)
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            batch[camera_key] = image_tensor.to(device)
    else:
        # Single image provided - duplicate for all cameras
        image_array = images
        for camera_key in expected_cameras:
            image_tensor = torch.tensor(image_array).float() / 255.0
            if image_tensor.dim() == 3 and image_tensor.shape[2] == 3:
                image_tensor = image_tensor.permute(2, 0, 1)
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            batch[camera_key] = image_tensor.to(device)
    
    # Handle state
    state_dim = policy.config.input_features['observation.state'].shape[0]
    state_array = np.array(state)
    
    if len(state_array) != state_dim:
        print(f"Adjusting state from {len(state_array)} to {state_dim} dimensions")
        if len(state_array) > state_dim:
            state_array = state_array[:state_dim]
        else:
            state_array = np.pad(state_array, (0, state_dim - len(state_array)), 'constant')
    
    state_tensor = torch.tensor(state_array).float().unsqueeze(0).to(device)
    batch['observation.state'] = state_tensor
    
    # Add task
    batch['task'] = [task]
    
    # Run inference
    with torch.no_grad():
        action = policy.select_action(batch)
    
    # Convert to numpy
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()
    
    # Process output
    if action.ndim > 2:
        action = action.squeeze()
    if action.ndim > 1:
        action = action[0]
    
    action = np.atleast_1d(action.squeeze())
    
    print(f"Predicted action: {action}")
    
    return action.tolist()

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def list_available_models():
    """List all available SmolVLA models with their configurations"""
    import os
    from huggingface_hub import login, HfApi
    
    if "HF_TOKEN" in os.environ:
        login(token=os.environ["HF_TOKEN"])
    
    api = HfApi()
    
    print("Available SmolVLA models:\n")
    
    # Search for SmolVLA models
    models = api.list_models(search="smolvla", limit=100)
    
    valid_models = []
    for model in models:
        model_id = model.modelId
        try:
            # Check if it has the required files
            files = api.list_repo_files(model_id)
            if "config.json" in files and any("safetensors" in f for f in files):
                valid_models.append(model_id)
                print(f"✓ {model_id}")
        except:
            pass
    
    print(f"\nFound {len(valid_models)} valid SmolVLA models")
    return valid_models

@app.local_entrypoint()
def test():
    """Test SmolVLA inference with example data"""
    import numpy as np
    
    print("=== SmolVLA Inference Test ===\n")
    
    # Create dummy data
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_state = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]  # 6D state
    
    # Test 1: Single image
    print("Test 1: Single image input")
    action = predict_action.remote(
        dummy_image,
        dummy_state,
        "Pick up the red cube",
        "kaku/my_smolvla_pickplace"
    )
    print(f"Result: {len(action)}D action vector\n")
    
    # Test 2: Multiple cameras
    print("Test 2: Multiple camera input")
    images_dict = {
        "top": dummy_image,
        "wrist": dummy_image,
    }
    action = predict_action.remote(
        images_dict,
        dummy_state,
        "Move the object to the left",
        "kaku/my_smolvla_pickplace"
    )
    print(f"Result: {len(action)}D action vector\n")
    
    # Test 3: Different model
    print("Test 3: Different model")
    try:
        action = predict_action.remote(
            dummy_image,
            dummy_state,
            "Push the cube forward",
            "jccj/smolvla_pickup_cube_full_res"
        )
        print(f"Result: {len(action)}D action vector\n")
    except Exception as e:
        print(f"Failed: {str(e)[:100]}...\n")
    
    # List available models
    print("\nListing all available models...")
    models = list_available_models.remote()
    
    return action

# For deployment
@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    volumes={"/cache": modal.Volume.from_name("model-cache", create_if_missing=True)},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def health_check():
    """Simple health check endpoint"""
    import torch
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    }