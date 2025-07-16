import modal
from typing import Dict, List, Optional

# Build the image with all dependencies
LERO_IMAGE = (
    modal.Image.debian_slim()
    .apt_install("git", "build-essential", "cmake", "pkg-config")
    .run_commands("git clone https://github.com/huggingface/lerobot.git /lerobot")
    .pip_install(
        "torch",
        "torchvision", 
        "transformers",
        "num2words",
        "sentencepiece",
        "opencv-python-headless",
        "pillow",
        "numpy",
    )
    .run_commands("cd /lerobot && pip install -e .")
)

app = modal.App("lerobot-smolvla-inference", image=LERO_IMAGE)

# Store the model in a volume for persistence
model_volume = modal.Volume.from_name("smolvla-model-cache", create_if_missing=True)

@app.cls(
    gpu="T4",
    scaledown_window=300,  # Updated from container_idle_timeout
    volumes={"/model_cache": model_volume},
)
class SmolVLAInference:
    # Use modal.parameter() for initialization instead of __init__
    model_name: str = modal.parameter(default="lerobot/smolvla_base")
    
    @modal.enter()
    def setup(self):
        """Setup method called when container starts"""
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        
        print("🚀 Initializing SmolVLA...")
        
        # Load the pretrained model with proper configuration
        self.policy = SmolVLAPolicy.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir="/model_cache",  # Use persistent volume
        )
        
        # Ensure model is in eval mode
        self.policy.eval()
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = self.policy.to(self.device)
        
        print(f"✅ Model loaded on {self.device}")
        
        # Initialize normalization stats if needed
        self._initialize_normalization()
        
    def _initialize_normalization(self):
        """Initialize normalization statistics if they're not set"""
        import torch
        
        # Check if normalization layers exist and have proper stats
        if hasattr(self.policy, 'normalize_inputs'):
            # Common robot state dimensions for SO-101
            state_dim = 7
            
            # Initialize with reasonable defaults if needed
            if hasattr(self.policy.normalize_inputs, 'buffer_observation_state'):
                buffer = self.policy.normalize_inputs.buffer_observation_state
                if not hasattr(buffer, 'mean') or torch.isnan(buffer.mean).any() or torch.isinf(buffer.mean).any():
                    # Set reasonable defaults for joint positions (radians)
                    buffer.mean = torch.zeros(state_dim)
                    buffer.std = torch.ones(state_dim)
                    print("📊 Initialized state normalization with defaults")
        
        # Do the same for output normalization
        if hasattr(self.policy, 'normalize_targets'):
            if hasattr(self.policy.normalize_targets, 'buffer_action'):
                buffer = self.policy.normalize_targets.buffer_action
                if not hasattr(buffer, 'mean') or torch.isnan(buffer.mean).any() or torch.isinf(buffer.mean).any():
                    buffer.mean = torch.zeros(7)  # 7 DoF
                    buffer.std = torch.ones(7)
                    print("📊 Initialized action normalization with defaults")
    
    @modal.method()
    def run_inference(self, image, state):
        """
        Run inference on image and state
        
        Args:
            image: numpy array of shape (H, W, 3) or base64 string
            state: list or array of joint positions
        
        Returns:
            dict with action and metadata
        """
        import torch
        import cv2
        import numpy as np
        import base64
        from PIL import Image
        from io import BytesIO
        
        try:
            # Handle base64 input
            if isinstance(image, str):
                image_data = base64.b64decode(image)
                pil_image = Image.open(BytesIO(image_data))
                image = np.array(pil_image)
            
            # Ensure image is numpy array
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            
            # Resize image to expected size
            if image.shape[:2] != (224, 224):
                image = cv2.resize(image, (224, 224))
            
            # Convert to tensor and normalize
            image_tensor = torch.from_numpy(image).float() / 255.0
            
            # Ensure correct shape (C, H, W)
            if image_tensor.dim() == 3:
                if image_tensor.shape[2] == 3:  # H, W, C -> C, H, W
                    image_tensor = image_tensor.permute(2, 0, 1)
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Convert state to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Ensure state has correct dimension
            if state_tensor.shape[1] != 7:
                # Pad or truncate to 7 dimensions
                if state_tensor.shape[1] < 7:
                    state_tensor = torch.nn.functional.pad(state_tensor, (0, 7 - state_tensor.shape[1]))
                else:
                    state_tensor = state_tensor[:, :7]
            
            # Create batch in the format the model expects
            batch = {
                "observation.image": image_tensor,
                "observation.state": state_tensor,
            }
            
            # Run inference
            with torch.no_grad():
                output = self.policy(batch)
            
            # Extract action
            if isinstance(output, dict) and 'action' in output:
                action = output['action'].cpu().numpy().squeeze()
            else:
                action = output.cpu().numpy().squeeze()
            
            # Ensure action is the right size
            if len(action) != 7:
                action = np.resize(action, 7)
            
            return {
                "success": True,
                "action": action.tolist(),
                "device": str(self.device),
            }
            
        except Exception as e:
            print(f"❌ Inference error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "error": str(e),
                "action": [0.0] * 7,  # Safe default
            }

# Create the function that wraps the class
@app.function(gpu="T4", timeout=300)
def run_inference(image, state):
    """Function wrapper for the inference class"""
    inference = SmolVLAInference()
    return inference.run_inference.remote(image, state)

# Web endpoint for REST API access
@app.function(
    gpu="T4",
    scaledown_window=300,
)
@modal.web_endpoint()
def inference_endpoint(request: Dict):
    """REST API endpoint for inference"""
    image = request.get("image")
    state = request.get("state", [0.0] * 7)
    
    inference = SmolVLAInference()
    return inference.run_inference.remote(image, state)

# Local testing endpoint
@app.local_entrypoint()
def test_inference():
    """Test the inference locally"""
    print("🧪 Testing SmolVLA inference...")
    
    # Create dummy inputs - numpy must be imported inside Modal context
    import numpy as np
    
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_state = [0.0] * 7
    
    # Run inference
    result = run_inference.remote(dummy_image.tolist(), dummy_state)
    
    print(f"✅ Result: {result}")
    return result