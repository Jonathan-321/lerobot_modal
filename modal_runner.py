# modal_runner.py
import modal
from typing import Dict, List, Optional
import json

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
        "accelerate",
        "opencv-python-headless",
        "pillow",
        "numpy",
        "fastapi[standard]",
        "pydantic",
    )
    .run_commands("cd /lerobot && pip install -e .")
)

app = modal.App("lerobot-smolvla-inference", image=LERO_IMAGE)

# Store the model in a volume for persistence
model_volume = modal.Volume.from_name("smolvla-model-cache", create_if_missing=True)

@app.cls(
    gpu="T4",
    scaledown_window=300,
    volumes={"/model_cache": model_volume},
    timeout=600,
)
class SmolVLAInference:
    model_name: str = modal.parameter(default="lerobot/smolvla_base")
    
    @modal.enter()
    def setup(self):
        """Setup method called when container starts"""
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        
        print("🚀 Initializing SmolVLA...")
        
        # Load the pretrained model with proper configuration
        print(f"Attempting to load model: {self.model_name}")
        self.policy = SmolVLAPolicy.from_pretrained(
            self.model_name,
            cache_dir="/model_cache",  # Use persistent volume
        )
        print("✅ SmolVLA Policy Loaded!")
        
        # Move model to the correct device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        print(f"✅ Model moved to device: {self.device}")
        
        # Ensure model is in eval mode
        self.policy.eval()
        
        print(f"✅ Model set to eval mode on {self.device}")
        print(f"✅ Model loaded on {self.device}")
        
        # Initialize normalization stats if needed
        self._initialize_normalization()
        
    def _initialize_normalization(self):
        """Initialize normalization statistics if they're not set"""
        import torch
        
        # Check if normalization layers exist and have proper stats
        if hasattr(self.policy, 'normalize_inputs'):
            if hasattr(self.policy.normalize_inputs, 'buffer_observation_state'):
                buffer = self.policy.normalize_inputs.buffer_observation_state
                if hasattr(buffer, 'mean') and (torch.isnan(buffer.mean).any() or torch.isinf(buffer.mean).any()):
                    state_dim = buffer.mean.shape[0]
                    buffer.mean.data.copy_(torch.zeros(state_dim))
                    buffer.std.data.copy_(torch.ones(state_dim))
                    print(f"📊 Initialized state normalization with defaults for dim={state_dim}")

        if hasattr(self.policy, 'normalize_targets'):
            if hasattr(self.policy.normalize_targets, 'buffer_action'):
                buffer = self.policy.normalize_targets.buffer_action
                if hasattr(buffer, 'mean') and (torch.isnan(buffer.mean).any() or torch.isinf(buffer.mean).any()):
                    action_dim = buffer.mean.shape[0]
                    buffer.mean.data.copy_(torch.zeros(action_dim))
                    buffer.std.data.copy_(torch.ones(action_dim))
                    print(f"📊 Initialized action normalization with defaults for dim={action_dim}")
    
    @modal.method()
    def run_inference(self, image, state, task="Pick up the object"):
        """
        Run inference on image and state
        
        Args:
            image: numpy array of shape (H, W, 3) or list
            state: list or array of joint positions
            task: task description string
        
        Returns:
            dict with action and metadata
        """
        import torch
        import cv2
        import numpy as np
        
        try:
            # Convert image to numpy array if needed
            if isinstance(image, list):
                image = np.array(image, dtype=np.uint8)

            # Ensure image is correct shape
            if image.shape[:2] != (224, 224):
                image = cv2.resize(image, (224, 224))

            # Convert to tensor and normalize
            image_tensor = torch.from_numpy(image).float() / 255.0

            # Ensure correct shape (C, H, W)
            if image_tensor.dim() == 3 and image_tensor.shape[2] == 3:
                image_tensor = image_tensor.permute(2, 0, 1)

            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0).to(self.device)

            # Convert state to tensor, ensuring it's 6-DoF for the model
            state_tensor = torch.tensor(state[:6], dtype=torch.float32).unsqueeze(0).to(self.device)

            # Create batch
            batch = {
                "observation.image": image_tensor,
                "observation.state": state_tensor,
                "task_description": [task],  # Pass task description
            }

            # Run inference
            with torch.no_grad():
                output = self.policy(batch)

            # Extract action
            action = output['action'].cpu().numpy().squeeze()

            # Pad action to 7-DoF for robot control if needed
            if action.shape[0] == 6:
                action = np.append(action, 0.0) # Add gripper value

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
                "action": [0.0] * 7,
            }

# Direct function for easier access
@app.function(gpu="T4", timeout=300)
def get_action_direct(image, state, task="Pick up the object"):
    """Direct function call for inference"""
    inference = SmolVLAInference()
    return inference.run_inference.local(image, state, task)

# FastAPI endpoint
@app.function(gpu="T4", scaledown_window=300)
@modal.asgi_app()
def fastapi_app():
    """Create FastAPI app for REST endpoint"""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional
    
    class InferenceRequest(BaseModel):
        image: List  # Image as list
        state: List[float] = [0.0] * 7
        task: str = "Pick up the object"
    
    web_app = FastAPI()
    inference = SmolVLAInference()
    
    @web_app.post("/inference")
    async def inference_endpoint(request: InferenceRequest):
        try:
            result = inference.run_inference.local(
                request.image,
                request.state,
                request.task
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @web_app.get("/health")
    async def health_check():
        return {"status": "healthy"}
    
    return web_app

# Test endpoint
@app.local_entrypoint()
def test_inference():
    """Test the inference locally"""
    print("🧪 Testing SmolVLA inference...")
    
    import numpy as np
    
    # Create dummy inputs
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_state = [0.0] * 7  # Use 7-DoF state to simulate real-world input
    dummy_task = "Pick up the red block"
    
    # Test the direct function
    result = get_action_direct.remote(
        dummy_image.tolist(), 
        dummy_state,
        dummy_task
    )
    
    print(f"✅ Result: {result}")
    return result