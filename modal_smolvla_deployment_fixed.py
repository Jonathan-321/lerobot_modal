# modal_smolvla_deployment_fixed.py
import modal
from typing import List, Dict, Union
import numpy as np

app = modal.App("smolvla-so101-deployment")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch", 
        "transformers", 
        "num2words", 
        "accelerate", 
        "pillow", 
        "numpy",
        "fastapi",
        "pydantic",
    )
    .run_commands("git clone https://github.com/huggingface/lerobot.git /lerobot")
    .run_commands("cd /lerobot && pip install -e .")
)

@app.cls(
    image=image,
    gpu="T4",
    scaledown_window=300,
)
class SmolVLAController:
    @modal.enter()
    def setup(self):
        """Initialize the model once when container starts"""
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        import torch
        
        print("🚀 Initializing SmolVLA for SO-101...")
        
        # Load model
        self.policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        self.policy.eval()
        
        # Move to GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = self.policy.to(self.device)
        
        # Fix normalization parameters
        self._fix_normalization()
        
        print(f"✅ Model ready on {self.device}")
        
    def _fix_normalization(self):
        """Fix the infinity normalization issue"""
        import torch
        
        if hasattr(self.policy.normalize_inputs, 'buffer_observation_state'):
            self.policy.normalize_inputs.buffer_observation_state.mean.data = torch.zeros(6).to(self.device)
            self.policy.normalize_inputs.buffer_observation_state.std.data = torch.ones(6).to(self.device)
        
        if hasattr(self.policy.normalize_targets, 'buffer_action'):
            self.policy.normalize_targets.buffer_action.mean.data = torch.zeros(6).to(self.device)
            self.policy.normalize_targets.buffer_action.std.data = torch.ones(6).to(self.device)
        
        if hasattr(self.policy.unnormalize_outputs, 'buffer_action'):
            self.policy.unnormalize_outputs.buffer_action.mean.data = torch.zeros(6).to(self.device)
            self.policy.unnormalize_outputs.buffer_action.std.data = torch.ones(6).to(self.device)
    
    @modal.method()
    def get_action(
        self, 
        image: Union[np.ndarray, str],
        robot_state: List[float],
        task: str = "Pick up the object"
    ) -> Dict:
        """Get robot action from current observation"""
        import torch
        import numpy as np
        from PIL import Image
        import base64
        from io import BytesIO
        
        try:
            # Handle base64 image
            if isinstance(image, str):
                image_data = base64.b64decode(image)
                pil_image = Image.open(BytesIO(image_data))
                image = np.array(pil_image)
            
            # Convert to tensor
            image_tensor = torch.from_numpy(image).float() / 255.0
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.permute(2, 0, 1)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Handle state dimensions
            state_6d = robot_state[:6] if len(robot_state) >= 6 else robot_state + [0] * (6 - len(robot_state))
            state_tensor = torch.tensor(state_6d).float().unsqueeze(0).to(self.device)
            
            # Prepare batch
            batch = {
                "observation.image": image_tensor,
                "observation.state": state_tensor,
                "task": task,
            }
            
            # Get action
            with torch.no_grad():
                action = self.policy.select_action(batch)
            
            # Convert to numpy
            action_np = action.cpu().numpy()
            if action_np.ndim > 1:
                action_np = action_np[0]
            
            # Map to 7D
            action_7d = np.zeros(7)
            action_7d[:6] = action_np
            action_7d[6] = action_np[5]
            
            return {
                "success": True,
                "action": action_7d.tolist(),
                "raw_action": action_np.tolist(),
                "task": task
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return {
                "success": False,
                "error": str(e),
                "action": [0.0] * 7
            }

# FastAPI endpoint with proper request model
@app.function(image=image, gpu="T4", scaledown_window=300)
@modal.fastapi_endpoint()  # Use the new decorator
def inference_endpoint():
    from fastapi import FastAPI
    from pydantic import BaseModel
    from typing import List, Optional
    
    web_app = FastAPI()
    controller = SmolVLAController()
    
    class InferenceRequest(BaseModel):
        image: str  # base64 encoded
        robot_state: List[float]
        task: Optional[str] = "Pick up the object"
    
    class InferenceResponse(BaseModel):
        success: bool
        action: List[float]
        raw_action: Optional[List[float]] = None
        task: Optional[str] = None
        error: Optional[str] = None
    
    @web_app.post("/", response_model=InferenceResponse)
    async def predict(request: InferenceRequest):
        result = controller.get_action.remote(
            request.image,
            request.robot_state,
            request.task
        )
        return InferenceResponse(**result)
    
    @web_app.get("/")
    async def health_check():
        return {"status": "healthy", "model": "SmolVLA", "target": "SO-101"}
    
    return web_app

# Direct function for programmatic access
@app.function(image=image, gpu="T4")
def get_action_direct(image_array: List, robot_state: List[float], task: str = "Pick up the object"):
    """Direct function call without HTTP"""
    controller = SmolVLAController()
    # Convert list back to numpy array
    import numpy as np
    image = np.array(image_array, dtype=np.uint8)
    return controller.get_action.remote(image, robot_state, task)

# Test function
@app.local_entrypoint()
def test():
    import numpy as np
    
    # Test direct function
    print("🧪 Testing direct function call...")
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_state = [0.0, 0.1, -0.1, 0.0, 0.2, -0.2, 0.5]
    
    result = get_action_direct.remote(dummy_image.tolist(), dummy_state, "Pick up the red block")
    
    print(f"\n✅ Test Result:")
    print(f"Success: {result['success']}")
    print(f"Action (7D): {result['action']}")
    print(f"Raw Action (6D): {result['raw_action']}")
    
    print("\n🎉 SmolVLA deployed! Ready for SO-101!")