"""Optional: Alternative web server approach for SmolVLA inference.

Note: This is optional - the main inference is handled by modal_runner.py.
This file shows how you could also expose it as a web API if needed.
"""

import modal
from modal import Image, App, web_endpoint
import numpy as np
from typing import Dict, Any
import json

# Use the same image as modal_runner.py
LERO_IMAGE = (
    modal.Image.debian_slim()
    .apt_install("git", "build-essential", "cmake", "pkg-config")
    .run_commands("git clone https://github.com/huggingface/lerobot.git /lerobot")
    .pip_install(
        "torch", 
        "transformers", 
        "accelerate",
        "sentencepiece", 
        "numpy", 
        "opencv-python-headless"
    )
    .run_commands("cd /lerobot && pip install -e .")
)

app = App("lerobot-web-server", image=LERO_IMAGE)

@app.function(
    gpu="A10G",
    memory=8192,
    timeout=600,
    keep_warm=1,
)
@web_endpoint(method="POST")
def predict(request_data: Dict[str, Any]):
    """
    Web endpoint for SmolVLA inference.
    
    Expected request format:
    {
        "image": [[...]] or "base64_image": "...",
        "state": [7 joint values],
    }
    """
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        import torch
        
        # Load model once
        if not hasattr(predict, "policy"):
            predict.policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
            predict.policy.eval()
        
        # Extract data
        image_data = request_data.get("image")
        state_data = request_data.get("state")
        
        if image_data is None or state_data is None:
            return {"error": "Missing image or state data"}
        
        # Convert to numpy arrays
        image = np.array(image_data, dtype=np.float32)
        state = np.array(state_data, dtype=np.float32)
        
        # Preprocess
        if len(image.shape) == 3 and image.shape[-1] == 3:
            image = np.transpose(image, (2, 0, 1))
        
        image_tensor = torch.tensor(image).float().unsqueeze(0)
        state_tensor = torch.tensor(state).float().unsqueeze(0)
        
        inputs = {
            "observation.image": image_tensor,
            "observation.state": state_tensor,
        }
        
        # Inference
        with torch.no_grad():
            action = predict.policy(inputs)
        
        return {
            "status": "success",
            "action": action.cpu().numpy().squeeze(0).tolist()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.function(
    memory=1024,
)
@web_endpoint(method="GET")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "message": "SmolVLA web server is running"}

if __name__ == "__main__":
    print("Optional: Starting SmolVLA web server...")
    print("Note: Use modal_runner.py for direct RPC inference instead")
