# test_base_model.py
import modal
import numpy as np

# Test different basic tasks
test_tasks = [
    "Open the gripper",
    "Close the gripper", 
    "Move up",
    "Move down",
    "Move to the left",
    "Move to the right",
    "Pick up the object",
    "Put down the object"
]

predict = modal.Function.lookup("smolvla-simple", "predict_action")

for task in test_tasks:
    print(f"\nTesting: {task}")
    
    # Dummy inputs
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    state = [0.0] * 7
    
    # Get action
    action = predict.remote(image.tolist(), state, task)
    print(f"Action: {[f'{a:.2f}' for a in action[:3]]}...")