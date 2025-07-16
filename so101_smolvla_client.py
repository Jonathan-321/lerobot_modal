# so101_smolvla_client.py (updated)
import cv2
import numpy as np
import requests
import json
import base64
import time
from typing import Optional, List, Dict
from threading import Thread, Lock
import queue

class SO101SmolVLAClient:
    def __init__(self, modal_endpoint_url: str):
        """
        Initialize the SO-101 SmolVLA client
        
        Args:
            modal_endpoint_url: Your Modal endpoint URL
        """
        self.endpoint = modal_endpoint_url
        self.camera = None
        self.action_queue = queue.Queue(maxsize=10)
        self.latest_action = [0.0] * 7
        self.action_lock = Lock()
        self.running = False
        
        # Initialize camera
        self._init_camera()
        
        print(f"✅ SO-101 SmolVLA Client initialized")
        print(f"📡 Endpoint: {self.endpoint}")
        
    def _init_camera(self):
        """Initialize camera with retries"""
        for i in range(3):
            try:
                self.camera = cv2.VideoCapture(0)
                if self.camera.isOpened():
                    # Set camera properties
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    print("✅ Camera initialized")
                    return
            except Exception as e:
                print(f"Camera attempt {i+1} failed: {e}")
            time.sleep(1)
        print("⚠️ Camera initialization failed - will use dummy images")
        print("💡 On macOS: Grant camera permission in System Preferences > Security & Privacy > Camera")
        
    def get_robot_state(self) -> List[float]:
        """
        Get current robot joint positions
        Replace this with actual SO-101 API calls
        """
        # TODO: Replace with actual SO-101 robot state reading
        # For now, return dummy state
        return [0.0, 0.1, -0.1, 0.0, 0.2, -0.2, 0.5]
        
    def send_action_to_robot(self, action: List[float]):
        """
        Send action to SO-101 robot
        Replace this with actual SO-101 API calls
        """
        # TODO: Replace with actual SO-101 command sending
        print(f"🤖 Sending action to robot: {[f'{a:.3f}' for a in action]}")
        
    def capture_image(self) -> Optional[np.ndarray]:
        """Capture and preprocess image from camera"""
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                # Resize to 224x224 as expected by SmolVLA
                frame = cv2.resize(frame, (224, 224))
                return frame
        
        # Return dummy image if camera fails
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
    def image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
        
    def call_modal_endpoint(self, image: np.ndarray, robot_state: List[float], task: str) -> Optional[List[float]]:
        """Call the Modal endpoint for inference"""
        try:
            # Modal web endpoints expect JSON with proper content-type
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            # Prepare request data
            request_data = {
                "image": self.image_to_base64(image),
                "robot_state": robot_state,
                "task": task
            }
            
            # Make request with timeout
            response = requests.post(
                self.endpoint,
                json=request_data,
                headers=headers,
                timeout=5.0  # Increased timeout for first call
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    return result.get('action')
                else:
                    print(f"❌ Inference failed: {result.get('error')}")
            else:
                print(f"❌ HTTP error: {response.status_code}")
                print(f"Response: {response.text[:200]}")  # Show first 200 chars
                
        except requests.exceptions.Timeout:
            print("⚠️ Request timeout - using previous action")
        except Exception as e:
            print(f"❌ Request error: {e}")
            
        return None
        
    def control_loop(self, task: str = "Pick up the object", display: bool = True):
        """
        Main control loop
        
        Args:
            task: Task instruction for SmolVLA
            display: Whether to display camera feed
        """
        print(f"\n🚀 Starting control loop")
        print(f"📋 Task: {task}")
        print("Press 'q' to quit\n")
        
        self.running = True
        inference_count = 0
        total_inference_time = 0
        
        try:
            while self.running:
                loop_start = time.time()
                
                # 1. Capture image
                image = self.capture_image()
                if image is None:
                    continue
                    
                # 2. Get robot state
                robot_state = self.get_robot_state()
                
                # 3. Get action from SmolVLA
                inference_start = time.time()
                action = self.call_modal_endpoint(image, robot_state, task)
                inference_time = time.time() - inference_start
                
                if action:
                    # Update latest action
                    with self.action_lock:
                        self.latest_action = action
                    
                    # Send to robot
                    self.send_action_to_robot(action)
                    
                    # Track timing
                    inference_count += 1
                    total_inference_time += inference_time
                    avg_inference_time = total_inference_time / inference_count
                    
                    print(f"✅ Step {inference_count} | Inference: {inference_time:.3f}s | Avg: {avg_inference_time:.3f}s")
                
                # 4. Display camera feed (optional)
                if display:
                    # Draw task text on image
                    display_image = image.copy()
                    cv2.putText(display_image, f"Task: {task}", (10, 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Show current action
                    if action:
                        action_text = f"Action: [{action[0]:.2f}, {action[1]:.2f}, ...]"
                        cv2.putText(display_image, action_text, (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    cv2.imshow('SO-101 SmolVLA Control', display_image)
                    
                    # Check for quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Control frequency (aim for ~10Hz)
                elapsed = time.time() - loop_start
                sleep_time = max(0, 0.1 - elapsed)  # 10Hz = 0.1s per loop
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("\n⚡ Control loop interrupted")
        finally:
            self.cleanup()
            
    def test_connection(self):
        """Test the Modal endpoint connection"""
        print("\n🧪 Testing Modal endpoint...")
        
        # Create test data
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_state = [0.0] * 7
        test_task = "Test connection"
        
        # Test inference
        start = time.time()
        action = self.call_modal_endpoint(test_image, test_state, test_task)
        elapsed = time.time() - start
        
        if action:
            print(f"✅ Connection test successful!")
            print(f"⏱️  Inference time: {elapsed:.3f}s")
            print(f"🤖 Test action: {[f'{a:.3f}' for a in action]}")
            return True
        else:
            print("❌ Connection test failed")
            return False
            
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")

def test_endpoint_directly():
    """Test the endpoint with curl equivalent"""
    import requests
    
    endpoint = "https://muhirejonathan123--smolvla-so101-deployment-inference-endpoint.modal.run"
    
    # Test data
    test_data = {
        "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",  # 1x1 red pixel
        "robot_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "task": "Pick up the object"
    }
    
    print("🧪 Direct endpoint test...")
    try:
        response = requests.post(
            endpoint,
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:500]}")  # First 500 chars
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Example usage"""
    # Your Modal endpoint
    ENDPOINT = "https://muhirejonathan123--smolvla-so101-deployment-inference-endpoint.modal.run"
    
    # First test the endpoint directly
    print("="*50)
    test_endpoint_directly()
    print("="*50)
    
    # Create client
    client = SO101SmolVLAClient(ENDPOINT)
    
    # Test connection
    if not client.test_connection():
        print("\n💡 Troubleshooting tips:")
        print("1. Check if the Modal app is still deployed")
        print("2. Try redeploying: modal deploy modal_smolvla_deployment.py")
        print("3. Check Modal dashboard for logs")
        return
    
    # Run control loop with specific task
    print("\n" + "="*50)
    print("Choose a task:")
    print("1. Pick up the red block")
    print("2. Move forward")
    print("3. Open the gripper")
    print("4. Custom task")
    print("5. Run without display (headless)")
    
    choice = input("\nEnter choice (1-5): ")
    
    tasks = {
        "1": "Pick up the red block",
        "2": "Move forward",
        "3": "Open the gripper",
        "4": input("Enter custom task: ") if choice == "4" else "",
        "5": "Pick up the object"
    }
    
    task = tasks.get(choice, "Pick up the object")
    display = choice != "5"
    
    # Start control
    client.control_loop(task=task, display=display)

if __name__ == "__main__":
    main()