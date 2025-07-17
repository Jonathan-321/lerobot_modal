# so101_control_simple.py
"""
Simplified SO-101 Robot Control
Only the essentials for making the robot work with SmolVLA
"""

import cv2
import numpy as np
import time
import serial
import modal
from typing import List, Optional

class SO101Controller:
    """Minimal controller for SO-101 robot"""
    
    def __init__(self):
        # Modal function
        self.get_action = modal.Function.from_name(
            "lerobot-smolvla-inference",
            "get_action_direct"
        )
        
        # Camera - try multiple indices
        self.camera = None
        for idx in [0, 7, 16]:  # Known working indices
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.camera = cap
                    self.camera_idx = idx
                    print(f"✅ Camera found at index {idx}")
                    break
                cap.release()
        
        if not self.camera:
            print("❌ No camera found")
        
        # Robot connection
        try:
            self.robot = serial.Serial('/dev/ttyACM0', 1000000, timeout=0.1)
            print("✅ Robot connected")
        except:
            self.robot = None
            print("❌ Robot not connected")
        
        # State
        self.current_pos = [0.0] * 7
    
    def move_robot(self, actions: List[float]):
        """Send actions to robot"""
        if not self.robot:
            print(f"[SIM] Move: {[f'{a:.2f}' for a in actions[:3]]}...")
            return
        
        # Convert actions [-1,1] to servo positions [1000,3000]
        for i, action in enumerate(actions[:7]):
            servo_id = i + 1
            pos = int(2000 + action * 1000)
            pos = max(1000, min(3000, pos))  # Safety limits
            
            # Build command
            cmd = bytearray([
                0xFF, 0xFF, servo_id, 0x05, 0x03, 0x1E,
                pos & 0xFF, (pos >> 8) & 0xFF
            ])
            checksum = (~sum(cmd[2:])) & 0xFF
            cmd.append(checksum)
            
            # Send
            self.robot.write(cmd)
            time.sleep(0.005)
        
        print(f"✅ Sent: {[f'{a:.2f}' for a in actions[:3]]}...")
    
    def run(self, task: str, max_steps: int = 30):
        """Main control loop"""
        print(f"\n🤖 Task: {task}")
        print("Press Ctrl+C to stop\n")
        
        for step in range(max_steps):
            try:
                # Get camera image
                if self.camera:
                    ret, frame = self.camera.read()
                    if not ret:
                        print("Camera error")
                        break
                    image = cv2.resize(frame, (224, 224))
                else:
                    # Dummy image if no camera
                    image = np.zeros((224, 224, 3), dtype=np.uint8)
                
                # Get action from Modal
                start = time.time()
                result = self.get_action.remote(
                    image.tolist(),
                    self.current_pos,
                    task
                )
                
                if result['success']:
                    actions = result['action']
                    print(f"Step {step+1}: Inference {time.time()-start:.1f}s")
                    
                    # Execute action
                    self.move_robot(actions)
                    self.current_pos = actions
                else:
                    print(f"Error: {result.get('error')}")
                
                time.sleep(0.5)  # 2Hz control
                
            except KeyboardInterrupt:
                print("\n🛑 Stopped")
                break
            except Exception as e:
                print(f"Error: {e}")
                break
        
        self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        if self.robot:
            # Disable torque on all servos
            for i in range(1, 8):
                cmd = bytearray([0xFF, 0xFF, i, 0x04, 0x03, 0x18, 0x00])
                checksum = (~sum(cmd[2:])) & 0xFF
                cmd.append(checksum)
                self.robot.write(cmd)
                time.sleep(0.01)
            self.robot.close()
        
        if self.camera:
            self.camera.release()
        
        print("✅ Cleanup done")

def main():
    """Simple main function"""
    controller = SO101Controller()
    
    if not controller.robot:
        print("\n⚠️  No robot connected - running in simulation")
    
    # Simple task selection
    print("\nTasks:")
    print("1. Pick up object")
    print("2. Open gripper")
    print("3. Home position")
    
    choice = input("\nChoice (1-3): ")
    tasks = {
        "1": "Pick up the red block",
        "2": "Open the gripper", 
        "3": "Move to home position"
    }
    
    task = tasks.get(choice, "Pick up the object")
    controller.run(task)

if __name__ == "__main__":
    main()