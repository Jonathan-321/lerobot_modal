# robot_controller.py - RUN THIS ON LINUX WITH ROBOT
import cv2
import numpy as np
import serial
import time
import modal

# Configuration
CAMERA_INDEX = 16  # Change based on your setup
ROBOT_PORT = '/dev/ttyACM0'
BAUDRATE = 1000000

class SimpleRobotController:
    def __init__(self):
        # Get Modal function - this is all you need!
        self.predict = modal.Function.lookup("smolvla-simple", "predict_action")
        
        # Camera setup
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        if not self.camera.isOpened():
            print("⚠️ No camera - using dummy images")
            self.camera = None
        
        # Robot setup
        try:
            self.robot = serial.Serial(ROBOT_PORT, BAUDRATE, timeout=0.1)
            print("✅ Robot connected")
        except:
            print("❌ No robot - simulation mode")
            self.robot = None
        
        self.state = [0.0] * 7  # Current joint positions
    
    def get_image(self):
        """Get camera image or dummy"""
        if self.camera:
            ret, frame = self.camera.read()
            if ret:
                return cv2.resize(frame, (224, 224))
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    def move_servos(self, actions):
        """Send actions to robot"""
        if not self.robot:
            print(f"[SIM] Actions: {[f'{a:.2f}' for a in actions[:3]]}...")
            return
        
        # Send to each servo
        for i, action in enumerate(actions[:7]):
            servo_id = i + 1
            
            # Convert action [-1,1] to servo position [1000,3000]
            position = int(2000 + action * 1000)
            position = max(1000, min(3000, position))
            
            # Send position command
            cmd = bytearray([
                0xFF, 0xFF, servo_id, 0x05, 0x03, 0x1E,
                position & 0xFF, (position >> 8) & 0xFF
            ])
            checksum = (~sum(cmd[2:])) & 0xFF
            cmd.append(checksum)
            
            self.robot.write(cmd)
            time.sleep(0.005)
        
        print(f"✅ Sent: {[f'{a:.2f}' for a in actions[:3]]}...")
    
    def run_task(self, task, steps=30):
        """Main control loop"""
        print(f"\n🤖 Task: {task}")
        print("Press Ctrl+C to stop\n")
        
        for step in range(steps):
            try:
                # Get image
                image = self.get_image()
                
                # Get action from Modal (this is the magic!)
                action = self.predict.remote(
                    image.tolist(),
                    self.state,
                    task
                )
                
                print(f"Step {step+1}/{steps}")
                
                # Execute action
                self.move_servos(action)
                self.state = action
                
                # Control rate
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                print("\n🛑 Stopped")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)
        
        # Cleanup
        if self.robot:
            # Release all servos
            for i in range(1, 8):
                cmd = bytearray([0xFF, 0xFF, i, 0x04, 0x03, 0x18, 0x00])
                checksum = (~sum(cmd[2:])) & 0xFF
                cmd.append(checksum)
                self.robot.write(cmd)
            self.robot.close()
        
        if self.camera:
            self.camera.release()

# Quick test script
if __name__ == "__main__":
    controller = SimpleRobotController()
    
    # Simple menu
    print("\nTasks:")
    print("1. Open gripper")
    print("2. Close gripper")
    print("3. Pick up object")
    print("4. Custom task")
    
    choice = input("\nChoice (1-4): ")
    
    tasks = {
        "1": "Open the gripper",
        "2": "Close the gripper",
        "3": "Pick up the red block",
        "4": input("Enter task: ") if choice == "4" else "Move to home"
    }
    
    task = tasks.get(choice, "Open the gripper")
    controller.run_task(task, steps=20)