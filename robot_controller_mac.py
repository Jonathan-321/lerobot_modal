# robot_controller_mac.py - Run this on your Mac
import subprocess
import numpy as np
import time
import modal

# For Mac, we'll use system commands to find the robot
def find_robot_port():
    """Find robot USB port on Mac"""
    try:
        # List USB devices
        result = subprocess.run(['ls', '/dev/tty.*'], capture_output=True, text=True, shell=True)
        ports = result.stdout.strip().split('\n')
        
        # Look for USB serial ports
        for port in ports:
            if 'usbserial' in port or 'usbmodem' in port or 'ACM' in port:
                print(f"Found potential robot port: {port}")
                return port
    except:
        pass
    return None

class SimpleRobotController:
    def __init__(self):
        # Get Modal function
        self.predict = modal.Function.lookup("smolvla-simple", "predict_action")
        
        # Find robot port
        self.port = find_robot_port()
        if self.port:
            try:
                import serial
                self.robot = serial.Serial(self.port, 1000000, timeout=0.1)
                print(f"✅ Robot connected on {self.port}")
            except:
                print("❌ Could not open robot port")
                self.robot = None
        else:
            print("❌ No robot found")
            self.robot = None
        
        # For Mac, we might not have camera access easily
        self.use_dummy_camera = True
        self.state = [0.0] * 7
    
    def get_image(self):
        """Get dummy image for testing"""
        # Create a simple test pattern
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        # Add some features so the model has something to work with
        image[50:150, 50:150] = [255, 0, 0]  # Red square
        return image
    
    def move_servos(self, actions):
        """Send actions to robot"""
        if not self.robot:
            print(f"[SIM] Actions: {[f'{a:.2f}' for a in actions[:3]]}...")
            return
        
        import serial
        
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
    
    def run_task(self, task, steps=20):
        """Main control loop"""
        print(f"\n🤖 Task: {task}")
        print("Press Ctrl+C to stop\n")
        
        for step in range(steps):
            try:
                # Get image
                image = self.get_image()
                
                # Get action from Modal
                print(f"Step {step+1}/{steps}: Getting action from Modal...", end='', flush=True)
                start = time.time()
                action = self.predict.remote(
                    image.tolist(),
                    self.state,
                    task
                )
                print(f" ({time.time()-start:.1f}s)")
                
                # Execute action
                self.move_servos(action)
                self.state = action
                
                # Control rate
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                print("\n🛑 Stopped")
                break
            except Exception as e:
                print(f"\nError: {e}")
                time.sleep(1)
        
        # Cleanup
        if self.robot:
            print("Releasing servos...")
            for i in range(1, 8):
                cmd = bytearray([0xFF, 0xFF, i, 0x04, 0x03, 0x18, 0x00])
                checksum = (~sum(cmd[2:])) & 0xFF
                cmd.append(checksum)
                self.robot.write(cmd)
            self.robot.close()

if __name__ == "__main__":
    # First check if we have the dependencies
    try:
        import serial
    except ImportError:
        print("Installing pyserial...")
        subprocess.run(['pip', 'install', 'pyserial'])
        import serial
    
    controller = SimpleRobotController()
    
    # Simple menu
    print("\nTasks:")
    print("1. Open gripper")
    print("2. Close gripper")
    print("3. Wave")
    print("4. Home position")
    
    choice = input("\nChoice (1-4): ")
    
    tasks = {
        "1": "Open the gripper",
        "2": "Close the gripper",
        "3": "Wave the arm",
        "4": "Move to home position"
    }
    
    task = tasks.get(choice, "Open the gripper")
    controller.run_task(task, steps=10)