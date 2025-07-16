# so101_real_robot_control.py
import modal
import cv2
import numpy as np
import time
import serial
import serial.tools.list_ports
from typing import List, Optional
import struct

class SO101RobotController:
    def __init__(self):
        # Get Modal function
        self.get_action_fn = modal.Function.lookup(
            "smolvla-so101-deployment", 
            "get_action_direct"
        )
        
        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        
        # Find and connect to SO-101
        self.serial_port = self._find_so101_port()
        if self.serial_port:
            print(f"✅ Connected to SO-101 on {self.serial_port.name}")
        else:
            print("⚠️ SO-101 not found - running in simulation mode")
            
        # Robot state
        self.current_positions = [0.0] * 7  # 7 joints
        self.target_positions = [0.0] * 7
        
        print("🤖 SO-101 SmolVLA Controller Ready!")
        
    def _find_so101_port(self) -> Optional[serial.Serial]:
        """Find SO-101 USB serial port"""
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            print(f"Found port: {port.device} - {port.description}")
            
            # SO-101 typically shows up as these
            if any(name in port.description.lower() for name in ['so-101', 'so101', 'feetech', 'dynamixel', 'usb serial']):
                try:
                    ser = serial.Serial(
                        port=port.device,
                        baudrate=115200,  # SO-101 default baud rate
                        timeout=0.1,
                        write_timeout=0.1
                    )
                    print(f"✅ Connected to {port.device}")
                    return ser
                except Exception as e:
                    print(f"Failed to open {port.device}: {e}")
                    
        return None
    
    def read_robot_state(self) -> List[float]:
        """Read current joint positions from SO-101"""
        if not self.serial_port:
            return self.current_positions
            
        try:
            # SO-101 protocol: Send read command
            # Format depends on your specific SO-101 firmware
            # Common format: [0xFF, 0xFF, ID, LENGTH, INSTRUCTION, PARAMS, CHECKSUM]
            
            # Example: Read all joint positions
            cmd = bytearray([0xFF, 0xFF, 0xFE, 0x04, 0x02, 0x24, 0x02])  # Read present position
            checksum = (~(sum(cmd[2:]) & 0xFF)) & 0xFF
            cmd.append(checksum)
            
            self.serial_port.write(cmd)
            time.sleep(0.01)
            
            # Read response
            if self.serial_port.in_waiting > 0:
                response = self.serial_port.read(self.serial_port.in_waiting)
                # Parse response - this depends on your protocol
                # For now, return current positions
                return self.current_positions
                
        except Exception as e:
            print(f"Error reading robot state: {e}")
            
        return self.current_positions
    
    def send_joint_positions(self, positions: List[float]):
        """Send target positions to SO-101"""
        if not self.serial_port:
            print(f"🤖 [SIM] Moving to: {[f'{p:.3f}' for p in positions]}")
            self.current_positions = positions
            return
            
        try:
            # Convert radians to servo units (0-4095 for Dynamixel/Feetech)
            # Adjust these conversion factors based on your servos
            servo_positions = []
            for i, pos in enumerate(positions):
                # Map [-π, π] to [0, 4095]
                servo_pos = int((pos + np.pi) / (2 * np.pi) * 4095)
                servo_pos = np.clip(servo_pos, 0, 4095)
                servo_positions.append(servo_pos)
            
            # Send to each servo
            for servo_id, servo_pos in enumerate(servo_positions):
                # Dynamixel/Feetech protocol
                cmd = bytearray([
                    0xFF, 0xFF,  # Header
                    servo_id + 1,  # Servo ID (1-indexed)
                    0x07,  # Length
                    0x03,  # Write command
                    0x2A,  # Goal position register
                    servo_pos & 0xFF,  # Low byte
                    (servo_pos >> 8) & 0xFF,  # High byte
                    0x00, 0x02  # Speed (512)
                ])
                
                checksum = (~(sum(cmd[2:-2]) & 0xFF)) & 0xFF
                cmd.append(checksum)
                
                self.serial_port.write(cmd)
                time.sleep(0.005)  # Small delay between commands
                
            self.current_positions = positions
            print(f"✅ Sent positions to robot")
            
        except Exception as e:
            print(f"Error sending positions: {e}")
    
    def capture_and_process(self, task: str):
        """Capture image and get action from SmolVLA"""
        # Capture image
        ret, frame = self.camera.read()
        if not ret:
            print("Failed to capture image")
            return None
            
        # Resize for model
        model_frame = cv2.resize(frame, (224, 224))
        
        # Get current robot state
        current_state = self.read_robot_state()
        
        # Call SmolVLA
        start = time.time()
        result = self.get_action_fn.remote(
            model_frame.tolist(),
            current_state,
            task
        )
        inference_time = time.time() - start
        
        if result['success']:
            action = result['action']
            print(f"✅ Inference: {inference_time:.3f}s | Action: {[f'{a:.3f}' for a in action[:3]]}...")
            return action, frame
        else:
            print(f"❌ Inference failed: {result.get('error')}")
            return None, frame
    
    def control_loop(self, task: str = "Pick up the object", max_steps: int = None):
        """Main control loop"""
        print(f"\n🚀 Starting SO-101 control")
        print(f"📋 Task: {task}")
        print(f"🔌 Port: {self.serial_port.name if self.serial_port else 'Simulation'}")
        print("Press 'q' to quit, 's' for emergency stop\n")
        
        step = 0
        
        try:
            while True:
                if max_steps and step >= max_steps:
                    print(f"Reached max steps ({max_steps})")
                    break
                    
                # Get action from SmolVLA
                action, frame = self.capture_and_process(task)
                
                if action is not None:
                    # Apply safety limits
                    safe_action = self.apply_safety_limits(action)
                    
                    # Send to robot
                    self.send_joint_positions(safe_action)
                    
                    step += 1
                
                # Display
                if frame is not None:
                    cv2.putText(frame, f"Task: {task}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Step: {step}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Show current positions
                    pos_text = f"Pos: [{self.current_positions[0]:.2f}, {self.current_positions[1]:.2f}, ...]"
                    cv2.putText(frame, pos_text, (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    cv2.imshow('SO-101 SmolVLA Control', frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    print("🛑 EMERGENCY STOP!")
                    self.emergency_stop()
                    
                # Control rate (10Hz)
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n⚡ Interrupted")
        finally:
            self.cleanup()
    
    def apply_safety_limits(self, action: List[float]) -> List[float]:
        """Apply safety limits to prevent damage"""
        # Joint limits for SO-101 (in radians)
        # Adjust these based on your robot's specifications
        joint_limits = [
            (-2.0, 2.0),    # Base rotation
            (-1.5, 1.5),    # Shoulder
            (-1.5, 1.5),    # Elbow
            (-2.0, 2.0),    # Wrist rotation
            (-1.5, 1.5),    # Wrist pitch
            (-2.0, 2.0),    # Wrist roll
            (0.0, 1.0),     # Gripper (0=closed, 1=open)
        ]
        
        # Apply limits
        safe_action = []
        for i, (val, (min_val, max_val)) in enumerate(zip(action, joint_limits)):
            safe_val = np.clip(val, min_val, max_val)
            
            # Also limit velocity (max change per step)
            max_delta = 0.2  # radians per step
            if i < len(self.current_positions):
                delta = safe_val - self.current_positions[i]
                if abs(delta) > max_delta:
                    safe_val = self.current_positions[i] + np.sign(delta) * max_delta
                    
            safe_action.append(safe_val)
            
        return safe_action
    
    def emergency_stop(self):
        """Emergency stop - halt all motors"""
        if self.serial_port:
            # Send stop command to all servos
            # This depends on your servo protocol
            stop_cmd = bytearray([0xFF, 0xFF, 0xFE, 0x02, 0x06])  # Broadcast stop
            checksum = (~(sum(stop_cmd[2:]) & 0xFF)) & 0xFF
            stop_cmd.append(checksum)
            self.serial_port.write(stop_cmd)
            
    def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        self.emergency_stop()
        if self.serial_port:
            self.serial_port.close()
        self.camera.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")

def main():
    """Main entry point"""
    controller = SO101RobotController()
    
    # Test connection
    if controller.serial_port:
        print("\n✅ SO-101 connected and ready!")
        print("\n⚠️  SAFETY CHECK:")
        print("1. Is the robot in a safe position?")
        print("2. Is the workspace clear?")
        print("3. Are you ready to monitor the robot?")
        
        confirm = input("\nType 'yes' to continue: ")
        if confirm.lower() != 'yes':
            print("Aborted.")
            return
    
    # Task selection
    print("\n📋 Select a task:")
    print("1. Pick up the red block")
    print("2. Stack blocks")
    print("3. Open gripper")
    print("4. Home position")
    print("5. Custom task")
    
    choice = input("\nEnter choice (1-5): ")
    
    tasks = {
        "1": "Pick up the red block",
        "2": "Stack the blue block on the red block",
        "3": "Open the gripper",
        "4": "Move to home position",
        "5": input("Enter custom task: ") if choice == "5" else ""
    }
    
    task = tasks.get(choice, "Pick up the object")
    
    # Run control loop
    controller.control_loop(task)

if __name__ == "__main__":
    main()