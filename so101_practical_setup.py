# so101_practical_setup.py
import modal
import cv2
import numpy as np
import time
import serial
import serial.tools.list_ports
from typing import List, Optional, Tuple
import json
import os
from datetime import datetime

class SO101PracticalController:
    def __init__(self, camera_index=0, simulate=False):
        """
        Initialize controller with practical considerations
        
        Args:
            camera_index: 0 for laptop cam, 1+ for external cameras
            simulate: Run without real robot for testing
        """
        self.simulate = simulate
        
        # Modal function
        self.get_action_fn = modal.Function.lookup(
            "smolvla-so101-deployment", 
            "get_action_direct"
        )
        
        # Camera setup with multiple options
        self.camera = self._setup_camera(camera_index)
        
        # Robot connection
        if not simulate:
            self.serial_port = self._find_so101_port()
            if not self.serial_port:
                print("⚠️ No robot found - switching to simulation mode")
                self.simulate = True
        else:
            self.serial_port = None
            print("🎮 Running in simulation mode")
        
        # State tracking
        self.current_positions = [0.0] * 7
        self.command_history = []
        self.safety_enabled = True
        
        # Logging
        self.log_dir = f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.log_dir, exist_ok=True)
        
    def _setup_camera(self, index) -> cv2.VideoCapture:
        """Setup camera with fallback options"""
        # Try requested camera
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"✅ Camera {index} opened successfully")
            # Set resolution for better quality
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
        
        # Try other cameras
        print(f"Camera {index} failed, trying others...")
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"✅ Using camera {i} instead")
                return cap
        
        print("❌ No camera available!")
        return None
    
    def _find_so101_port(self) -> Optional[serial.Serial]:
        """Find SO-101 with better detection"""
        ports = serial.tools.list_ports.comports()
        
        # Common SO-101 identifiers
        so101_identifiers = [
            'so-101', 'so101', 'iso-101', 'iso101',
            'feetech', 'dynamixel', 'robotis',
            'ch340', 'cp210', 'ftdi'  # Common USB-serial chips
        ]
        
        for port in ports:
            port_info = f"{port.device}: {port.description} [{port.hwid}]"
            print(f"Found: {port_info}")
            
            # Check if any identifier matches
            if any(id in port_info.lower() for id in so101_identifiers):
                try:
                    ser = serial.Serial(
                        port=port.device,
                        baudrate=115200,
                        timeout=0.5,
                        write_timeout=0.5
                    )
                    
                    # Test communication
                    if self._test_robot_connection(ser):
                        print(f"✅ SO-101 confirmed on {port.device}")
                        return ser
                    else:
                        ser.close()
                        
                except Exception as e:
                    print(f"Failed {port.device}: {e}")
        
        return None
    
    def _test_robot_connection(self, ser: serial.Serial) -> bool:
        """Test if this is actually an SO-101"""
        try:
            # Send ping command (adjust for your protocol)
            ser.write(b'\xFF\xFF\xFE\x02\x01\xFB')  # Dynamixel ping
            time.sleep(0.1)
            
            if ser.in_waiting > 0:
                response = ser.read(ser.in_waiting)
                return len(response) > 0
        except:
            pass
        return False
    
    def capture_workspace(self) -> Tuple[np.ndarray, np.ndarray]:
        """Capture and preprocess image"""
        if not self.camera:
            # Generate test pattern if no camera
            test_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(test_img, "NO CAMERA", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            return test_img, cv2.resize(test_img, (224, 224))
        
        ret, frame = self.camera.read()
        if not ret:
            return None, None
        
        # Save original for display
        display_frame = frame.copy()
        
        # Crop to workspace area (adjust these values!)
        # This focuses on where the robot actually works
        workspace_roi = frame[100:580, 160:800]  # Example ROI
        
        # Resize for model
        model_input = cv2.resize(workspace_roi, (224, 224))
        
        return display_frame, model_input
    
    def validate_action(self, action: List[float]) -> Tuple[bool, str]:
        """Validate action before sending to robot"""
        # Check for NaN or inf
        if any(np.isnan(a) or np.isinf(a) for a in action):
            return False, "Invalid values in action"
        
        # Check for sudden large movements
        max_delta = 0.5  # radians
        for i, (new, old) in enumerate(zip(action, self.current_positions)):
            if abs(new - old) > max_delta:
                return False, f"Joint {i} moving too fast: {abs(new-old):.2f} rad"
        
        # Check workspace limits (customize for your setup)
        if action[1] < -1.0:  # Shoulder too low
            return False, "Shoulder position too low - collision risk"
        
        return True, "OK"
    
    def execute_action(self, action: List[float]):
        """Execute action with safety checks"""
        # Validate
        if self.safety_enabled:
            valid, msg = self.validate_action(action)
            if not valid:
                print(f"⚠️ Action blocked: {msg}")
                return
        
        # Log action
        self.command_history.append({
            'timestamp': time.time(),
            'action': action,
            'position': self.current_positions.copy()
        })
        
        # Send to robot or simulate
        if self.simulate:
            print(f"🎮 [SIM] Action: {[f'{a:.2f}' for a in action[:3]]}...")
            self.current_positions = action
        else:
            self.send_to_robot(action)
    
    def send_to_robot(self, action: List[float]):
        """Send commands to real robot"""
        # This is where you implement your specific protocol
        # Example for Dynamixel-style servos:
        
        for i, position in enumerate(action):
            # Convert radians to servo units (0-4095)
            servo_value = int((position + np.pi) / (2 * np.pi) * 4095)
            servo_value = np.clip(servo_value, 0, 4095)
            
            # Build command packet
            cmd = self._build_position_command(i+1, servo_value)
            self.serial_port.write(cmd)
            time.sleep(0.005)
        
        self.current_positions = action
    
    def _build_position_command(self, servo_id: int, position: int) -> bytes:
        """Build position command for servo"""
        # Dynamixel protocol example
        cmd = bytearray([
            0xFF, 0xFF,          # Header
            servo_id,            # ID
            0x05,                # Length
            0x03,                # Write instruction
            0x1E,                # Goal position address
            position & 0xFF,     # Position low byte
            (position >> 8) & 0xFF  # Position high byte
        ])
        
        # Calculate checksum
        checksum = (~(sum(cmd[2:]) & 0xFF)) & 0xFF
        cmd.append(checksum)
        
        return bytes(cmd)
    
    def run_task(self, task: str, max_duration: int = 60):
        """Run a specific task with monitoring"""
        print(f"\n🚀 Starting task: {task}")
        print(f"⏱️  Max duration: {max_duration}s")
        print(f"🛡️  Safety: {'ON' if self.safety_enabled else 'OFF'}")
        print("\nControls: 'q'=quit, 's'=stop, 'p'=pause, 'r'=resume\n")
        
        start_time = time.time()
        paused = False
        step = 0
        
        try:
            while (time.time() - start_time) < max_duration:
                if paused:
                    time.sleep(0.1)
                    continue
                
                # Capture image
                display_frame, model_input = self.capture_workspace()
                if model_input is None:
                    continue
                
                # Get action from SmolVLA
                inference_start = time.time()
                result = self.get_action_fn.remote(
                    model_input.tolist(),
                    self.current_positions,
                    task
                )
                inference_time = time.time() - inference_start
                
                if result['success']:
                    action = result['action']
                    self.execute_action(action)
                    step += 1
                    
                    # Display status
                    print(f"Step {step} | Time: {inference_time:.3f}s | "
                          f"Pos: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}...]")
                
                # Show visualization
                if display_frame is not None:
                    self._draw_status(display_frame, task, step, paused)
                    cv2.imshow('SO-101 Control', display_frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.emergency_stop()
                    break
                elif key == ord('p'):
                    paused = True
                    print("⏸️  Paused")
                elif key == ord('r'):
                    paused = False
                    print("▶️  Resumed")
                
                # Control rate
                time.sleep(0.05)  # 20Hz max
                
        except Exception as e:
            print(f"❌ Error: {e}")
            self.emergency_stop()
        finally:
            self.save_session_data()
    
    def _draw_status(self, frame, task, step, paused):
        """Draw status overlay on frame"""
        # Task info
        cv2.putText(frame, f"Task: {task}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Step counter
        cv2.putText(frame, f"Step: {step}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Paused indicator
        if paused:
            cv2.putText(frame, "PAUSED", (500, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        # Joint positions
        pos_text = "Joints: " + " ".join([f"{p:.1f}" for p in self.current_positions[:3]])
        cv2.putText(frame, pos_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Safety indicator
        safety_color = (0, 255, 0) if self.safety_enabled else (0, 0, 255)
        cv2.circle(frame, (600, 50), 20, safety_color, -1)
        cv2.putText(frame, "SAFE" if self.safety_enabled else "UNSAFE", 
                   (550, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, safety_color, 1)
    
    def emergency_stop(self):
        """Emergency stop all motors"""
        print("\n🛑 EMERGENCY STOP!")
        if self.serial_port and not self.simulate:
            # Send stop command - adjust for your protocol
            stop_cmd = bytes([0xFF, 0xFF, 0xFE, 0x02, 0x06, 0xF7])
            self.serial_port.write(stop_cmd)
    
    def save_session_data(self):
        """Save session data for debugging"""
        session_data = {
            'task': 'current_task',
            'duration': time.time(),
            'commands': self.command_history,
            'final_position': self.current_positions
        }
        
        with open(f"{self.log_dir}/session.json", 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"📁 Session saved to {self.log_dir}")
    
    def cleanup(self):
        """Clean shutdown"""
        self.emergency_stop()
        if self.camera:
            self.camera.release()
        if self.serial_port:
            self.serial_port.close()
        cv2.destroyAllWindows()

def main():
    """Main program with menu"""
    print("🤖 SO-101 SmolVLA Controller")
    print("=" * 50)
    
    # Setup options
    print("\n1. Real robot with laptop camera")
    print("2. Real robot with external camera") 
    print("3. Simulation mode (no robot)")
    
    mode = input("\nSelect mode (1-3): ")
    
    if mode == "1":
        controller = SO101PracticalController(camera_index=0, simulate=False)
    elif mode == "2":
        controller = SO101PracticalController(camera_index=1, simulate=False)
    else:
        controller = SO101PracticalController(camera_index=0, simulate=True)
    
    # Task menu
    while True:
        print("\n📋 Tasks:")
        print("1. Pick up red block")
        print("2. Stack blocks")
        print("3. Sort objects by color")
        print("4. Free play (continuous)")
        print("5. Calibrate workspace")
        print("0. Exit")
        
        choice = input("\nSelect task: ")
        
        if choice == "0":
            break
        elif choice == "1":
            controller.run_task("Pick up the red block", max_duration=30)
        elif choice == "2":
            controller.run_task("Stack the blue block on the red block", max_duration=45)
        elif choice == "3":
            controller.run_task("Sort objects by color", max_duration=60)
        elif choice == "4":
            controller.run_task("Interact with objects", max_duration=300)
        elif choice == "5":
            print("Show the workspace to the camera...")
            # Calibration mode
    
    controller.cleanup()

if __name__ == "__main__":
    main()