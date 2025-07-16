#!/usr/bin/env python3
"""
Test 6: Complete SO-101 + Modal SmolVLA Integration
Combines camera, Modal inference, and robot control (real or simulated)
"""

import cv2
import numpy as np
import time
import threading
import queue
import serial
import serial.tools.list_ports
from typing import Optional, List, Tuple
import json
import os
from datetime import datetime

try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    print("⚠️ Modal not available - using dummy inference")

def _calculate_checksum(data: bytearray) -> int:
    """Calculates the Dynamixel checksum."""
    # Sum all bytes except the start bytes (0xFF, 0xFF) and the checksum byte itself
    return (~sum(data[2:])) & 0xFF

class RobotModalIntegration:
    def __init__(self, camera_index=1, simulate_robot=False):
        """
        Complete integration: Camera → Modal → Robot
        
        Args:
            camera_index: Camera to use (1 = Mac camera)
            simulate_robot: Use simulated robot if True
        """
        self.camera_index = camera_index
        self.simulate_robot = simulate_robot
        self.num_joints = 7  # Define the number of joints for the robot
        
        # Performance tracking
        self.stats = {
            'frames_captured': 0,
            'inferences_sent': 0,
            'inferences_completed': 0,
            'actions_executed': 0,
            'start_time': time.time(),
            'inference_times': [],
            'action_times': []
        }
        
        # State
        self.running = False
        self.inference_enabled = True
        self.action_enabled = True
        
        # Queues for async processing
        self.inference_queue = queue.Queue(maxsize=3)
        self.action_queue = queue.Queue(maxsize=10)
        
        # Robot state
        self.current_joint_positions = [0.0] * 7
        self.target_joint_positions = [0.0] * 7
        
        # Setup components
        self.setup_camera()
        self.setup_modal()
        self.setup_robot()
        
        # Logging
        self.log_dir = f"logs/integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        print(f"\n🚀 Robot-Modal Integration Ready!")
        print(f"   Camera: {self.camera_working}")
        print(f"   Modal: {self.modal_working}")
        print(f"   Robot: {self.robot_working} ({'Real' if not self.simulate_robot else 'Simulated'})")
    
    def setup_camera(self):
        """Initialize camera."""
        print(f"📷 Setting up camera {self.camera_index}...")
        
        self.camera = cv2.VideoCapture(self.camera_index)
        
        if not self.camera.isOpened():
            print(f"❌ Camera {self.camera_index} failed to open")
            self.camera_working = False
            return
        
        # Configure camera
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Test capture
        ret, frame = self.camera.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"✅ Camera working: {w}x{h}")
            self.camera_working = True
        else:
            print(f"❌ Camera can't capture frames")
            self.camera_working = False
    
    def setup_modal(self):
        """Initialize Modal connection."""
        print(f"☁️ Setting up Modal connection...")
        
        if not MODAL_AVAILABLE:
            print(f"❌ Modal not available")
            self.modal_working = False
            return
        
        try:
            self.get_action_fn = modal.Function.lookup(
                "smolvla-so101-deployment", 
                "get_action_direct"
            )
            print(f"✅ Modal connection established")
            self.modal_working = True
        except Exception as e:
            print(f"❌ Modal connection failed: {e}")
            self.modal_working = False
    
    def setup_robot(self):
        """Initialize robot connection."""
        if self.simulate_robot:
            print(f"🎮 Using simulated robot")
            self.robot_port = None
            self.robot_working = True
            return
        
        print(f"🤖 Setting up robot connection...")
        
        self._connect_to_robot()
    
    def _connect_to_robot(self):
        """Find and connect to the SO-101 robot via serial."""
        print("\n🤖 Searching for SO-101 robot...")
        
        # Common VIDs/PIDs for robot USB-serial adapters
        # (CH340, FTDI, CP210x, etc.)
        known_devices = {
            (0x1A86, 0x7523): "CH340",
            (0x10C4, 0xEA60): "CP210x",
            (0x0403, 0x6001): "FTDI",
        }

        ports = serial.tools.list_ports.comports()
        found_port = None
        for port in ports:
            if (port.vid, port.pid) in known_devices:
                print(f"  🔎 Found potential device: {known_devices[(port.vid, port.pid)]} on {port.device}")
                found_port = port.device
                break
            if "usbmodem" in port.device or "ttyUSB" in port.device or "ttyACM" in port.device:
                found_port = port.device # Fallback for generic serial

        if not found_port:
            print("  ❌ No robot found. Check connection.")
            self.robot_working = False
            return

        self.robot_port = found_port
        print(f"  🔌 Attempting to connect to {self.robot_port}...")

        try:
            self.serial_conn = serial.Serial(self.robot_port, baudrate=115200, timeout=0.5, write_timeout=0.5)
            self.robot_working = True
            print(f"  ✅ Robot connected successfully on {self.robot_port}")
        except serial.SerialException as e:
            print(f"  ❌ Failed to connect to robot: {e}")
            self.robot_working = False
    
    def _read_joint_positions(self):
        """Read current joint positions from robot."""
        if self.simulate_robot or not self.serial_conn:
            time.sleep(0.01)  # Simulate work
            return self.current_joint_positions

        new_positions = list(self.current_joint_positions) # Start with last known
        try:
            for i in range(self.num_joints):
                servo_id = i + 1
                # Dynamixel Protocol 1.0: READ_DATA command
                # Ask for 2 bytes from address 36 (Present Position)
                cmd = bytearray([0xFF, 0xFF, servo_id, 0x04, 0x02, 36, 2])
                cmd.append(_calculate_checksum(cmd))

                self.serial_conn.reset_input_buffer()
                self.serial_conn.write(cmd)

                # Expect a Status Packet in response: 0xFF, 0xFF, ID, Length, Error, Param1, Param2, Checksum
                response = self.serial_conn.read(8)

                if len(response) == 8 and response[0] == 0xFF and response[1] == 0xFF and response[2] == servo_id:
                    # Basic validation passed, now parse the position
                    # Position is a 16-bit value (little-endian)
                    raw_pos = int.from_bytes(response[5:7], 'little')
                    # Convert from 0-1023 range to -1 to 1 range
                    # This mapping is an assumption and may need tuning
                    normalized_pos = (raw_pos - 512) / 512.0
                    new_positions[i] = normalized_pos
                # else: we missed the packet or got an error, so we'll just use the last known value for this joint

            self.current_joint_positions = new_positions
            return self.current_joint_positions

        except Exception as e:
            print(f"Error reading robot state: {e}")
            return self.current_joint_positions
    
    def preprocess_frame(self, frame):
        """Preprocess frame for SmolVLA."""
        # Center crop to square
        h, w = frame.shape[:2]
        size = min(h, w)
        y_start = (h - size) // 2
        x_start = (w - size) // 2
        square = frame[y_start:y_start+size, x_start:x_start+size]
        
        # Resize to 224x224
        processed = cv2.resize(square, (224, 224))
        
        return processed
    
    def inference_worker(self):
        """Background thread for Modal inference."""
        print("🧠 Inference worker started")
        
        while self.running:
            try:
                # Get frame from queue
                frame_data = self.inference_queue.get(timeout=1.0)
                if frame_data is None:  # Shutdown signal
                    break
                
                frame, timestamp = frame_data
                
                if not self.inference_enabled:
                    continue
                
                # Get current robot state
                robot_state = self.current_joint_positions.copy()
                
                start_time = time.time()
                
                if self.modal_working:
                    try:
                        # Call Modal function
                        result = self.get_action_fn.remote(
                            frame.tolist(),
                            robot_state,
                            "Move to pick up the object"
                        )
                        
                        action = result['action']
                        inference_time = time.time() - start_time
                        
                        # Queue action for execution
                        self.action_queue.put({
                            'action': action,
                            'timestamp': timestamp,
                            'inference_time': inference_time
                        })
                        
                        # Update stats
                        self.stats['inferences_completed'] += 1
                        self.stats['inference_times'].append(inference_time)
                        
                    except Exception as e:
                        print(f"❌ Modal inference error: {e}")
                else:
                    # Dummy inference for testing
                    time.sleep(0.1)  # Simulate processing time
                    dummy_action = [0.01 * np.sin(time.time() + i) for i in range(7)]
                    
                    self.action_queue.put({
                        'action': dummy_action,
                        'timestamp': timestamp,
                        'inference_time': 0.1
                    })
                    
                    self.stats['inferences_completed'] += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Inference worker error: {e}")
        
        print("🧠 Inference worker stopped")
    
    def action_worker(self):
        """Background thread for robot actions."""
        print("🦾 Action worker started")
        
        while self.running:
            try:
                # Get action from queue
                action_data = self.action_queue.get(timeout=1.0)
                if action_data is None:  # Shutdown signal
                    break
                
                if not self.action_enabled:
                    continue
                
                action = action_data['action']
                start_time = time.time()
                
                # Execute action
                success = self._execute_action(action)
                
                if success:
                    action_time = time.time() - start_time
                    self.stats['actions_executed'] += 1
                    self.stats['action_times'].append(action_time)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Action worker error: {e}")
        
        print("🦾 Action worker stopped")
    
    def _execute_action(self, action):
        """Execute robot action."""
        if self.simulate_robot or not self.serial_conn:
            # Simulate action execution
            time.sleep(0.1) # Simulate time to move
            self.current_joint_positions = [(p + (a * 0.1)) for p, a in zip(self.current_joint_positions, action)]
            return

        
        try:
            # Send action to real robot
            # This needs to be customized for SO-101 protocol
            # For now, just update internal state
            self.target_joint_positions = action
            return True
            
        except Exception as e:
            print(f"❌ Action execution error: {e}")
            return False
    
    def create_display_frame(self, frame, processed_frame):
        """Create display frame with overlays."""
        # Resize frames for display
        display_frame = cv2.resize(frame, (640, 480))
        display_processed = cv2.resize(processed_frame, (320, 240))
        
        # Add processed frame overlay
        display_frame[10:250, 10:330] = display_processed
        cv2.rectangle(display_frame, (8, 8), (332, 252), (0, 255, 0), 2)
        
        # Add stats overlay
        runtime = max(time.time() - self.stats['start_time'], 1)
        stats_text = [
            f"FPS: {self.stats['frames_captured'] / runtime:.1f}",
            f"Inferences: {self.stats['inferences_completed']}",
            f"Actions: {self.stats['actions_executed']}",
            f"Inference: {'ON' if self.inference_enabled else 'OFF'}",
            f"Actions: {'ON' if self.action_enabled else 'OFF'}"
        ]
        
        if self.stats['inference_times']:
            avg_inference = np.mean(self.stats['inference_times'][-10:])
            stats_text.append(f"Avg Inference: {avg_inference:.2f}s")
        
        # Current joint positions
        joint_text = "Joints: " + ", ".join([f"{j:.3f}" for j in self.current_joint_positions])
        
        # Draw text
        y = 30
        for text in stats_text:
            cv2.putText(display_frame, text, (350, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25
        
        # Joint positions at bottom
        cv2.putText(display_frame, joint_text, (10, 460), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Instructions
        instructions = [
            "SPACE: Toggle inference",
            "A: Toggle actions", 
            "S: Save frame",
            "R: Reset stats",
            "Q: Quit"
        ]
        
        y = 300
        for instruction in instructions:
            cv2.putText(display_frame, instruction, (350, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y += 20
        
        return display_frame
    
    def run_integration(self):
        """Run the complete integration."""
        if not self.camera_working:
            print("❌ Cannot run without working camera")
            return
        
        print("\n🚀 Starting Robot-Modal Integration...")
        print("Controls:")
        print("  SPACE: Toggle inference")
        print("  A: Toggle actions")
        print("  S: Save frame")
        print("  R: Reset stats")
        print("  Q: Quit")
        
        self.running = True
        
        # Start worker threads
        inference_thread = threading.Thread(target=self.inference_worker)
        action_thread = threading.Thread(target=self.action_worker)
        
        inference_thread.start()
        action_thread.start()
        
        try:
            frame_count = 0
            
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break
                
                self.stats['frames_captured'] += 1
                frame_count += 1
                
                # Preprocess frame
                processed_frame = self.preprocess_frame(frame)
                
                # Send to inference (if queue not full)
                if not self.inference_queue.full():
                    self.inference_queue.put((processed_frame, time.time()))
                    self.stats['inferences_sent'] += 1
                
                # Update robot state
                if frame_count % 30 == 0:  # Every 30 frames
                    self._read_joint_positions()
                
                # Create display
                display_frame = self.create_display_frame(frame, processed_frame)
                
                # Show frame
                cv2.imshow('Robot-Modal Integration', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self.inference_enabled = not self.inference_enabled
                    print(f"Inference: {'ON' if self.inference_enabled else 'OFF'}")
                elif key == ord('a'):
                    self.action_enabled = not self.action_enabled
                    print(f"Actions: {'ON' if self.action_enabled else 'OFF'}")
                elif key == ord('s'):
                    filename = f"{self.log_dir}/frame_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"💾 Saved: {filename}")
                elif key == ord('r'):
                    self._reset_stats()
                    print("📊 Stats reset")
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        
        finally:
            print("\n🛑 Shutting down...")
            self.running = False
            
            # Signal workers to stop
            self.inference_queue.put(None)
            self.action_queue.put(None)
            
            # Wait for threads
            inference_thread.join(timeout=2)
            action_thread.join(timeout=2)
            
            # Cleanup
            self.camera.release()
            cv2.destroyAllWindows()
            
            if self.robot_port:
                self.robot_port.close()
            
            self._print_final_stats()
    
    def _reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            'frames_captured': 0,
            'inferences_sent': 0,
            'inferences_completed': 0,
            'actions_executed': 0,
            'start_time': time.time(),
            'inference_times': [],
            'action_times': []
        }
    
    def _print_final_stats(self):
        """Print final performance statistics."""
        runtime = time.time() - self.stats['start_time']
        
        print("\n📊 Final Performance Stats:")
        print(f"  Runtime: {runtime:.1f}s")
        print(f"  Frames captured: {self.stats['frames_captured']}")
        print(f"  Camera FPS: {self.stats['frames_captured'] / runtime:.1f}")
        print(f"  Inferences sent: {self.stats['inferences_sent']}")
        print(f"  Inferences completed: {self.stats['inferences_completed']}")
        print(f"  Actions executed: {self.stats['actions_executed']}")
        
        if self.stats['inference_times']:
            avg_inference = np.mean(self.stats['inference_times'])
            print(f"  Avg inference time: {avg_inference:.2f}s")
            print(f"  Inference FPS: {1/avg_inference:.2f}")
        
        if self.stats['action_times']:
            avg_action = np.mean(self.stats['action_times'])
            print(f"  Avg action time: {avg_action:.4f}s")

def main():
    print("🤖 SO-101 + Modal SmolVLA Integration Test")
    print("=" * 50)
    
    # Configuration
    camera_index = 1  # Mac camera
    simulate_robot = False  # Set to True to try simulation
    
    # Create integration
    integration = RobotModalIntegration(
        camera_index=camera_index,
        simulate_robot=simulate_robot
    )
    
    # Run integration
    integration.run_integration()
    
    print("\n✅ Integration test complete!")
    print("\n🐧 Next step: Test on Linux with real robot hardware")

if __name__ == "__main__":
    main()
