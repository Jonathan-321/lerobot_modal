# async_robot_controller.py
import asyncio
import modal
import numpy as np
import time
import serial
from collections import deque
import json

class AsyncRobotController:
    def __init__(self):
        # Load calibration
        try:
            with open("so101_calibration.json", "r") as f:
                self.calibration = json.load(f)
        except:
            print("⚠️ No calibration found, using defaults")
            self.calibration = None
        
        # Modal function
        self.predict = modal.Function.lookup("smolvla-simple", "predict_action")
        
        # Robot connection
        try:
            self.robot = serial.Serial('/dev/ttyACM0', 1000000, timeout=0.01)
            print("✅ Robot connected")
        except:
            self.robot = None
            print("❌ Robot not connected")
        
        # State
        self.state = [0.0] * 7
        self.action_queue = deque(maxlen=5)  # Buffer actions
        self.inference_running = False
    
    async def inference_loop(self, task):
        """Continuously run inference in background"""
        while self.inference_running:
            try:
                # Get dummy image (or real camera if available)
                image = np.zeros((224, 224, 3), dtype=np.uint8)
                
                # Async inference call
                start = time.time()
                action = await asyncio.to_thread(
                    self.predict.remote,
                    image.tolist(),
                    self.state,
                    task
                )
                inference_time = time.time() - start
                
                # Add to queue
                self.action_queue.append(action)
                print(f"Inference: {inference_time:.2f}s, Queue: {len(self.action_queue)}")
                
                # Don't overwhelm the system
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"Inference error: {e}")
                await asyncio.sleep(1)
    
    async def control_loop(self):
        """Execute actions from queue"""
        while self.inference_running:
            if self.action_queue:
                action = self.action_queue.popleft()
                self.execute_action(action)
                self.state = action
            
            await asyncio.sleep(0.05)  # 20Hz control
    
    def execute_action(self, actions):
        """Send calibrated actions to robot"""
        if not self.robot:
            print(f"[SIM] Actions: {[f'{a:.2f}' for a in actions[:3]]}...")
            return
        
        for i, action in enumerate(actions[:7]):
            servo_id = i + 1
            
            # Use calibration if available
            if self.calibration and f"servo_{servo_id}" in self.calibration:
                cal = self.calibration[f"servo_{servo_id}"]
                # Map [-1, 1] to calibrated range
                position = int(cal["center"] + action * (cal["max"] - cal["min"]) / 2)
                position = max(cal["min"], min(cal["max"], position))
            else:
                # Default mapping
                position = int(2000 + action * 1000)
                position = max(1000, min(3000, position))
            
            # Send command
            cmd = bytearray([
                0xFF, 0xFF, servo_id, 0x05, 0x03, 0x1E,
                position & 0xFF, (position >> 8) & 0xFF
            ])
            checksum = (~sum(cmd[2:])) & 0xFF
            cmd.append(checksum)
            
            self.robot.write(cmd)
            time.sleep(0.002)  # Faster communication
    
    async def run(self, task, duration=30):
        """Main async execution"""
        print(f"\n🤖 Task: {task}")
        print(f"Running for {duration} seconds...\n")
        
        self.inference_running = True
        
        # Start both loops
        inference_task = asyncio.create_task(self.inference_loop(task))
        control_task = asyncio.create_task(self.control_loop())
        
        # Run for specified duration
        await asyncio.sleep(duration)
        
        # Stop
        self.inference_running = False
        await inference_task
        await control_task
        
        # Cleanup
        if self.robot:
            for i in range(1, 8):
                cmd = bytearray([0xFF, 0xFF, i, 0x04, 0x03, 0x18, 0x00])
                checksum = (~sum(cmd[2:])) & 0xFF
                cmd.append(checksum)
                self.robot.write(cmd)
            self.robot.close()

async def main():
    controller = AsyncRobotController()
    
    # Menu
    tasks = {
        "1": "Open the gripper",
        "2": "Close the gripper",
        "3": "Pick up the red block",
        "4": "Move to home position"
    }
    
    print("Tasks:")
    for k, v in tasks.items():
        print(f"{k}. {v}")
    
    choice = input("\nChoice: ")
    task = tasks.get(choice, "Open the gripper")
    
    await controller.run(task, duration=20)

if __name__ == "__main__":
    asyncio.run(main())
    