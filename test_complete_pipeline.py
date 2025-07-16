# test_complete_pipeline.py
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation
import modal
import cv2
import numpy as np
import time
import json

class RobotVisionPipeline:
    def __init__(self, camera_index=0):
        # Setup camera with LeRobot
        self.camera_config = OpenCVCameraConfig(
            index_or_path=camera_index,
            fps=30,
            width=1280,
            height=720,
            color_mode=ColorMode.RGB,
            rotation=Cv2Rotation.NO_ROTATION
        )
        
        self.camera = OpenCVCamera(self.camera_config)
        self.camera.connect()
        
        # Load calibration if exists
        try:
            with open('camera_calibration.json', 'r') as f:
                self.calibration = json.load(f)
                self.workspace_corners = self.calibration['workspace_corners']
                print("✅ Loaded calibration")
        except:
            self.workspace_corners = None
            print("⚠️ No calibration found")
        
        # Get Modal function
        self.modal_fn = modal.Function.lookup(
            "smolvla-so101-deployment", 
            "get_action_direct"
        )
        
        print("✅ Pipeline initialized")
    
    def extract_workspace(self, frame):
        """Extract and transform workspace area"""
        if self.workspace_corners is None:
            # No calibration, use center crop
            h, w = frame.shape[:2]
            return frame[h//4:3*h//4, w//4:3*w//4]
        
        # Use perspective transform
        pts1 = np.float32(self.workspace_corners)
        pts2 = np.float32([[0, 0], [224, 0], [224, 224], [0, 224]])
        
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        workspace = cv2.warpPerspective(frame, matrix, (224, 224))
        
        return workspace
    
    def process_frame(self, task="Pick up the object"):
        """Complete processing pipeline"""
        # 1. Capture
        frame = self.camera.async_read(timeout_ms=100)
        if frame is None:
            return None, None, None
        
        # 2. Extract workspace
        workspace = self.extract_workspace(frame)
        
        # 3. Prepare for model (already 224x224 from extraction)
        model_input = workspace
        
        # 4. Get robot state (dummy for now)
        robot_state = [0.0] * 7
        
        # 5. Call Modal
        start = time.time()
        result = self.modal_fn.remote(
            model_input.tolist(),
            robot_state,
            task
        )
        inference_time = time.time() - start
        
        if result['success']:
            action = result['action']
            return frame, action, inference_time
        
        return frame, None, inference_time
    
    def visualize_pipeline(self):
        """Show all pipeline stages"""
        print("\n👁️ Pipeline Visualization")
        print("Press 'q' to quit")
        
        while True:
            frame, action, inference_time = self.process_frame()
            
            if frame is None:
                continue
            
            # Create visualization
            h, w = frame.shape[:2]
            viz = np.zeros((h, w*2 + 20, 3), dtype=np.uint8)
            
            # Original frame
            viz[:, :w] = frame
            cv2.putText(viz, "Camera View", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Workspace view
            workspace = self.extract_workspace(frame)
            workspace_large = cv2.resize(workspace, (w, h))
            viz[:, w+20:] = workspace_large
            cv2.putText(viz, "Workspace (224x224)", (w+30, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show action
            if action:
                action_text = f"Action: [{action[0]:.2f}, {action[1]:.2f}, ...]"
                cv2.putText(viz, action_text, (10, h-60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                time_text = f"Inference: {inference_time*1000:.0f}ms"
                cv2.putText(viz, time_text, (10, h-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Pipeline', viz)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def benchmark(self, num_frames=50):
        """Benchmark the pipeline"""
        print(f"\n⏱️ Benchmarking {num_frames} frames...")
        
        times = {
            'capture': [],
            'process': [],
            'inference': [],
            'total': []
        }
        
        for i in range(num_frames):
            total_start = time.time()
            
            # Capture
            t1 = time.time()
            frame = self.camera.async_read(timeout_ms=100)
            times['capture'].append(time.time() - t1)
            
            if frame is None:
                continue
            
            # Process
            t2 = time.time()
            workspace = self.extract_workspace(frame)
            times['process'].append(time.time() - t2)
            
            # Inference
            t3 = time.time()
            result = self.modal_fn.remote(
                workspace.tolist(),
                [0.0] * 7,
                "Test"
            )
            times['inference'].append(time.time() - t3)
            
            times['total'].append(time.time() - total_start)
            
            if (i+1) % 10 == 0:
                print(f"Progress: {i+1}/{num_frames}")
        
        # Report
        print("\n📊 Benchmark Results:")
        for stage, measurements in times.items():
            if measurements:
                avg = np.mean(measurements) * 1000
                std = np.std(measurements) * 1000
                print(f"{stage:10s}: {avg:6.1f} ± {std:4.1f} ms")
        
        print(f"\n🎯 Max FPS possible: {1000/np.mean(times['total'])/1000:.1f}")
    
    def cleanup(self):
        self.camera.disconnect()

if __name__ == "__main__":
    print("🚀 Complete Pipeline Test")
    
    # Find best camera
    import subprocess
    result = subprocess.run(
        ["python", "-m", "lerobot.find_cameras", "opencv"],
        capture_output=True,
        text=True
    )
    print(result.stdout[:500])  # Show available cameras
    
    camera_idx = int(input("\nEnter camera index: "))
    
    pipeline = RobotVisionPipeline(camera_idx)
    
    print("\n1. Visualize pipeline")
    print("2. Benchmark performance")
    print("3. Both")
    
    choice = input("\nSelect: ")
    
    try:
        if choice == "1":
            pipeline.visualize_pipeline()
        elif choice == "2":
            pipeline.benchmark()
        else:
            pipeline.visualize_pipeline()
            pipeline.benchmark()
    finally:
        pipeline.cleanup()