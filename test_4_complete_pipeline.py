#!/usr/bin/env python3
"""
Test 4: Complete Pipeline
Combines Mac camera with Modal SmolVLA inference for real-time robot vision
"""

import cv2
import numpy as np
import time
import modal
import threading
import queue

class RobotVisionPipeline:
    def __init__(self, camera_index=1):
        self.camera_index = camera_index
        self.cap = None
        self.modal_func = None
        self.running = False
        
        # Frame processing
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)
        
        # Stats
        self.stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'total_inference_time': 0,
            'start_time': time.time()
        }
        
    def connect_camera(self):
        """Connect to camera."""
        print(f"📹 Connecting to camera {self.camera_index}...")
        
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise Exception(f"Could not open camera {self.camera_index}")
        
        # Get camera properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Camera connected: {width}x{height} @ {fps:.1f}fps")
        return True
    
    def connect_modal(self):
        """Connect to Modal inference function."""
        print("🔌 Connecting to Modal...")
        
        try:
            self.modal_func = modal.Function.from_name(
                "smolvla-so101-deployment", 
                "get_action_direct"
            )
            print("✅ Modal connected!")
            
            # Test inference
            print("🧪 Testing Modal inference...")
            dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            dummy_state = [0.0] * 7
            
            start = time.time()
            result = self.modal_func.remote(
                dummy_image.tolist(),
                dummy_state,
                "Test inference"
            )
            test_time = time.time() - start
            
            print(f"✅ Modal test successful! Time: {test_time:.2f}s")
            return True
            
        except Exception as e:
            print(f"❌ Modal connection failed: {e}")
            return False
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model input."""
        # Get center square crop
        h, w = frame.shape[:2]
        size = min(h, w)
        y = (h - size) // 2
        x = (w - size) // 2
        square = frame[y:y+size, x:x+size]
        
        # Resize to model input size
        processed = cv2.resize(square, (224, 224))
        
        return processed, (x, y, size)
    
    def inference_worker(self):
        """Worker thread for Modal inference."""
        print("🧠 Inference worker started")
        
        while self.running:
            try:
                # Get frame from queue (timeout to check running status)
                frame_data = self.frame_queue.get(timeout=1.0)
                
                if frame_data is None:  # Poison pill
                    break
                
                frame, timestamp = frame_data
                
                # Preprocess
                processed_frame, crop_info = self.preprocess_frame(frame)
                
                # Run inference
                start_time = time.time()
                
                result = self.modal_func.remote(
                    processed_frame.tolist(),
                    [0.0] * 7,  # Robot state (dummy for now)
                    "Pick up the object"
                )
                
                inference_time = time.time() - start_time
                
                # Update stats
                self.stats['frames_processed'] += 1
                self.stats['total_inference_time'] += inference_time
                
                # Store result
                result_data = {
                    'result': result,
                    'inference_time': inference_time,
                    'timestamp': timestamp,
                    'crop_info': crop_info
                }
                
                # Add to result queue (non-blocking)
                try:
                    self.result_queue.put_nowait(result_data)
                except queue.Full:
                    # Drop oldest result if queue is full
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(result_data)
                    except queue.Empty:
                        pass
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Inference error: {e}")
                continue
        
        print("🧠 Inference worker stopped")
    
    def run_pipeline(self):
        """Run the complete vision pipeline."""
        print("🚀 Starting complete robot vision pipeline")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print("  'p' - Pause/Resume inference")
        print("  'r' - Reset stats")
        
        self.running = True
        
        # Start inference worker thread
        inference_thread = threading.Thread(target=self.inference_worker)
        inference_thread.start()
        
        # Main display loop
        last_result = None
        inference_paused = False
        
        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break
                
                self.stats['frames_captured'] += 1
                timestamp = time.time()
                
                # Add frame to inference queue (if not paused)
                if not inference_paused and not self.frame_queue.full():
                    try:
                        self.frame_queue.put_nowait((frame.copy(), timestamp))
                    except queue.Full:
                        pass  # Skip if queue is full
                
                # Get latest inference result
                try:
                    while True:
                        last_result = self.result_queue.get_nowait()
                except queue.Empty:
                    pass  # Keep using last result
                
                # Create display frame
                display_frame = self.create_display_frame(frame, last_result, inference_paused)
                
                # Show frame
                cv2.imshow('Robot Vision Pipeline', display_frame)
                
                # Handle keypresses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"robot_vision_{int(timestamp)}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Saved {filename}")
                elif key == ord('p'):
                    inference_paused = not inference_paused
                    status = "PAUSED" if inference_paused else "RESUMED"
                    print(f"⏸️ Inference {status}")
                elif key == ord('r'):
                    self.reset_stats()
                    print("🔄 Stats reset")
        
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
        
        finally:
            self.running = False
            
            # Stop inference worker
            self.frame_queue.put(None)  # Poison pill
            inference_thread.join(timeout=5)
            
            # Cleanup
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
            # Final stats
            self.print_stats()
    
    def create_display_frame(self, frame, result_data, inference_paused):
        """Create the display frame with overlays."""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Calculate stats
        elapsed = time.time() - self.stats['start_time']
        capture_fps = self.stats['frames_captured'] / elapsed if elapsed > 0 else 0
        
        if self.stats['frames_processed'] > 0:
            avg_inference_time = self.stats['total_inference_time'] / self.stats['frames_processed']
            inference_fps = 1 / avg_inference_time if avg_inference_time > 0 else 0
        else:
            avg_inference_time = 0
            inference_fps = 0
        
        # Status overlay
        status = "PAUSED" if inference_paused else "LIVE"
        cv2.putText(display, f"Status: {status}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Performance overlay
        cv2.putText(display, f"Capture: {capture_fps:.1f} FPS", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f"Inference: {inference_fps:.1f} FPS", 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f"Avg Time: {avg_inference_time*1000:.0f}ms", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Inference result overlay
        if result_data and result_data['result']['success']:
            action = result_data['result']['action']
            
            # Draw crop area
            x, y, size = result_data['crop_info']
            cv2.rectangle(display, (x, y), (x+size, y+size), (0, 255, 0), 2)
            cv2.putText(display, "Model Input", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Action display
            action_text = f"Action: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]"
            cv2.putText(display, action_text, 
                       (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Age of result
            age = time.time() - result_data['timestamp']
            cv2.putText(display, f"Result age: {age:.1f}s", 
                       (10, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls
        controls = ["q:Quit", "s:Save", "p:Pause", "r:Reset"]
        for i, control in enumerate(controls):
            cv2.putText(display, control, 
                       (w-150, 30+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'total_inference_time': 0,
            'start_time': time.time()
        }
    
    def print_stats(self):
        """Print final statistics."""
        elapsed = time.time() - self.stats['start_time']
        
        print(f"\n📊 Final Statistics:")
        print(f"   Runtime: {elapsed:.1f}s")
        print(f"   Frames captured: {self.stats['frames_captured']}")
        print(f"   Frames processed: {self.stats['frames_processed']}")
        print(f"   Capture FPS: {self.stats['frames_captured']/elapsed:.1f}")
        
        if self.stats['frames_processed'] > 0:
            avg_time = self.stats['total_inference_time'] / self.stats['frames_processed']
            print(f"   Avg inference time: {avg_time:.2f}s")
            print(f"   Inference FPS: {1/avg_time:.1f}")
        
        print(f"   Processing efficiency: {self.stats['frames_processed']/self.stats['frames_captured']*100:.1f}%")

def main():
    print("🚀 Complete Robot Vision Pipeline")
    print("=" * 50)
    
    pipeline = RobotVisionPipeline(camera_index=1)
    
    try:
        # Connect camera
        if not pipeline.connect_camera():
            return
        
        # Connect Modal
        if not pipeline.connect_modal():
            return
        
        # Run pipeline
        pipeline.run_pipeline()
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        return

if __name__ == "__main__":
    main()
