# setup_lerobot_camera.py
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation
import cv2
import numpy as np
import time

def setup_iphone_camera():
    """Setup iPhone via Continuity Camera"""
    print("📱 Setting up iPhone camera...")
    
    # iPhone usually shows up as index 1 or 2 when connected
    # Try different indices
    for idx in [0, 1, 2]:
        try:
            config = OpenCVCameraConfig(
                index_or_path=idx,
                fps=30,  # iPhone can do 30fps
                width=1280,  # Good balance of quality/speed
                height=720,
                color_mode=ColorMode.RGB,
                rotation=Cv2Rotation.NO_ROTATION
            )
            
            camera = OpenCVCamera(config)
            camera.connect()
            
            # Test read
            frame = camera.async_read(timeout_ms=1000)
            if frame is not None:
                print(f"✅ iPhone camera connected at index {idx}")
                print(f"   Frame shape: {frame.shape}")
                return camera, idx
                
        except Exception as e:
            print(f"Camera {idx} failed: {e}")
            
    return None, -1

def setup_laptop_camera():
    """Setup built-in laptop camera"""
    print("💻 Setting up laptop camera...")
    
    config = OpenCVCameraConfig(
        index_or_path=0,  # Usually 0 for built-in
        fps=15,  # Lower FPS for laptop cameras
        width=640,
        height=480,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION
    )
    
    try:
        camera = OpenCVCamera(config)
        camera.connect()
        print("✅ Laptop camera connected")
        return camera
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None

def calibrate_workspace_view(camera):
    """Interactive workspace calibration"""
    print("\n📐 Workspace Calibration")
    print("Position camera to see:")
    print("1. Robot base in bottom center")
    print("2. Full workspace area")
    print("3. Good lighting, minimal shadows")
    print("\nPress 's' to save calibration")
    
    workspace_corners = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            workspace_corners.append((x, y))
            print(f"Corner {len(workspace_corners)}: ({x}, {y})")
    
    cv2.namedWindow('Calibration')
    cv2.setMouseCallback('Calibration', mouse_callback)
    
    while True:
        frame = camera.async_read(timeout_ms=100)
        if frame is None:
            continue
            
        display = frame.copy()
        
        # Draw helper grid
        h, w = frame.shape[:2]
        # Thirds
        cv2.line(display, (w//3, 0), (w//3, h), (128, 128, 128), 1)
        cv2.line(display, (2*w//3, 0), (2*w//3, h), (128, 128, 128), 1)
        cv2.line(display, (0, h//3), (w, h//3), (128, 128, 128), 1)
        cv2.line(display, (0, 2*h//3), (w, 2*h//3), (128, 128, 128), 1)
        
        # Draw workspace corners if clicked
        for i, corner in enumerate(workspace_corners):
            cv2.circle(display, corner, 5, (0, 255, 0), -1)
            cv2.putText(display, f"{i+1}", (corner[0]+10, corner[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Connect corners
        if len(workspace_corners) >= 4:
            pts = np.array(workspace_corners[:4], np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)
            cv2.putText(display, "Workspace defined! Press 's' to save", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, f"Click {4-len(workspace_corners)} corners of workspace", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('Calibration', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and len(workspace_corners) >= 4:
            # Save calibration
            calibration = {
                'camera_resolution': (w, h),
                'workspace_corners': workspace_corners[:4],
                'timestamp': time.time()
            }
            
            import json
            with open('camera_calibration.json', 'w') as f:
                json.dump(calibration, f, indent=2)
            
            print("✅ Calibration saved!")
            break
        elif key == ord('q'):
            break
        elif key == ord('r'):
            workspace_corners = []
            print("Reset corners")
    
    cv2.destroyAllWindows()
    return workspace_corners[:4] if len(workspace_corners) >= 4 else None

if __name__ == "__main__":
    print("🎥 LeRobot Camera Setup")
    print("1. iPhone (Continuity Camera)")
    print("2. Laptop built-in camera")
    print("3. Auto-detect")
    
    choice = input("\nSelect camera: ")
    
    camera = None
    if choice == "1":
        camera, idx = setup_iphone_camera()
    elif choice == "2":
        camera = setup_laptop_camera()
    else:
        # Try iPhone first, then laptop
        camera, idx = setup_iphone_camera()
        if camera is None:
            camera = setup_laptop_camera()
    
    if camera:
        # Calibrate workspace
        workspace = calibrate_workspace_view(camera)
        
        # Test continuous capture
        print("\n📹 Testing continuous capture (press 'q' to quit)")
        fps_times = []
        
        try:
            while True:
                start = time.time()
                frame = camera.async_read(timeout_ms=100)
                
                if frame is not None:
                    # Calculate FPS
                    fps_times.append(time.time() - start)
                    if len(fps_times) > 30:
                        fps_times.pop(0)
                    fps = 1.0 / np.mean(fps_times) if fps_times else 0
                    
                    # Display
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('Camera Feed', frame)
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            camera.disconnect()
            cv2.destroyAllWindows()