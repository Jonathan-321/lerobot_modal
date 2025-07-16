#!/usr/bin/env python3
"""
Standalone camera detection script for lerobot setup.
Works without requiring lerobot installation locally.
"""

import cv2
import subprocess
import sys
import platform


def find_opencv_cameras():
    """Find available cameras using OpenCV."""
    print("🔍 Searching for cameras using OpenCV...")
    
    available_cameras = []
    
    # Test camera indices 0-10 (usually sufficient)
    for index in range(11):
        try:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                # Try to read a frame to verify camera works
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Get camera properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    camera_info = {
                        'index': index,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'backend': 'OpenCV'
                    }
                    available_cameras.append(camera_info)
                    print(f"✅ Camera {index}: {width}x{height} @ {fps:.1f}fps")
                
                cap.release()
            
        except Exception as e:
            # Camera index not available or error occurred
            continue
    
    return available_cameras


def find_system_cameras():
    """Find cameras using system-specific commands."""
    cameras = []
    
    if platform.system() == "Darwin":  # macOS
        try:
            # Use system_profiler to find cameras
            result = subprocess.run([
                'system_profiler', 'SPCameraDataType'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout:
                print("\n📱 System cameras detected:")
                print(result.stdout)
        except Exception as e:
            print(f"Could not query system cameras: {e}")
    
    elif platform.system() == "Linux":
        try:
            # Check /dev/video* devices
            result = subprocess.run([
                'ls', '/dev/video*'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                devices = result.stdout.strip().split('\n')
                print(f"\n📹 Found video devices: {devices}")
                
                # Get more info with v4l2-ctl if available
                try:
                    subprocess.run(['v4l2-ctl', '--version'], 
                                 capture_output=True, check=True)
                    
                    for device in devices:
                        if device.strip():
                            info_result = subprocess.run([
                                'v4l2-ctl', '--device', device, '--info'
                            ], capture_output=True, text=True)
                            
                            if info_result.returncode == 0:
                                print(f"\n{device} info:")
                                print(info_result.stdout)
                                
                except subprocess.CalledProcessError:
                    print("v4l2-ctl not available for detailed camera info")
                    
        except Exception as e:
            print(f"Could not query Linux video devices: {e}")


def test_camera_capture(camera_index):
    """Test camera capture and display a frame."""
    print(f"\n🧪 Testing camera {camera_index}...")
    
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ Could not open camera {camera_index}")
            return False
            
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"❌ Could not read frame from camera {camera_index}")
            cap.release()
            return False
            
        height, width = frame.shape[:2]
        print(f"✅ Successfully captured {width}x{height} frame")
        
        # Save test image
        test_filename = f"camera_{camera_index}_test.jpg"
        cv2.imwrite(test_filename, frame)
        print(f"📸 Test image saved as {test_filename}")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Error testing camera {camera_index}: {e}")
        return False


def main():
    print("🎥 Camera Detection for LeRobot Setup")
    print("=" * 40)
    
    # Check if OpenCV is available
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not installed. Install with: pip install opencv-python")
        return
    
    # Find cameras
    cameras = find_opencv_cameras()
    
    if not cameras:
        print("\n❌ No cameras found with OpenCV")
    else:
        print(f"\n✅ Found {len(cameras)} camera(s)")
        print("\nCamera Summary:")
        print("-" * 50)
        for cam in cameras:
            print(f"Camera {cam['index']}: {cam['width']}x{cam['height']} @ {cam['fps']:.1f}fps")
    
    # System-level camera detection
    find_system_cameras()
    
    # Test first camera if available
    if cameras:
        first_camera = cameras[0]
        test_camera_capture(first_camera['index'])
        
        print(f"\n🔗 For LeRobot, use camera index: {first_camera['index']}")
        print("You can use this in your robot configuration like:")
        print(f"  cameras:")
        print(f"    laptop:")
        print(f"      _target_: lerobot.common.robot.devices.cameras.opencv.OpenCVCamera")
        print(f"      camera_index: {first_camera['index']}")
        print(f"      fps: 30")
        print(f"      width: {first_camera['width']}")
        print(f"      height: {first_camera['height']}")


if __name__ == "__main__":
    main()
