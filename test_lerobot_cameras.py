# test_lerobot_cameras.py
import subprocess
import cv2

def find_cameras_lerobot_way():
    """Use LeRobot's built-in camera finder"""
    print("🔍 Finding cameras using LeRobot...")
    
    # Run LeRobot's camera finder
    result = subprocess.run(
        ["python", "-m", "lerobot.find_cameras", "opencv"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    
    if "iPhone" in result.stdout:
        print("✅ iPhone detected via Continuity Camera!")
    
    return result.stdout

def test_camera_direct():
    """Quick test without LeRobot"""
    print("\n🎥 Quick camera test...")
    
    # Try different camera indices
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera {i} works: {frame.shape}")
                # Show quick preview
                cv2.imshow(f'Camera {i}', cv2.resize(frame, (320, 240)))
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
            cap.release()
        else:
            print(f"❌ Camera {i} not available")

if __name__ == "__main__":
    find_cameras_lerobot_way()
    test_camera_direct()