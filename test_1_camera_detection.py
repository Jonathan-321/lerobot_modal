#!/usr/bin/env python3
"""
Test 1: Enhanced Camera Detection
Tests all available cameras and shows live preview
"""

import cv2
import time

def test_all_cameras():
    """Test all camera indices and show live preview."""
    print("🎥 Testing all cameras with live preview...")
    
    working_cameras = []
    
    for i in range(5):
        print(f"\n🔍 Testing Camera {i}...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            # Get camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ Camera {i} works: {width}x{height} @ {fps:.1f}fps")
                working_cameras.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'frame_shape': frame.shape
                })
                
                # Show preview for 3 seconds
                print(f"📺 Showing Camera {i} preview for 3 seconds...")
                start_time = time.time()
                
                while time.time() - start_time < 3:
                    ret, frame = cap.read()
                    if ret:
                        # Add camera info overlay
                        cv2.putText(frame, f"Camera {i}: {width}x{height}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"FPS: {fps:.1f}", 
                                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, "Press 'q' to skip", 
                                  (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        cv2.imshow(f'Camera {i} Preview', frame)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                    else:
                        break
                
                cv2.destroyAllWindows()
                
                # Save test image
                test_filename = f"test_camera_{i}.jpg"
                cv2.imwrite(test_filename, frame)
                print(f"📸 Test image saved as {test_filename}")
                
            else:
                print(f"❌ Camera {i} opened but can't read frames")
        else:
            print(f"❌ Camera {i} not available")
        
        cap.release()
    
    return working_cameras

def main():
    print("🎥 Enhanced Camera Detection Test")
    print("=" * 50)
    
    cameras = test_all_cameras()
    
    if cameras:
        print(f"\n✅ Found {len(cameras)} working camera(s):")
        print("-" * 50)
        for cam in cameras:
            print(f"Camera {cam['index']}: {cam['width']}x{cam['height']} @ {cam['fps']:.1f}fps")
        
        # Recommend best camera for lerobot
        laptop_camera = next((cam for cam in cameras if cam['index'] == 0), None)
        if laptop_camera:
            print(f"\n🎯 Recommended for LeRobot: Camera 0 (Laptop)")
            print(f"   Resolution: {laptop_camera['width']}x{laptop_camera['height']}")
            print(f"   FPS: {laptop_camera['fps']:.1f}")
        
    else:
        print("\n❌ No working cameras found!")
    
    print(f"\n📁 Test images saved in current directory")

if __name__ == "__main__":
    main()
