#!/usr/bin/env python3
"""
Test 2: Live Camera Feed for LeRobot
Tests camera with proper preprocessing for robot vision
"""

import cv2
import numpy as np
import time

def test_camera_live(camera_index=1):
    """Test camera with live feed and preprocessing."""
    print(f"📹 Testing Camera {camera_index} with live feed...")
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Could not open camera {camera_index}")
        return False
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✅ Camera {camera_index}: {width}x{height} @ {fps:.1f}fps")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("  'r' - Show raw vs processed")
    print("  SPACE - Pause/Resume")
    
    frame_count = 0
    start_time = time.time()
    show_processed = True
    paused = False
    current_fps = 0  # Initialize fps variable
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                frame_count += 1
            
            # Calculate FPS
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Create processed version (for robot vision)
            processed_frame = preprocess_for_robot(frame.copy())
            
            # Choose what to display
            display_frame = processed_frame if show_processed else frame
            
            # Add info overlay
            status = "PAUSED" if paused else "LIVE"
            cv2.putText(display_frame, f"Camera {camera_index} - {status}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"FPS: {current_fps:.1f}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Mode: {'Processed' if show_processed else 'Raw'}", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # Show controls
            controls = ["q:Quit", "s:Save", "r:Raw/Proc", "SPACE:Pause"]
            for i, control in enumerate(controls):
                cv2.putText(display_frame, control, 
                           (10, height - 30 - i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Camera Feed', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                filename = f"camera_{camera_index}_frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Saved {filename}")
            elif key == ord('r'):
                show_processed = not show_processed
                print(f"🔄 Switched to {'Processed' if show_processed else 'Raw'} view")
            elif key == ord(' '):  # Space key
                paused = not paused
                print(f"⏸️ {'Paused' if paused else 'Resumed'}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 Statistics:")
        print(f"   Frames processed: {frame_count}")
        print(f"   Average FPS: {current_fps:.1f}")
        print(f"   Total time: {elapsed:.1f}s")
    
    return True

def preprocess_for_robot(frame):
    """Preprocess frame for robot vision (similar to what Modal will receive)."""
    # Get center square crop (common for robot vision)
    h, w = frame.shape[:2]
    size = min(h, w)
    y = (h - size) // 2
    x = (w - size) // 2
    square = frame[y:y+size, x:x+size]
    
    # Resize to model input size (224x224 is common)
    processed = cv2.resize(square, (224, 224))
    
    # Draw crop area on original frame
    cv2.rectangle(frame, (x, y), (x+size, y+size), (0, 255, 0), 3)
    cv2.putText(frame, "Crop Area", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Create side-by-side display
    # Resize original to match processed height for display
    display_orig = cv2.resize(frame, (224, 224))
    
    # Combine frames side by side
    combined = np.hstack([display_orig, processed])
    
    # Add labels
    cv2.putText(combined, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, "Processed", (234, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return combined

def main():
    print("📹 Live Camera Feed Test")
    print("=" * 40)
    
    # Test with Mac camera (index 1) first
    print("Testing with Mac camera (index 1)...")
    success = test_camera_live(1)
    
    if not success:
        print("\n🔄 Trying other camera indices...")
        for i in [0, 2]:
            if test_camera_live(i):
                break

if __name__ == "__main__":
    main()
