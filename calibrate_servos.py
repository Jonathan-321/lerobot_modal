# calibrate_servos.py
import serial
import time
import json
import os

def calibrate_so101():
    """Calibrate SO-101 servos to find their ranges"""
    port = serial.Serial('/dev/ttyACM0', 1000000, timeout=0.1)
    
    calibration = {}
    
    print("🔧 SO-101 Calibration")
    print("Move each joint to its limits when prompted")
    print("Press ENTER when ready at each position\n")
    
    for servo_id in range(1, 8):
        print(f"\n--- Servo {servo_id} ---")
        
        # Enable torque
        cmd = bytearray([0xFF, 0xFF, servo_id, 0x04, 0x03, 0x18, 0x01])
        checksum = (~sum(cmd[2:])) & 0xFF
        cmd.append(checksum)
        port.write(cmd)
        time.sleep(0.1)
        
        # Get min position
        input(f"Move servo {servo_id} to MINIMUM position and press ENTER: ")
        
        # Read current position
        cmd = bytearray([0xFF, 0xFF, servo_id, 0x04, 0x02, 0x1E, 0x02])
        checksum = (~sum(cmd[2:])) & 0xFF
        cmd.append(checksum)
        port.write(cmd)
        time.sleep(0.1)
        
        if port.in_waiting >= 8:
            response = port.read(8)
            min_pos = (response[6] << 8) | response[5]
        else:
            min_pos = 1000  # default
        
        # Get max position
        input(f"Move servo {servo_id} to MAXIMUM position and press ENTER: ")
        
        # Read current position again
        port.write(cmd)
        time.sleep(0.1)
        
        if port.in_waiting >= 8:
            response = port.read(8)
            max_pos = (response[6] << 8) | response[5]
        else:
            max_pos = 3000  # default
        
        # Get center position
        center_pos = (min_pos + max_pos) // 2
        
        # Move to center
        cmd = bytearray([
            0xFF, 0xFF, servo_id, 0x05, 0x03, 0x1E,
            center_pos & 0xFF, (center_pos >> 8) & 0xFF
        ])
        checksum = (~sum(cmd[2:])) & 0xFF
        cmd.append(checksum)
        port.write(cmd)
        
        calibration[f"servo_{servo_id}"] = {
            "min": min_pos,
            "max": max_pos,
            "center": center_pos
        }
        
        print(f"Servo {servo_id}: min={min_pos}, max={max_pos}, center={center_pos}")
    
    # Save calibration
    with open("so101_calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)
    
    print("\n✅ Calibration saved to so101_calibration.json")
    
    # Release all servos
    for servo_id in range(1, 8):
        cmd = bytearray([0xFF, 0xFF, servo_id, 0x04, 0x03, 0x18, 0x00])
        checksum = (~sum(cmd[2:])) & 0xFF
        cmd.append(checksum)
        port.write(cmd)
    
    port.close()

if __name__ == "__main__":
    calibrate_so101()