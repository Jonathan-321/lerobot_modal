# test_robot_communication.py
import serial
import serial.tools.list_ports
import time

def find_so101():
    """Find SO-101 robot port"""
    print("🔍 Looking for SO-101...")
    
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print(f"Found: {port.device} - {port.description}")
        
        # SO-101 usually has these in description
        if any(x in port.description.lower() for x in ['usb', 'serial', 'uart']):
            try:
                ser = serial.Serial(port.device, 115200, timeout=1)
                print(f"✅ Opened {port.device}")
                
                # Try to communicate
                ser.write(b'\r\n')  # Send newline
                time.sleep(0.1)
                
                if ser.in_waiting:
                    response = ser.read(ser.in_waiting)
                    print(f"Response: {response}")
                    return ser
                    
                ser.close()
            except Exception as e:
                print(f"Failed {port.device}: {e}")
    
    return None

def test_servo_movement(ser):
    """Test moving a single servo"""
    print("\n🔧 Testing servo movement...")
    print("WARNING: Robot will move! Press Enter to continue...")
    input()
    
    # Example commands (adjust for your protocol)
    commands = [
        b"M1:90\n",   # Move servo 1 to 90 degrees
        b"M1:0\n",    # Move back to 0
        b"M1:45\n",   # Move to 45
    ]
    
    for cmd in commands:
        print(f"Sending: {cmd}")
        ser.write(cmd)
        time.sleep(2)  # Wait for movement
        
        if ser.in_waiting:
            response = ser.read(ser.in_waiting)
            print(f"Response: {response}")

if __name__ == "__main__":
    ser = find_so101()
    if ser:
        test_servo_movement(ser)
        ser.close()
    else:
        print("❌ No robot found")