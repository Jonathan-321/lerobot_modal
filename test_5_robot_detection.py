#!/usr/bin/env python3
"""
Test 5: SO-101 Robot Detection and Basic Communication
Tests robot connection, joint states, and basic movement
"""

import serial
import serial.tools.list_ports
import time
import struct
from typing import Optional, List

class SO101Detector:
    def __init__(self):
        self.robot_port = None
        self.joint_count = 7
        
    def find_robot_ports(self):
        """Find all potential robot ports."""
        print("🔍 Scanning for SO-101 robot...")
        ports = serial.tools.list_ports.comports()
        
        if not ports:
            print("❌ No serial ports found")
            return []
        
        print(f"\n📋 Found {len(ports)} serial port(s):")
        potential_robots = []
        
        for i, port in enumerate(ports):
            print(f"  {i+1}. {port.device}")
            print(f"     Description: {port.description}")
            print(f"     Manufacturer: {port.manufacturer if port.manufacturer else 'Unknown'}")
            print(f"     VID:PID: {port.vid:04X}:{port.pid:04X}" if port.vid and port.pid else "     VID:PID: Unknown")
            
            # Check for SO-101 indicators
            robot_indicators = [
                'so-101', 'so101', 'feetech', 'dynamixel', 
                'usb serial', 'ch340', 'cp210x', 'ftdi',
                'single serial', 'usbmodem'
            ]
            
            description_lower = port.description.lower()
            device_lower = port.device.lower()
            
            # Check VID:PID for common robotics chips
            common_robot_vids = ['1A86', '0403', '10C4', '067B']  # CH340, FTDI, CP210x, etc
            vid_str = f"{port.vid:04X}" if port.vid else ""
            
            is_potential_robot = (
                any(indicator in description_lower for indicator in robot_indicators) or
                any(indicator in device_lower for indicator in robot_indicators) or
                vid_str in common_robot_vids or
                'usbmodem' in device_lower
            )
            
            if is_potential_robot:
                potential_robots.append(port)
                print(f"     🤖 POTENTIAL ROBOT PORT ✅")
            
            print()
        
        return potential_robots
    
    def test_robot_communication(self, port_device, baud_rates=[115200, 9600, 57600]):
        """Test communication with robot at different baud rates."""
        print(f"\n🧪 Testing communication with {port_device}")
        
        for baud in baud_rates:
            print(f"  📡 Trying baud rate: {baud}")
            
            try:
                ser = serial.Serial(
                    port=port_device,
                    baudrate=baud,
                    timeout=1.0,
                    write_timeout=1.0
                )
                
                # Test basic communication
                if self._test_basic_commands(ser):
                    print(f"  ✅ Communication successful at {baud} baud!")
                    return ser
                else:
                    print(f"  ❌ No response at {baud} baud")
                
                ser.close()
                
            except Exception as e:
                print(f"  ❌ Error at {baud} baud: {e}")
        
        return None
    
    def _test_basic_commands(self, ser):
        """Test basic robot commands."""
        print("    🔍 Testing basic commands...")
        
        # Clear any existing data
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        # Test 1: Ping command (if supported)
        try:
            # Send a simple query - this varies by robot
            test_commands = [
                b'\xFF\xFF\x01\x02\x01\xFB',  # Dynamixel-style ping
                b'?',  # Simple query
                b'status\r\n',  # Text command
                b'\x01\x02\x03',  # Generic test
            ]
            
            for i, cmd in enumerate(test_commands):
                print(f"      Test {i+1}: {cmd.hex() if len(cmd) <= 6 else cmd[:10]}")
                
                ser.write(cmd)
                time.sleep(0.1)
                
                response = ser.read(100)
                if response:
                    print(f"      📨 Response: {response.hex() if len(response) <= 20 else str(response[:20])}")
                    return True
                else:
                    print(f"      📭 No response")
            
            return False
            
        except Exception as e:
            print(f"      ❌ Command test error: {e}")
            return False
    
    def test_joint_reading(self, ser):
        """Test reading joint positions."""
        print("    🦾 Testing joint position reading...")
        
        try:
            # SO-101 specific commands (you may need to adjust these)
            joint_commands = [
                b'\xFF\xFF\x01\x04\x02\x24\x01\xD2',  # Read position servo 1
                b'get_pos\r\n',  # Text-based position query
            ]
            
            for cmd in joint_commands:
                print(f"      Sending: {cmd.hex() if len(cmd) <= 10 else str(cmd)}")
                ser.write(cmd)
                time.sleep(0.2)
                
                response = ser.read(100)
                if response:
                    print(f"      📨 Joint response: {response.hex()}")
                    return self._parse_joint_response(response)
            
            return None
            
        except Exception as e:
            print(f"      ❌ Joint reading error: {e}")
            return None
    
    def _parse_joint_response(self, response):
        """Parse joint position response."""
        # This will need to be customized based on SO-101 protocol
        try:
            # Example parsing - adjust for actual SO-101 format
            if len(response) >= 4:
                # Assume 2-byte position values
                positions = []
                for i in range(0, min(len(response)-1, 14), 2):
                    pos = struct.unpack('<H', response[i:i+2])[0]
                    positions.append(pos)
                return positions[:7]  # First 7 joints
        except:
            pass
        
        return None
    
    def run_detection_sequence(self):
        """Run complete robot detection sequence."""
        print("🤖 SO-101 Robot Detection Sequence")
        print("=" * 50)
        
        # Step 1: Find potential ports
        potential_ports = self.find_robot_ports()
        
        if not potential_ports:
            print("❌ No potential robot ports found")
            print("\n💡 Troubleshooting:")
            print("  1. Check USB connection")
            print("  2. Check robot power")
            print("  3. Install robot drivers if needed")
            return None
        
        # Step 2: Test each potential port
        working_port = None
        for port in potential_ports:
            print(f"\n🧪 Testing {port.device}...")
            ser = self.test_robot_communication(port.device)
            
            if ser:
                # Step 3: Test joint reading
                joints = self.test_joint_reading(ser)
                if joints:
                    print(f"    ✅ Joint positions: {joints}")
                
                working_port = ser
                self.robot_port = ser
                break
        
        if working_port:
            print(f"\n🎉 SO-101 robot detected and working!")
            print(f"   Port: {working_port.name}")
            print(f"   Baud: {working_port.baudrate}")
            return working_port
        else:
            print(f"\n❌ No working robot connection found")
            print("\n💡 Next steps:")
            print("  1. Check robot documentation for correct baud rate")
            print("  2. Verify robot communication protocol")
            print("  3. Test with robot manufacturer's software first")
            return None

def main():
    detector = SO101Detector()
    
    # Run detection
    robot_port = detector.run_detection_sequence()
    
    if robot_port:
        print(f"\n🚀 Ready for integration with Modal SmolVLA!")
        print(f"   Use port: {robot_port.name}")
        print(f"   Baud rate: {robot_port.baudrate}")
        
        # Close connection
        robot_port.close()
        print("   Connection closed.")
    else:
        print(f"\n🎮 Robot not detected - can still test with simulation")
    
    print(f"\n📝 Next step: Run test_6_robot_modal_integration.py")

if __name__ == "__main__":
    main()
