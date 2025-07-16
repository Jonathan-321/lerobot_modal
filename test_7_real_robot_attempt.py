#!/usr/bin/env python3
"""
Test 7: Attempt Real SO-101 Robot Connection
Try to connect to the real robot and test basic communication
"""

import cv2
import numpy as np
import time
import threading
import queue
import serial
import serial.tools.list_ports
from typing import Optional, List, Tuple
import json
import os
from datetime import datetime

try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    print("⚠️ Modal not available - using dummy inference")

def test_real_robot():
    """Test connection to real SO-101 robot."""
    print("🤖 Testing Real SO-101 Robot Connection")
    print("=" * 50)
    
    # Try to connect to the USB device
    robot_port = '/dev/cu.usbmodem5A460843581'
    
    print(f"🔍 Attempting connection to {robot_port}")
    
    # Try different baud rates
    baud_rates = [115200, 57600, 9600, 38400, 19200]
    
    for baud in baud_rates:
        print(f"\n📡 Trying baud rate: {baud}")
        
        try:
            ser = serial.Serial(
                port=robot_port,
                baudrate=baud,
                timeout=2.0,
                write_timeout=2.0
            )
            
            print(f"  ✅ Port opened successfully")
            
            # Clear buffers
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            time.sleep(0.5)
            
            # Try various commands that SO-101 might understand
            test_commands = [
                # Dynamixel-style commands
                (b'\xFF\xFF\x01\x02\x01\xFB', "Dynamixel ping"),
                (b'\xFF\xFF\x01\x04\x02\x24\x01\xD2', "Read position"),
                
                # Text-based commands
                (b'?\r\n', "Query"),
                (b'status\r\n', "Status"),
                (b'pos\r\n', "Position"),
                (b'help\r\n', "Help"),
                (b'info\r\n', "Info"),
                
                # Simple bytes
                (b'\x01', "Single byte"),
                (b'\x00', "Zero byte"),
                (b'ping\r\n', "Ping text"),
            ]
            
            for cmd, desc in test_commands:
                print(f"    🧪 Testing: {desc}")
                print(f"       Command: {cmd.hex() if len(cmd) <= 10 else str(cmd)}")
                
                try:
                    # Send command
                    ser.write(cmd)
                    time.sleep(0.2)  # Wait for response
                    
                    # Read response
                    response = ser.read(100)
                    
                    if response:
                        print(f"       📨 Response ({len(response)} bytes): {response.hex()}")
                        if len(response) <= 50:
                            try:
                                print(f"       📝 As text: {response.decode('ascii', errors='ignore')}")
                            except:
                                pass
                        
                        # This looks promising!
                        print(f"       🎉 GOT RESPONSE! This might be the right protocol")
                        
                        # Try to parse as joint positions
                        if len(response) >= 14:  # 7 joints * 2 bytes
                            try:
                                positions = []
                                for i in range(0, min(14, len(response)), 2):
                                    if i+1 < len(response):
                                        pos = int.from_bytes(response[i:i+2], 'little')
                                        positions.append(pos)
                                print(f"       🦾 Parsed positions: {positions}")
                            except Exception as e:
                                print(f"       ❌ Parse error: {e}")
                    else:
                        print(f"       📭 No response")
                
                except Exception as e:
                    print(f"       ❌ Command error: {e}")
            
            # Try to read any spontaneous data
            print(f"    👂 Listening for spontaneous data...")
            time.sleep(1.0)
            spontaneous = ser.read(100)
            if spontaneous:
                print(f"       📨 Spontaneous data: {spontaneous.hex()}")
            else:
                print(f"       🔇 No spontaneous data")
            
            ser.close()
            print(f"  🔌 Port closed")
            
        except Exception as e:
            print(f"  ❌ Failed to open port: {e}")
    
    print(f"\n💡 Next steps:")
    print(f"  1. Check SO-101 documentation for correct protocol")
    print(f"  2. Try manufacturer's software to verify communication")
    print(f"  3. Look for protocol examples in SO-101 SDK")
    print(f"  4. Consider using robot-specific libraries")

def run_real_robot_integration():
    """Run integration test with real robot attempt."""
    print("\n🚀 Real Robot Integration Attempt")
    print("=" * 40)
    
    # Import the working integration
    from test_6_robot_modal_integration import RobotModalIntegration
    
    # Try with real robot
    integration = RobotModalIntegration(
        camera_index=1,
        simulate_robot=False  # Try real robot!
    )
    
    if integration.robot_working:
        print("🎉 Real robot connected! Running full integration...")
        integration.run_integration()
    else:
        print("❌ Real robot failed, but camera + Modal still work")
        print("💡 You can still test the vision pipeline")
        
        # Ask user if they want to run with simulation
        response = input("\nRun with simulated robot? (y/n): ")
        if response.lower() == 'y':
            integration.simulate_robot = True
            integration.robot_working = True
            integration.run_integration()

def main():
    print("🤖 SO-101 Real Robot Testing")
    print("=" * 50)
    
    print("Choose test mode:")
    print("1. Test robot communication only")
    print("2. Try full integration with real robot")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice in ['1', '3']:
        test_real_robot()
    
    if choice in ['2', '3']:
        run_real_robot_integration()
    
    print("\n✅ Real robot testing complete!")

if __name__ == "__main__":
    main()
