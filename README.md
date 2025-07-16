# LeRobot Modal SmolVLA Setup

Real-time robot vision system using Modal-hosted SmolVLA inference with local camera input.

## ✅ Working Status

### Current Setup (Mac)
- **Camera**: Mac camera (index 1) at 1280x720@30fps ✅
- **Modal Deployment**: `smolvla-so101-deployment` ✅
- **Inference**: ~2.8s average, ~0.4 FPS ✅
- **Pipeline**: Complete vision → inference → actions ✅

### Next Steps (Linux)
- [ ] Connect to real robot hardware (USB)
- [ ] Get actual robot joint states
- [ ] Calibrate workspace
- [ ] Execute real robot actions

## 🏗️ Project Structure

```
lerobot_modal/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
│
├── modal_runner.py                     # Main Modal GPU inference function
├── inference_server.py                 # Optional web API
├── client_test.py                      # Modal RPC client
│
├── find_cameras.py                     # Standalone camera detection
├── test_1_camera_detection.py          # Enhanced camera testing
├── test_2_camera_live.py               # Live camera feed with preprocessing
├── test_3_modal_connection.py          # Modal connection testing
├── test_4_complete_pipeline.py         # Full camera + Modal pipeline
│
├── setup_lerobot_camera.py            # LeRobot camera configuration
├── test_lerobot_cameras.py            # LeRobot camera tests
├── test_robot_communication.py        # Robot communication tests
├── test_complete_pipeline.py           # Original pipeline test
├── so101_practical_setup.py           # SO-101 robot setup
├── so101_real_robot_control.py        # SO-101 control logic
├── so101_smolvla_client.py            # SO-101 SmolVLA client
│
└── lerobot/                           # Optional LeRobot clone
```

## 🚀 Quick Start

### 1. Setup on New Machine (Linux)

```bash
# Clone repository
git clone <your-repo-url>
cd lerobot_modal

# Create virtual environment (Python 3.11 or 3.12)
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install lerobot
pip install git+https://github.com/huggingface/lerobot.git
```

### 2. Test Camera

```bash
# Find available cameras
python3 find_cameras.py

# Test live camera feed
python3 test_2_camera_live.py
```

### 3. Test Modal Connection

```bash
# Test Modal deployment
python3 test_3_modal_connection.py

# Run complete pipeline
python3 test_4_complete_pipeline.py
```

## 📋 Modal Deployment

### Current Deployment
- **App Name**: `smolvla-so101-deployment`
- **Function**: `get_action_direct`
- **GPU**: A10G with model caching
- **Performance**: ~2.8s inference time

### Check Status
```bash
modal app list
modal app logs smolvla-so101-deployment
```

### Redeploy if Needed
```bash
modal deploy modal_runner.py
```

## 🎥 Camera Configuration

### Working Setup (Mac)
- **Camera 1**: Mac camera (1280x720@30fps)
- **OpenCV**: Version 4.12.0

### For Linux
- Check available cameras: `python3 find_cameras.py`
- Typical indices: 0 (built-in), 1+ (external)
- USB cameras should auto-detect

## 🤖 Robot Integration (Next Phase)

### Hardware Requirements
- USB connection to robot
- Serial/UART communication
- Robot firmware supporting joint state queries

### Steps for Linux Machine
1. **Hardware Setup**
   - Connect robot via USB
   - Install robot drivers/firmware
   - Test basic communication

2. **Get Joint States**
   ```python
   # Replace dummy robot state in pipeline
   robot_state = get_real_robot_joints()  # [x, y, z, rx, ry, rz, gripper]
   ```

3. **Workspace Calibration**
   - Define robot working area
   - Camera-to-robot coordinate mapping
   - Safety boundaries

4. **Action Execution**
   - Send predicted actions to robot
   - Implement safety checks
   - Real-time control loop

## 🧪 Test Scripts

### Camera Tests
- `find_cameras.py` - Detect all cameras
- `test_1_camera_detection.py` - Enhanced camera testing
- `test_2_camera_live.py` - Live feed with preprocessing

### Modal Tests  
- `test_3_modal_connection.py` - Test inference connection
- `test_4_complete_pipeline.py` - Complete vision pipeline

### Robot Tests (Linux)
- `test_robot_communication.py` - Basic robot communication
- `so101_practical_setup.py` - SO-101 specific setup
- `so101_real_robot_control.py` - Real robot control

## 🐛 Troubleshooting

### Camera Issues
- **No cameras found**: Check USB connections, camera permissions
- **Permission denied**: Add user to video group: `sudo usermod -a -G video $USER`
- **OpenCV errors**: Install: `pip install opencv-python`

### Modal Issues
- **Connection failed**: Check `modal token set`
- **App not found**: Redeploy with `modal deploy modal_runner.py`
- **Slow inference**: First call is cold start (~55s), subsequent calls ~2.8s

### Robot Issues (Linux)
- **USB not detected**: Check `lsusb`, install drivers
- **Permission denied**: Add to dialout group: `sudo usermod -a -G dialout $USER`
- **Serial errors**: Check device path `/dev/ttyUSB*` or `/dev/ttyACM*`

## 📊 Performance Metrics

### Current Performance (Mac)
- **Camera FPS**: 30 (capture), ~0.4 (processed)
- **Inference Time**: 2.8s average
- **Bottleneck**: Modal inference (network + GPU)
- **Memory Usage**: ~500MB

### Expected Linux Performance
- Similar or better camera performance
- Same Modal inference times
- Better USB/serial performance
- More reliable robot connections

## 🔗 Dependencies

### Core
- `opencv-python` - Camera capture and processing
- `modal` - GPU inference hosting
- `numpy` - Array operations
- `threading`, `queue` - Async processing

### Robot (Linux)
- `lerobot` - Robot framework
- `pyserial` - Serial communication
- `transformers` - Model utilities
- `torch` - Deep learning framework

## 📝 Notes

### Mac Limitations
- Limited USB ports
- Camera 0 (laptop) doesn't work properly
- Camera 1 (Mac camera) works well

### Linux Advantages
- Better USB hardware support
- Native robot driver support
- More reliable serial communication
- Better performance for robot control

## 🎯 Current Status Summary

**✅ Completed on Mac:**
- Modal SmolVLA deployment working
- Camera detection and live feed
- Complete vision pipeline
- Action prediction working

**🔄 Next on Linux:**
- Real robot hardware connection
- Joint state integration
- Workspace calibration
- Action execution

**🚀 Ready for Production:**
- Vision system tested and working
- Modal inference optimized
- Code organized and documented
- Ready for robot integration

---

**Last Updated**: 2025-07-16 by Jonathan  
**Platform**: Mac → Linux transition  
**Status**: Ready for hardware integration
