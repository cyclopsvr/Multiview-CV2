#!/usr/bin/env python3
import cv2
import json
import time

def detect_cameras(max_cameras=10):
    """
    Detect all available cameras and their properties
    """
    available_cameras = []
    
    print("Scanning for available cameras...")
    
    for i in range(max_cameras):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to read a frame to verify the camera works
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Get camera properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    # Try to get camera name (this might not work on all systems)
                    backend_name = cap.getBackendName()
                    
                    camera_info = {
                        'index': i,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'backend': backend_name,
                        'working': True
                    }
                    
                    available_cameras.append(camera_info)
                    print(f"✓ Camera {i}: {width}x{height} @ {fps:.1f}fps ({backend_name})")
                else:
                    print(f"✗ Camera {i}: Opened but failed to read frame")
                cap.release()
            else:
                # Don't print anything for non-existent cameras
                pass
        except Exception as e:
            print(f"✗ Camera {i}: Error - {str(e)}")
    
    return available_cameras

def generate_camera_config(cameras):
    """
    Generate camera configuration based on detected cameras
    """
    camera_names = [
        "MacBook Pro Camera",
        "HERO11 Black", 
        "Insta360 X3",
        "OBS Virtual Camera"
    ]
    
    config = []
    for i, camera in enumerate(cameras):
        name = camera_names[i] if i < len(camera_names) else f"Camera {camera['index']}"
        
        camera_config = {
            'id': camera['index'],
            'name': name,
            'type': 'usb' if 'USB' in camera['backend'] else 'builtin',
            'resolution': (camera['width'], camera['height']),
            'fps': min(30, int(camera['fps']) if camera['fps'] > 0 else 30),
            'enabled': True
        }
        config.append(camera_config)
    
    return config

def test_camera_capture(camera_index, duration=3):
    """
    Test camera capture for a few seconds
    """
    print(f"\nTesting camera {camera_index} for {duration} seconds...")
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Failed to open camera {camera_index}")
        return False
    
    frames_captured = 0
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if ret:
                frames_captured += 1
            else:
                print(f"Failed to read frame from camera {camera_index}")
                break
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("Test interrupted by user")
    
    finally:
        cap.release()
    
    elapsed = time.time() - start_time
    fps = frames_captured / elapsed if elapsed > 0 else 0
    
    print(f"Camera {camera_index}: Captured {frames_captured} frames in {elapsed:.1f}s ({fps:.1f} fps)")
    return frames_captured > 0

if __name__ == "__main__":
    print("Camera Detection and Testing Tool")
    print("=" * 40)
    
    # Detect all available cameras
    cameras = detect_cameras()
    
    if not cameras:
        print("\n❌ No working cameras detected!")
        exit(1)
    
    print(f"\n✅ Found {len(cameras)} working cameras")
    
    # Generate configuration
    config = generate_camera_config(cameras)
    
    print("\nGenerated Camera Configuration:")
    print(json.dumps(config, indent=2))
    
    # Test each camera
    print("\n" + "=" * 40)
    print("Testing Camera Capture")
    print("=" * 40)
    
    working_cameras = []
    for camera in cameras:
        if test_camera_capture(camera['index']):
            working_cameras.append(camera)
    
    print(f"\n✅ {len(working_cameras)} cameras are working properly")
    
    # Save configuration to file
    config_file = "detected_cameras.json"
    with open(config_file, 'w') as f:
        json.dump({
            'cameras': config,
            'working_cameras': len(working_cameras),
            'detection_time': time.time()
        }, f, indent=2)
    
    print(f"\n📁 Configuration saved to {config_file}")
    
    if len(working_cameras) >= 4:
        print("🎉 All 4 cameras detected and working!")
    else:
        print(f"⚠️  Only {len(working_cameras)} cameras working (expected 4)")
