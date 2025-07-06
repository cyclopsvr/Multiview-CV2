#!/usr/bin/env python3
import cv2
import time
import subprocess
import json

def get_system_cameras():
    """Get camera info using system_profiler"""
    try:
        result = subprocess.run(['system_profiler', 'SPCameraDataType'], 
                              capture_output=True, text=True)
        print("System Camera Information:")
        print("=" * 50)
        print(result.stdout)
        return result.stdout
    except Exception as e:
        print(f"Error getting system camera info: {e}")
        return None

def test_opencv_backends():
    """Test different OpenCV backends"""
    backends = [
        (cv2.CAP_AVFOUNDATION, "AVFoundation (macOS)"),
        (cv2.CAP_V4L2, "Video4Linux"),
        (cv2.CAP_GSTREAMER, "GStreamer"),
        (cv2.CAP_FFMPEG, "FFmpeg"),
        (cv2.CAP_ANY, "Any available")
    ]
    
    print("\nTesting OpenCV Backends:")
    print("=" * 50)
    
    working_cameras = {}
    
    for backend_id, backend_name in backends:
        print(f"\nTesting {backend_name} backend...")
        
        for camera_id in range(10):
            try:
                cap = cv2.VideoCapture(camera_id, backend_id)
                if cap.isOpened():
                    # Try to read multiple frames
                    success_count = 0
                    for attempt in range(10):
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            success_count += 1
                        time.sleep(0.1)
                    
                    if success_count > 5:  # At least half successful
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        key = f"{backend_name}_{camera_id}"
                        working_cameras[key] = {
                            'backend': backend_name,
                            'backend_id': backend_id,
                            'camera_id': camera_id,
                            'resolution': (width, height),
                            'fps': fps,
                            'success_rate': f"{success_count}/10"
                        }
                        
                        print(f"  ✅ Camera {camera_id}: {width}x{height} @ {fps}fps ({success_count}/10 frames)")
                    else:
                        print(f"  ⚠️  Camera {camera_id}: Low success rate ({success_count}/10)")
                        
                cap.release()
            except Exception as e:
                print(f"  ❌ Camera {camera_id}: Error - {e}")
    
    return working_cameras

def test_different_indices():
    """Test a wider range of camera indices"""
    print("\nExtensive Camera Index Testing:")
    print("=" * 50)
    
    working_cameras = []
    
    # Test indices 0-20
    for i in range(21):
        try:
            print(f"Testing camera index {i}...")
            cap = cv2.VideoCapture(i)
            
            if cap.isOpened():
                # Try to set properties first
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Test multiple frame reads
                frames_captured = 0
                total_attempts = 20
                
                for attempt in range(total_attempts):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        frames_captured += 1
                    time.sleep(0.05)  # 50ms delay
                
                if frames_captured > total_attempts // 2:  # More than 50% success
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                    
                    camera_info = {
                        'index': i,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'fourcc': fourcc,
                        'success_rate': f"{frames_captured}/{total_attempts}",
                        'working': True
                    }
                    
                    working_cameras.append(camera_info)
                    print(f"  ✅ Camera {i}: {width}x{height} @ {fps}fps - {frames_captured}/{total_attempts} frames")
                else:
                    print(f"  ⚠️  Camera {i}: Poor performance - {frames_captured}/{total_attempts} frames")
            else:
                print(f"  ❌ Camera {i}: Failed to open")
                
            cap.release()
            
        except Exception as e:
            print(f"  ❌ Camera {i}: Exception - {e}")
    
    return working_cameras

def test_with_photobooth_running():
    """Test cameras while PhotoBooth might be running"""
    print("\nTesting with potential PhotoBooth conflicts:")
    print("=" * 50)
    
    # Check if PhotoBooth or other camera apps are running
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        camera_apps = []
        for line in result.stdout.split('\n'):
            if any(app in line.lower() for app in ['photobooth', 'camera', 'obs', 'zoom', 'facetime']):
                camera_apps.append(line.strip())
        
        if camera_apps:
            print("Detected camera applications running:")
            for app in camera_apps:
                print(f"  - {app}")
        else:
            print("No obvious camera applications detected")
            
    except Exception as e:
        print(f"Error checking running applications: {e}")

def create_camera_config(working_cameras):
    """Create a camera configuration based on detected cameras"""
    if not working_cameras:
        print("\n❌ No working cameras found!")
        return None
    
    camera_names = [
        "MacBook Pro Camera",
        "HERO11 Black",
        "Insta360 X3", 
        "OBS Virtual Camera"
    ]
    
    config = []
    for i, camera in enumerate(working_cameras):
        name = camera_names[i] if i < len(camera_names) else f"Camera {camera['index']}"
        
        camera_config = {
            'id': camera['index'],
            'name': name,
            'type': 'usb' if camera['index'] > 0 else 'builtin',
            'resolution': [camera['width'], camera['height']],
            'fps': min(30, int(camera['fps']) if camera['fps'] > 0 else 30),
            'enabled': True,
            'success_rate': camera['success_rate']
        }
        config.append(camera_config)
    
    return config

def test_camera_exclusive_access():
    """Test if cameras can be accessed exclusively"""
    print("\nTesting Exclusive Camera Access:")
    print("=" * 50)
    
    for i in range(5):
        try:
            print(f"Testing exclusive access to camera {i}...")
            
            # Try to open camera
            cap = cv2.VideoCapture(i)
            if not cap.isOpened():
                print(f"  ❌ Camera {i}: Cannot open")
                continue
            
            # Check if we can read frames
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"  ⚠️  Camera {i}: Opens but no frames")
                cap.release()
                continue
                
            # Try to open same camera again (should fail if exclusive)
            cap2 = cv2.VideoCapture(i)
            ret2, frame2 = cap2.read()
            
            if ret2 and frame2 is not None:
                print(f"  ⚠️  Camera {i}: Non-exclusive access (may cause conflicts)")
            else:
                print(f"  ✅ Camera {i}: Exclusive access working")
            
            cap.release()
            cap2.release()
            
        except Exception as e:
            print(f"  ❌ Camera {i}: Error - {e}")

if __name__ == "__main__":
    print("Advanced Camera Detection Tool")
    print("=" * 60)
    
    # Get system camera information
    get_system_cameras()
    
    # Test running applications
    test_with_photobooth_running()
    
    # Test exclusive access
    test_camera_exclusive_access()
    
    # Test different backends
    backend_results = test_opencv_backends()
    
    # Test extensive camera indices
    working_cameras = test_different_indices()
    
    print(f"\n🎯 RESULTS SUMMARY")
    print("=" * 60)
    print(f"Working cameras found: {len(working_cameras)}")
    
    if working_cameras:
        print("\nDetected working cameras:")
        for camera in working_cameras:
            print(f"  Camera {camera['index']}: {camera['width']}x{camera['height']} @ {camera['fps']}fps ({camera['success_rate']})")
        
        # Create configuration
        config = create_camera_config(working_cameras)
        
        # Save results
        results = {
            'timestamp': time.time(),
            'working_cameras': working_cameras,
            'backend_results': backend_results,
            'recommended_config': config
        }
        
        with open('advanced_camera_detection.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Results saved to 'advanced_camera_detection.json'")
        
        if len(working_cameras) >= 3:
            print("🎉 Found sufficient cameras for multiview detection!")
        else:
            print("⚠️  Found fewer cameras than expected")
    else:
        print("\n❌ No working cameras detected")
        print("\nTroubleshooting suggestions:")
        print("1. Make sure PhotoBooth and other camera apps are closed")
        print("2. Check System Preferences > Security & Privacy > Camera")
        print("3. Try running with sudo (not recommended but for testing)")
        print("4. Restart your Mac to reset camera states")
