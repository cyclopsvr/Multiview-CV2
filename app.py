import cv2
import numpy as np
import asyncio
import websockets
import json
import base64
import threading
import time
import os
from ultralytics import YOLO
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import logging
from config import Config
from pose_estimation import YOLOv7PoseEstimator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiCameraDetectionSystem:
    def __init__(self):
        self.model = YOLO(Config.YOLO_MODEL)
        self.pose_estimator = YOLOv7PoseEstimator(weights_path="yolov7-w6-pose.pt", device='cpu')
        self.cameras = {}
        # Load camera configs from Config class and detected cameras
        self.camera_configs = self.load_camera_configs()
        self.detection_results = {}
        self.running = False
        
    def load_camera_configs(self):
        """Load camera configurations from config and detected cameras file"""
        
        # Try to load from advanced detection first
        try:
            if os.path.exists('advanced_camera_detection.json'):
                with open('advanced_camera_detection.json', 'r') as f:
                    data = json.load(f)
                    working_cameras = data.get('working_cameras', [])
                    
                    # Convert to our format
                    camera_names = {
                        1: 'MacBook Pro Camera',
                        2: 'HERO11 Black',  
                        3: 'OBS Virtual Camera'
                    }
                    
                    configs = []
                    for cam in working_cameras:
                        config = {
                            'id': cam['index'],
                            'name': camera_names.get(cam['index'], f"Camera {cam['index']}"),
                            'type': 'builtin' if cam['index'] == 1 else 'usb',
                            'resolution': [cam['width'], cam['height']],
                            'fps': int(cam['fps']),
                            'enabled': True
                        }
                        configs.append(config)
                    
                    logger.info(f"Loaded {len(configs)} working cameras from advanced detection")
                    return configs
        except Exception as e:
            logger.warning(f"Could not load advanced camera detection: {e}")
        
        # Fallback to basic detected cameras
        try:
            if os.path.exists('detected_cameras.json'):
                with open('detected_cameras.json', 'r') as f:
                    detected = json.load(f)
                    detected_cameras = detected.get('cameras', [])
                    logger.info(f"Loaded {len(detected_cameras)} detected cameras")
                    return detected_cameras
        except Exception as e:
            logger.warning(f"Could not load detected cameras: {e}")
        
        # Final fallback to config
        configs = Config.get_enabled_cameras()
        logger.info(f"Using default camera configs: {len(configs)} cameras")
        return configs
        
    def initialize_cameras(self):
        """Initialize all available cameras"""
        logger.info(f"Attempting to initialize {len(self.camera_configs)} cameras...")
        
        for config in self.camera_configs:
            camera_id = config['id']
            camera_name = config['name']
            
            try:
                logger.info(f"Initializing camera {camera_id}: {camera_name}")
                cap = cv2.VideoCapture(camera_id)
                
                if cap.isOpened():
                    # Test if we can actually read frames
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Set camera properties based on config
                        resolution = config.get('resolution', [640, 480])
                        fps = config.get('fps', 30)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
                        cap.set(cv2.CAP_PROP_FPS, fps)
                        
                        # Verify final settings
                        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        actual_fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        self.cameras[camera_id] = {
                            'capture': cap,
                            'name': camera_name,
                            'type': config.get('type', 'unknown')
                        }
                        
                        logger.info(f"✅ Camera {camera_id} ({camera_name}) initialized successfully")
                        logger.info(f"   Resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
                    else:
                        logger.warning(f"⚠️  Camera {camera_id} ({camera_name}) opens but cannot read frames")
                        cap.release()
                else:
                    logger.warning(f"❌ Failed to open camera {camera_id}: {camera_name}")
                    
            except Exception as e:
                logger.error(f"❌ Error initializing camera {camera_id} ({camera_name}): {str(e)}")
        
        logger.info(f"Camera initialization complete. {len(self.cameras)} cameras ready.")
    
    def detect_objects(self, frame, camera_id):
        """Run YOLO detection on a frame"""
        try:
            results = self.model(frame)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class': class_name,
                            'class_id': class_id
                        }
                        detections.append(detection)
            
            return detections
        except Exception as e:
            logger.error(f"Error in object detection: {str(e)}")
            return []
    
    def draw_detections(self, frame, detections):
        """Draw detection boxes and labels on frame"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame
    
    def process_camera_feed(self, camera_id, socketio):
        """Process individual camera feed"""
        camera = self.cameras[camera_id]
        cap = camera['capture']
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read from camera {camera_id}")
                continue
            
            # Run object detection
            detections = self.detect_objects(frame, camera_id)
            
            # Run pose estimation for person detections
            poses = []
            person_detections = [d for d in detections if d['class'] == 'person']
            if person_detections:
                poses = self.pose_estimator.detect_poses(frame)
            
            # Draw detections and poses
            frame_with_detections = self.draw_detections(frame.copy(), detections)
            if poses:
                frame_with_detections = self.pose_estimator.draw_poses(frame_with_detections, poses)
            
            # Store detection results
            self.detection_results[camera_id] = {
                'detections': detections,
                'poses': poses,
                'timestamp': time.time(),
                'camera_name': camera['name']
            }
            
            # Convert frame to base64 for web streaming
            _, buffer = cv2.imencode('.jpg', frame_with_detections)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Emit to web interface
            socketio.emit('camera_frame', {
                'camera_id': camera_id,
                'camera_name': camera['name'],
                'frame': frame_base64,
                'detections': detections,
                'poses': poses
            })
            
            time.sleep(0.033)  # ~30 FPS
    
    def start_processing(self, socketio):
        """Start processing all camera feeds"""
        self.running = True
        threads = []
        
        for camera_id in self.cameras:
            thread = threading.Thread(target=self.process_camera_feed, args=(camera_id, socketio))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        return threads
    
    def stop_processing(self):
        """Stop processing all camera feeds"""
        self.running = False
        for camera_id in self.cameras:
            self.cameras[camera_id]['capture'].release()
    
    def get_detection_summary(self):
        """Get summary of all current detections"""
        summary = {
            'total_objects': 0,
            'cameras_active': len(self.cameras),
            'detections_by_camera': {}
        }
        
        for camera_id, results in self.detection_results.items():
            camera_detections = results['detections']
            summary['total_objects'] += len(camera_detections)
            summary['detections_by_camera'][camera_id] = {
                'camera_name': results['camera_name'],
                'object_count': len(camera_detections),
                'objects': [d['class'] for d in camera_detections]
            }
        
        return summary

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize detection system
detection_system = MultiCameraDetectionSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/cameras')
def get_cameras():
    return jsonify(list(detection_system.cameras.keys()))

@app.route('/api/detection-summary')
def get_detection_summary():
    return jsonify(detection_system.get_detection_summary())

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    emit('status', {'message': 'Connected to detection system'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

@socketio.on('start_detection')
def handle_start_detection():
    logger.info('Starting detection system')
    detection_system.initialize_cameras()
    detection_system.start_processing(socketio)
    emit('status', {'message': 'Detection system started'})

@socketio.on('stop_detection')
def handle_stop_detection():
    logger.info('Stopping detection system')
    detection_system.stop_processing()
    emit('status', {'message': 'Detection system stopped'})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5002, debug=True)
