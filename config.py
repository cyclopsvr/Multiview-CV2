import os

class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-this-in-production'
    
    # Camera configuration (detected working cameras)
    CAMERA_CONFIGS = [
        {
            'id': 0,
            'name': 'MacBook Pro Camera',
            'type': 'builtin',
            'resolution': (1920, 1080),
            'fps': 30,
            'enabled': True
        },
        {
            'id': 1,
            'name': 'iPhone Camera',  # Chewbacca Camera
            'type': 'usb',
            'resolution': (1920, 1080),
            'fps': 15,
            'enabled': True
        },
        {
            'id': 2,
            'name': 'HERO11 Black',  # GoPro camera
            'type': 'usb',
            'resolution': (1920, 1080),
            'fps': 30,
            'enabled': True
        },
        {
            'id': 3,
            'name': 'OBS Virtual Camera',
            'type': 'virtual',
            'resolution': (1920, 1080),
            'fps': 30,
            'enabled': True
        }
    ]
    
    # YOLO model configuration
    YOLO_MODEL = 'yolov8n.pt'  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    
    # Detection settings
    MAX_DETECTIONS = 100
    DETECTION_CLASSES = None  # None for all classes, or list of class IDs to detect
    
    # Streaming settings
    STREAM_QUALITY = 75  # JPEG quality (1-100)
    STREAM_FPS = 30
    FRAME_BUFFER_SIZE = 1
    
    # Performance settings
    USE_GPU = True  # Use GPU if available
    BATCH_SIZE = 1
    ASYNC_PROCESSING = True
    
    # Web interface settings
    WEB_HOST = '0.0.0.0'
    WEB_PORT = 5000
    DEBUG = os.environ.get('FLASK_ENV') == 'development'
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'detection_system.log'
    
    # Recording settings (optional)
    ENABLE_RECORDING = False
    RECORDING_PATH = './recordings'
    RECORDING_FORMAT = 'mp4'
    
    # Alert settings (optional)
    ENABLE_ALERTS = False
    ALERT_CLASSES = ['person', 'car', 'truck']  # Classes that trigger alerts
    ALERT_CONFIDENCE = 0.7
    
    @classmethod
    def get_enabled_cameras(cls):
        """Get list of enabled cameras"""
        return [cam for cam in cls.CAMERA_CONFIGS if cam['enabled']]
    
    @classmethod
    def get_camera_by_id(cls, camera_id):
        """Get camera configuration by ID"""
        for cam in cls.CAMERA_CONFIGS:
            if cam['id'] == camera_id:
                return cam
        return None
    
    @classmethod
    def update_camera_status(cls, camera_id, enabled):
        """Update camera enabled status"""
        for cam in cls.CAMERA_CONFIGS:
            if cam['id'] == camera_id:
                cam['enabled'] = enabled
                return True
        return False
