import cv2
import time
import torch
import numpy as np
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt, strip_optimizer
from utils.plots import output_to_keypoint, plot_one_box_kpt, colors
import logging

logger = logging.getLogger(__name__)

class YOLOv7PoseEstimator:
    def __init__(self, weights_path="yolov7-w6-pose.pt", device='cpu', conf_threshold=0.25, iou_threshold=0.65):
        """
        Initialize YOLOv7 Pose Estimation model
        
        Args:
            weights_path (str): Path to YOLOv7 pose weights
            device (str): Device to run inference on ('cpu' or 'cuda')
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
        """
        self.device = select_device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.names = None
        self.weights_path = weights_path
        
        # COCO pose keypoint indices
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Skeleton connections for drawing
        self.skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (11, 12), (5, 11), (6, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        self.load_model()
    
    def load_model(self):
        """Load the YOLOv7 pose estimation model"""
        try:
            logger.info(f"Loading YOLOv7 pose model from {self.weights_path}")
            self.model = attempt_load(self.weights_path, map_location=self.device)
            self.model.eval()
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            logger.info("YOLOv7 pose model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLOv7 pose model: {e}")
            self.model = None
    
    def detect_poses(self, frame):
        """
        Detect poses in a frame
        
        Args:
            frame (np.array): Input frame in BGR format
            
        Returns:
            list: List of detected poses with keypoints and bounding boxes
        """
        if self.model is None:
            return []
        
        try:
            # Preprocess frame
            original_shape = frame.shape[:2]
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = letterbox(image, (640), stride=64, auto=True)[0]
            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))
            image = image.to(self.device).float()
            
            # Run inference
            with torch.no_grad():
                output_data, _ = self.model(image)
            
            # Apply NMS
            output_data = non_max_suppression_kpt(
                output_data,
                self.conf_threshold,
                self.iou_threshold,
                nc=self.model.yaml['nc'],
                nkpt=self.model.yaml['nkpt'],
                kpt_label=True
            )
            
            poses = []
            for i, pose_data in enumerate(output_data):
                if len(pose_data):
                    for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose_data[:, :6])):
                        # Get keypoints
                        kpts = pose_data[det_index, 6:]
                        
                        # Scale bounding box back to original image size
                        bbox = self.scale_bbox(xyxy, image.shape[2:], original_shape)
                        
                        # Process keypoints
                        keypoints = self.process_keypoints(kpts, image.shape[2:], original_shape)
                        
                        pose_info = {
                            'bbox': bbox,
                            'confidence': float(conf),
                            'class': int(cls),
                            'keypoints': keypoints,
                            'raw_keypoints': kpts
                        }
                        poses.append(pose_info)
            
            return poses
            
        except Exception as e:
            logger.error(f"Error in pose detection: {e}")
            return []
    
    def scale_bbox(self, xyxy, input_shape, original_shape):
        """Scale bounding box from input size to original image size"""
        gain = min(input_shape[0] / original_shape[0], input_shape[1] / original_shape[1])
        pad = (input_shape[1] - original_shape[1] * gain) / 2, (input_shape[0] - original_shape[0] * gain) / 2
        
        x1, y1, x2, y2 = xyxy
        x1 = int((x1 - pad[0]) / gain)
        y1 = int((y1 - pad[1]) / gain)
        x2 = int((x2 - pad[0]) / gain)
        y2 = int((y2 - pad[1]) / gain)
        
        return [x1, y1, x2, y2]
    
    def process_keypoints(self, kpts, input_shape, original_shape):
        """Process and scale keypoints to original image size"""
        gain = min(input_shape[0] / original_shape[0], input_shape[1] / original_shape[1])
        pad = (input_shape[1] - original_shape[1] * gain) / 2, (input_shape[0] - original_shape[0] * gain) / 2
        
        keypoints = []
        kpts = kpts.cpu().numpy() if hasattr(kpts, 'cpu') else kpts
        
        # Process keypoints (x, y, visibility) for each point
        for i in range(0, len(kpts), 3):
            if i + 2 < len(kpts):
                x = int((kpts[i] - pad[0]) / gain)
                y = int((kpts[i + 1] - pad[1]) / gain)
                visibility = float(kpts[i + 2])
                
                keypoints.append({
                    'x': x,
                    'y': y,
                    'visibility': visibility,
                    'name': self.keypoint_names[i // 3] if i // 3 < len(self.keypoint_names) else f'point_{i // 3}'
                })
        
        return keypoints
    
    def draw_poses(self, frame, poses, line_thickness=2):
        """
        Draw detected poses on frame
        
        Args:
            frame (np.array): Input frame
            poses (list): List of detected poses
            line_thickness (int): Thickness of drawn lines
            
        Returns:
            np.array: Frame with poses drawn
        """
        if not poses:
            return frame
        
        result_frame = frame.copy()
        
        for pose in poses:
            bbox = pose['bbox']
            keypoints = pose['keypoints']
            confidence = pose['confidence']
            
            # Draw bounding box
            cv2.rectangle(result_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), line_thickness)
            
            # Draw confidence
            label = f'Person: {confidence:.2f}'
            cv2.putText(result_frame, label, (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), line_thickness)
            
            # Draw keypoints
            for kpt in keypoints:
                if kpt['visibility'] > 0.5:  # Only draw visible keypoints
                    cv2.circle(result_frame, (kpt['x'], kpt['y']), 4, (0, 0, 255), -1)
            
            # Draw skeleton connections
            for connection in self.skeleton_connections:
                kpt1_idx, kpt2_idx = connection
                if kpt1_idx < len(keypoints) and kpt2_idx < len(keypoints):
                    kpt1 = keypoints[kpt1_idx]
                    kpt2 = keypoints[kpt2_idx]
                    
                    if kpt1['visibility'] > 0.5 and kpt2['visibility'] > 0.5:
                        cv2.line(result_frame, 
                                (kpt1['x'], kpt1['y']), 
                                (kpt2['x'], kpt2['y']), 
                                (255, 0, 0), line_thickness)
        
        return result_frame
    
    def get_pose_summary(self, poses):
        """
        Get a summary of detected poses
        
        Args:
            poses (list): List of detected poses
            
        Returns:
            dict: Summary information
        """
        if not poses:
            return {
                'person_count': 0,
                'average_confidence': 0.0,
                'pose_data': []
            }
        
        total_confidence = sum(pose['confidence'] for pose in poses)
        average_confidence = total_confidence / len(poses)
        
        pose_data = []
        for i, pose in enumerate(poses):
            pose_summary = {
                'person_id': i + 1,
                'confidence': pose['confidence'],
                'bbox': pose['bbox'],
                'keypoint_count': len([kpt for kpt in pose['keypoints'] if kpt['visibility'] > 0.5])
            }
            pose_data.append(pose_summary)
        
        return {
            'person_count': len(poses),
            'average_confidence': average_confidence,
            'pose_data': pose_data
        }

# Utility function to download YOLOv7 pose weights if not present
def download_pose_weights():
    """Download YOLOv7 pose weights if not present"""
    import os
    import requests
    from tqdm import tqdm
    
    weights_url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt"
    weights_path = "yolov7-w6-pose.pt"
    
    if not os.path.exists(weights_path):
        logger.info("Downloading YOLOv7 pose weights...")
        try:
            response = requests.get(weights_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(weights_path, 'wb') as file, tqdm(
                desc=weights_path,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = file.write(chunk)
                    bar.update(size)
            
            logger.info(f"Downloaded {weights_path}")
        except Exception as e:
            logger.error(f"Failed to download weights: {e}")
            return False
    
    return True

if __name__ == "__main__":
    # Test the pose estimator
    download_pose_weights()
    
    estimator = YOLOv7PoseEstimator()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        poses = estimator.detect_poses(frame)
        result_frame = estimator.draw_poses(frame, poses)
        
        cv2.imshow("YOLOv7 Pose Estimation", result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
