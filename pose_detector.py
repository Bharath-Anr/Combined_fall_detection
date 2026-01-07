"""
Pose Detection Module
Detects human pose keypoints for fall detection
"""

import numpy as np
from ultralytics import YOLO
import cv2
import os
import json

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
# Config will be loaded by main_pipeline.py and passed to functions
config = None
model = None

# --------------------------------------------------------------------
# COCO keypoint indices used by YOLOv8 pose format
# --------------------------------------------------------------------
KP = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16
}

# --------------------------------------------------------------------
# Initialize pose model (called by main_pipeline after config is loaded)
# --------------------------------------------------------------------
def initialize_pose_model(config_dict):
    """
    Initialize the pose detection model
    Must be called before using detect_keypoints_on_crop()
    
    Args:
        config_dict: Configuration dictionary
    """
    global config, model
    
    config = config_dict
    POSE_MODEL_PATH = config.get("POSE_MODEL", "yolov8n-pose.pt")
    model = YOLO(POSE_MODEL_PATH)
    
    print(f"[pose_detector] Loaded pose model: {POSE_MODEL_PATH}")

# --------------------------------------------------------------------
# Get camera-specific or default config value
# --------------------------------------------------------------------
def get_config_value(param_name, camera_id=None, default=None):
    """
    Get configuration value, supporting both camera-specific and global values
    """
    if config is None:
        return default
    
    param = config.get(param_name, default)
    
    # If param is a dict and camera_id provided, get camera-specific value
    if isinstance(param, dict) and camera_id is not None:
        return param.get(camera_id, default)
    
    # If param is a single value or no camera_id, return as is
    return param if param is not None else default

# --------------------------------------------------------------------
# Helper: Extract keypoint safely with confidence check
# --------------------------------------------------------------------
def _get_kp_xy(kp_array, index, conf_threshold=0.3):
    """
    Extracts the (x, y) coordinates of a keypoint only if its confidence
    is above a threshold. Returns None if the keypoint is unreliable.
    
    Args:
        kp_array: shape (17, 3) -> [x, y, confidence]
        index: keypoint index (0-16)
        conf_threshold: minimum confidence required
    
    Returns:
        (x, y) tuple or None
    """
    if index >= len(kp_array):
        return None
    
    x, y, conf = kp_array[index]
    if conf >= conf_threshold:
        return (float(x), float(y))
    return None

# --------------------------------------------------------------------
# Helper: Calculate midpoint of two points
# --------------------------------------------------------------------
def _midpoint(p1, p2):
    """
    Returns midpoint between two points (x1, y1) and (x2, y2).
    If either point is None, returns None.
    """
    if p1 is None or p2 is None:
        return None
    return ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)

# --------------------------------------------------------------------
# Main pose detection function
# --------------------------------------------------------------------
def detect_keypoints_on_crop(crop, camera_id=None, conf=None):
    """
    Runs pose detection on a cropped person image.
    Returns a dict with keypoints needed for fall detection,
    or None if pose is unreliable.
    
    Args:
        crop: cropped image of person (numpy array)
        camera_id: camera identifier for camera-specific config
        conf: pose detection confidence threshold (overrides config)
    
    Returns:
        Dictionary with ankle_mid, hip_mid, body_mid coordinates or None
    """
    if model is None:
        raise RuntimeError("Pose model not initialized. Call initialize_pose_model() first.")
    
    # Get confidence thresholds
    if conf is None:
        conf = get_config_value("pose_conf_threshold", camera_id, 0.25)
    
    kp_conf_threshold = get_config_value("kp_conf_threshold", camera_id, 0.3)
    
    # Run pose detection
    try:
        results = model(crop, conf=conf, verbose=False)[0]
    except Exception as e:
        print(f"[pose_detector] Error running pose detection: {e}")
        return None
    
    if results.keypoints is None or len(results.keypoints) == 0:
        return None
    
    # Get keypoints array
    kp = results.keypoints[0].data[0].cpu().numpy()  # shape (17, 3)
    
    # COCO indices
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    LEFT_HIP = 11
    RIGHT_HIP = 12
    
    # Extract keypoints with confidence check
    left_ankle = _get_kp_xy(kp, LEFT_ANKLE, kp_conf_threshold)
    right_ankle = _get_kp_xy(kp, RIGHT_ANKLE, kp_conf_threshold)
    left_hip = _get_kp_xy(kp, LEFT_HIP, kp_conf_threshold)
    right_hip = _get_kp_xy(kp, RIGHT_HIP, kp_conf_threshold)
    
    # Check if we have minimum required keypoints
    if left_ankle is None and right_ankle is None:
        return None
    if left_hip is None and right_hip is None:
        return None
    
    # Calculate ankle midpoint
    if left_ankle and right_ankle:
        ankle_mid = (
            (left_ankle[0] + right_ankle[0]) * 0.5,
            (left_ankle[1] + right_ankle[1]) * 0.5
        )
    else:
        ankle_mid = left_ankle or right_ankle
    
    # Calculate hip midpoint
    if left_hip and right_hip:
        hip_mid = (
            (left_hip[0] + right_hip[0]) * 0.5,
            (left_hip[1] + right_hip[1]) * 0.5
        )
    else:
        hip_mid = left_hip or right_hip
    
    # Calculate body midpoint (between ankle and hip)
    body_mid = (
        (ankle_mid[0] * 0.6 + hip_mid[0] * 0.4) ,
        (ankle_mid[1] * 0.6 + hip_mid[1] * 0.4)
    ) 
    print(
    f"hip_y={hip_mid[1]:.1f}, "
    f"ankle_y={ankle_mid[1]:.1f}, "
    f"body_y={body_mid[1]:.1f}"
)

    return {
        "ankle_mid": ankle_mid,
        "hip_mid": hip_mid,
        "body_mid": body_mid
    }

# --------------------------------------------------------------------
# Optional: Visualize keypoints (for debugging)
# --------------------------------------------------------------------
def visualize_keypoints(image, pose_result):
    """
    Draw keypoints on image for debugging
    
    Args:
        image: input image
        pose_result: result from detect_keypoints_on_crop()
    
    Returns:
        Image with keypoints drawn
    """
    if pose_result is None:
        return image
    
    img = image.copy()
    
    # Draw ankle midpoint (green)
    if pose_result.get("ankle_mid"):
        x, y = pose_result["ankle_mid"]
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.putText(img, "Ankle", (int(x) + 10, int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw hip midpoint (blue)
    if pose_result.get("hip_mid"):
        x, y = pose_result["hip_mid"]
        cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), -1)
        cv2.putText(img, "Hip", (int(x) + 10, int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Draw body midpoint (red)
    if pose_result.get("body_mid"):
        x, y = pose_result["body_mid"]
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(img, "Body", (int(x) + 10, int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return img