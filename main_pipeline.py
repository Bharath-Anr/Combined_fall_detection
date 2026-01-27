"""
Main Pipeline for Fall Detection System
Combines: rope_utils, fall_logic, and process_camera functionality
"""

import os
import sys
import json
import csv
import argparse
from datetime import datetime
import time
import cv2
import numpy as np
from typing import List, Optional, Tuple
from ultralytics import YOLO
import math

# Import pose detection
import pose_detector
from pose_detector import detect_keypoints_on_crop

# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def load_config(path=None):
    """Load configuration from JSON file"""
    if path is None:
        path = DEFAULT_CONFIG_PATH
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    
    print(f"[main_pipeline] Loading config from: {path}")
    with open(path, "r") as f:
        return json.load(f)
    
def get_bool_config(key, default=True):
    val = config.get(key, default)
    if isinstance(val, str):
        return val.lower() == "true"
    return bool(val)
        
# Config will be loaded in main() after parsing CLI args
config = None

# =============================================================================
# HELPER: Get camera-specific config values
# =============================================================================
def get_camera_config(param_name, camera_id, default=None):
    """
    Get camera-specific configuration value.
    Handles both dict format (camera-specific) and single value format.
    """
    param = config.get(param_name, default)
    
    # If param is a dict, get camera-specific value
    if isinstance(param, dict):
        return param.get(camera_id, default)
    
    # If param is a single value, return it
    return param if param is not None else default

# =============================================================================
# ROPE UTILITIES SECTION
# =============================================================================

def load_ropes_from_config(config, camera_id):
    """Load rope polylines from config for specified camera"""
    import ast
    
    # Try "ropes" first (lowercase), then "ROPES" (uppercase)
    ropes_section = config.get("ropes", config.get("ROPES", {}))
    camera_ropes = ropes_section.get(camera_id, [])
    
    rope_polylines = []
    for rope in camera_ropes:
        # If rope is a string, parse it
        if isinstance(rope, str):
            try:
                # Remove extra quotes and parse as Python literal
                rope = rope.strip().strip('"').strip("'")
                rope = ast.literal_eval(rope)
            except (ValueError, SyntaxError) as e:
                print(f"[load_ropes] Warning: Could not parse rope string for {camera_id}: {e}")
                continue
        
        # Handle both list and tuple formats
        try:
            arr = np.array(rope, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) >= 2:
                rope_polylines.append(arr)
        except (ValueError, TypeError) as e:
            print(f"[load_ropes] Warning: Invalid rope format for {camera_id}: {e}")
            continue
            
    return rope_polylines

def interpolate_rope_y(x: float, rope_polyline: np.ndarray) -> float:
    """
    Interpolate rope y at given x.
    
    Args:
        x: x-coordinate to interpolate at
        rope_polyline: numpy array of rope points
    
    Returns:
        Interpolated y-coordinate
    """
    if rope_polyline is None or len(rope_polyline) == 0:
        raise ValueError("rope_polyline is empty or invalid")
    
    pts = np.asarray(rope_polyline, dtype=np.float32)
    
    xs = pts[:, 0]
    ys = pts[:, 1]
    
    # Sort by x coordinate
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    
    # Handle edge cases
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    
    # Find interpolation segment
    idx = np.searchsorted(xs, x)
    
    x1, y1 = xs[idx - 1], ys[idx - 1]
    x2, y2 = xs[idx], ys[idx]
    
    # Handle vertical segments
    if np.isclose(x1, x2):
        return float((y1 + y2) * 0.5)
    
    # Linear interpolation
    t = (x - x1) / (x2 - x1)
    return float(y1 + t * (y2 - y1))

def assign_rope_below_point(
    point_x: float,
    point_y: float,
    rope_polylines: list[np.ndarray],
    min_vertical_gap: float = 0.0
) -> Optional[int]:
    """
    Returns index of nearest rope below (point_x, point_y).
    
    Args:
        point_x: x-coordinate of point
        point_y: y-coordinate of point
        rope_polylines: list of rope polyline arrays
        min_vertical_gap: minimum gap required below point
    
    Returns:
        Index of nearest rope below point, or None if no rope found
    """
    if not rope_polylines:
        return None
    
    best_idx = None
    best_gap = float("inf")
    
    for idx, rope in enumerate(rope_polylines):
        try:
            rope_y = interpolate_rope_y(point_x, rope)
            gap = rope_y - point_y
            
            if gap > min_vertical_gap and gap < best_gap:
                best_gap = gap
                best_idx = idx
        except (ValueError, IndexError):
            continue
    
    return best_idx


def assign_rope_nearest_point(
    point_x: float,
    point_y: float,
    rope_polylines: list[np.ndarray]) -> Optional[int]:
    """
    Returns index of nearest rope (above or below).
    
    Args:
        point_x: x-coordinate of point
        point_y: y-coordinate of point
        rope_polylines: list of rope polyline arrays
    
    Returns:
        Index of nearest rope, or None if no ropes found
    """
    if not rope_polylines:
        return None
    
    best_idx = None
    best_gap = float("inf")
    
    for idx, rope in enumerate(rope_polylines):
        try:
            rope_y = interpolate_rope_y(point_x, rope)
            gap = abs(rope_y - point_y)  # Use absolute distance
            
            if gap < best_gap:
                best_gap = gap
                best_idx = idx
        except (ValueError, IndexError):
            continue
    
    return best_idx

def get_rope_bounds(rope_polyline: np.ndarray) -> Tuple[float, float]:
    """
    Get x-coordinate bounds of a rope.
    
    Args:
        rope_polyline: numpy array of rope points [[x1, y1], [x2, y2], ...]
    
    Returns:
        Tuple of (min_x, max_x)
    """
    if rope_polyline is None or len(rope_polyline) == 0:
        return None, None
    
    xs = rope_polyline[:, 0]
    return float(np.min(xs)), float(np.max(xs))


def calculate_horizontal_distance_to_rope(
    point_x: float,
    rope_polyline: np.ndarray
) -> float:
    """
    Calculate minimum horizontal distance from point to rope.
    
    Args:
        point_x: x-coordinate of the point
        rope_polyline: numpy array of rope points
    
    Returns:
        Minimum horizontal distance (0 if within bounds)
    """
    min_x, max_x = get_rope_bounds(rope_polyline)
    
    if min_x is None or max_x is None:
        return float('inf')
    
    if min_x <= point_x <= max_x:
        return 0.0
    elif point_x < min_x:
        return min_x - point_x
    else:
        return point_x - max_x


def get_average_rope_height(rope_polyline: np.ndarray) -> float:
    """
    Get average y-coordinate (height) of a rope.
    
    Args:
        rope_polyline: numpy array of rope points
    
    Returns:
        Average y-coordinate
    """
    if rope_polyline is None or len(rope_polyline) == 0:
        return float('inf')
    
    return float(np.mean(rope_polyline[:, 1]))


def assign_rope_nearest_point_improved(
    point_x: float,
    point_y: float,
    rope_polylines: list,
    horizontal_margin: float = 80.0
) -> Optional[int]:
    """
    Find nearest rope (above or below) with horizontal priority.
    Better for initialization when you don't know if person is above or below rope.
    
    Strategy:
    1. Filter ropes by horizontal proximity (within margin)
    2. Among horizontally close ropes, find nearest by vertical distance
    3. Fall back to nearest overall if no ropes in margin
    
    Args:
        point_x: x-coordinate of person
        point_y: y-coordinate of person
        rope_polylines: list of rope polyline arrays
        horizontal_margin: acceptable horizontal distance in pixels (typical: 50-100)
    
    Returns:
        Index of nearest rope, or None if no ropes found
    """
    if not rope_polylines:
        return None
    
    # Phase 1: Filter by horizontal proximity
    horizontally_close = []
    for idx, rope in enumerate(rope_polylines):
        h_dist = calculate_horizontal_distance_to_rope(point_x, rope)
        if h_dist <= horizontal_margin:
            horizontally_close.append(idx)
    
    # Phase 2: Among horizontally close ropes, find nearest vertically
    if horizontally_close:
        best_idx = None
        best_v_dist = float("inf")
        
        for idx in horizontally_close:
            try:
                rope_y = interpolate_rope_y(point_x, rope_polylines[idx])
                v_dist = abs(rope_y - point_y)
                
                if v_dist < best_v_dist:
                    best_v_dist = v_dist
                    best_idx = idx
            except (ValueError, IndexError):
                continue
        
        return best_idx
    
    # Phase 3: Fallback - find nearest overall
    best_idx = None
    best_combined_dist = float("inf")
    
    for idx, rope in enumerate(rope_polylines):
        try:
            h_dist = calculate_horizontal_distance_to_rope(point_x, rope)
            rope_y = interpolate_rope_y(point_x, rope)
            v_dist = abs(rope_y - point_y)
            
            # Weighted distance: horizontal matters more
            combined = h_dist * 2.0 + v_dist
            
            if combined < best_combined_dist:
                best_combined_dist = combined
                best_idx = idx
        except (ValueError, IndexError):
            continue
    
    return best_idx


def log_rope_assignment(tid, point_x, point_y, rope_polylines, assigned_rope_idx, verbose=False):
    """
    Log rope assignment details for debugging.
    
    Args:
        tid: track ID
        point_x: point x-coordinate
        point_y: point y-coordinate
        rope_polylines: list of rope polylines
        assigned_rope_idx: assigned rope index
        verbose: if True, print all ropes; if False, print only assigned
    """
    if assigned_rope_idx is None:
        print(f"[ROPE_ASSIGN] Person {tid} at ({int(point_x)}, {int(point_y)}) - NO ROPE ASSIGNED")
        return
    
    try:
        h_dist = calculate_horizontal_distance_to_rope(point_x, rope_polylines[assigned_rope_idx])
        rope_y = interpolate_rope_y(point_x, rope_polylines[assigned_rope_idx])
        v_dist = rope_y - point_y
        avg_height = get_average_rope_height(rope_polylines[assigned_rope_idx])
        
        print(f"[ROPE_ASSIGN] Person {tid} at ({int(point_x)}, {int(point_y)})")
        print(f"  ✓ Assigned: Rope {assigned_rope_idx}")
        print(f"    h_dist={h_dist:.1f}px, v_dist={v_dist:.1f}px, avg_height={avg_height:.1f}")
    except Exception as e:
        print(f"[ROPE_ASSIGN] Error logging assignment: {e}")


def validate_rope_assignment(
    point_x: float,
    point_y: float,
    rope_polylines: list,
    assigned_rope_idx: Optional[int],
    horizontal_margin: float = 80.0
) -> Tuple[bool, str]:
    """
    Validate if a rope assignment is reasonable.
    
    Returns:
        Tuple of (is_valid, reason_string)
    """
    if assigned_rope_idx is None:
        return False, "No rope assigned"
    
    if assigned_rope_idx >= len(rope_polylines):
        return False, f"Rope index {assigned_rope_idx} out of bounds"
    
    try:
        h_dist = calculate_horizontal_distance_to_rope(point_x, rope_polylines[assigned_rope_idx])
        
        if h_dist > horizontal_margin * 1.5:  # Allow 1.5x margin as threshold
            return False, f"Rope too far horizontally ({h_dist:.1f}px > {horizontal_margin*1.5:.1f}px)"
        
        return True, f"Valid assignment (h_dist={h_dist:.1f}px)"
    except (ValueError, IndexError) as e:
        return False, f"Error validating: {str(e)}"

# =============================================================================
# FALL LOGIC SECTION
# =============================================================================

# Fixed constants (not camera-specific)
def get_fixed_constants():
    """Get fixed constants from config"""
    return {
        'ABOVE_ROPE_MARGIN_PX': config.get("ABOVE_ROPE_MARGIN_PX", 15),
        'FALL_THRESHOLD_PX': config.get("FALL_THRESHOLD_PX", 10),
        'MAX_MISSED_FRAMES': config.get("MAX_MISSED_FRAMES", 60),
        'SNAPSHOT_ROOT': config.get("SNAPSHOT_FOLDER", "fall_snapshots"),
        'CSV_PREFIX': config.get("CSV_PREFIX", "falls_"),
        'POSE_MIN_HEIGHT_PX': 80
    }

# These will be set after config is loaded
ABOVE_ROPE_MARGIN_PX = 15
FALL_THRESHOLD_PX = 10
MAX_MISSED_FRAMES = 60
SNAPSHOT_ROOT = "fall_snapshots"
CSV_PREFIX = "falls_"
POSE_MIN_HEIGHT_PX = 80

# Global tracking ID
NEXT_TRACK_ID = 0

# Model will be loaded after config is loaded
people_model = None

def initialize_models():
    """Initialize YOLO models after config is loaded"""
    global people_model, ABOVE_ROPE_MARGIN_PX, FALL_THRESHOLD_PX, MAX_MISSED_FRAMES, SNAPSHOT_ROOT, CSV_PREFIX
    global NEED_BOUNDING_BOX, NEED_ROPE_IN_FRAME

    # Load constants from config
    constants = get_fixed_constants()
    ABOVE_ROPE_MARGIN_PX = constants['ABOVE_ROPE_MARGIN_PX']
    FALL_THRESHOLD_PX = constants['FALL_THRESHOLD_PX']
    MAX_MISSED_FRAMES = constants['MAX_MISSED_FRAMES']
    SNAPSHOT_ROOT = constants['SNAPSHOT_ROOT']
    CSV_PREFIX = constants['CSV_PREFIX']
    NEED_BOUNDING_BOX = get_bool_config("need_bounding_box", True)
    NEED_ROPE_IN_FRAME = get_bool_config("need_rope_in_frame", True)

    # Load person detection model
    PERSON_MODEL_PATH = config.get("PERSON_MODEL")
    if PERSON_MODEL_PATH is None:
        raise ValueError("PERSON_MODEL not defined in config.json")
    
    people_model = YOLO(PERSON_MODEL_PATH)
    print(f"[main_pipeline] Loaded person detection model: {PERSON_MODEL_PATH}")
    
    # Initialize pose detector
    pose_detector.initialize_pose_model(config)

def detect_people(frame, camera_id):
    """Detect people in frame using YOLO"""
    person_conf = get_camera_config("person_conf_threshold", camera_id, 0.35)
    
    results = people_model(frame, conf=person_conf)[0]
    people = []
    
    if results.boxes is None:
        return people
    
    for box in results.boxes:
        if int(box.cls[0]) != 0:  # Only person class
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        people.append({
            "bbox": (x1, y1, x2, y2),
            "conf": float(box.conf[0])
        })
    
    return people

def crop_person(frame, bbox, pad_ratio=0.15):
    """Crop person from frame with padding"""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    
    bw = x2 - x1
    bh = y2 - y1
    
    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    
    cx1 = max(0, x1 - px)
    cy1 = max(0, y1 - py)
    cx2 = min(w, x2 + px)
    cy2 = min(h, y2 + py)
    
    return frame[cy1:cy2, cx1:cx2], (cx1, cy1)
def bbox_center(b):
    return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)

def iou(a, b):
    """Calculate Intersection over Union"""
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(areaA + areaB - inter)



def save_snapshot(frame, camera_id, track_id, tracked_persons):
    """Save snapshot with correct bbox + body midpoint"""
    
    state = tracked_persons.get(track_id)
    if state is None:
        return None

    bbox = state.get("last_bbox")
    body_mid = state.get("body_mid")  # we will store this

    if bbox is None:
        return None

    snap = frame.copy()

    x1, y1, x2, y2 = map(int, bbox)
    h, w = snap.shape[:2]

    # Clamp
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    # Draw bbox
    cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Draw body midpoint
    if body_mid is not None:
        cv2.circle(
            snap,
            (int(body_mid[0]), int(body_mid[1])),
            6,
            (0, 255, 255),
            -1
        )

    date = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")

    folder = os.path.join(SNAPSHOT_ROOT, date, camera_id)
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, f"{time_str}_person{track_id}.jpg")
    cv2.imwrite(path, snap)

    return path

def log_fall(camera_id, track_id, rope_id, snapshot):
    """Log fall event to CSV"""
    csv_path = f"{CSV_PREFIX}{camera_id}.csv"
    write_header = not os.path.exists(csv_path)
    
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["date", "time", "camera", "person", "rope", "snapshot"])
        now = datetime.now()
        w.writerow([
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            camera_id,
            track_id,
            rope_id,
            snapshot
        ])

def classify_rope_side(mid_y, rope_y, margin):
    if mid_y < rope_y - margin:
        return "ABOVE"
    elif mid_y > rope_y + margin:
        return "BELOW"
    else:
        return "NEAR"

def create_person_state():
    return {
        "rope_id": None,
        "initialized_above": False,
        "rope_side": None,
        "prev_rope_side": None,
        "below_counter": 0,
        "transition_detected": False,
        "is_fallen": False,
        "fall_logged": False,
        "last_bbox": None,
        "last_seen": 0,
        "body_mid": None,
        "stable_side": None,
        "stabilization_frames": 0,
        "last_confirmed_side": None  # NEW: Track last confirmed stable side
    }


def process_frame_for_falls(frame, camera_id, rope_polylines, tracked_persons, frame_idx, draw_bounding_box=True):
    """Main fall detection processing for a single frame"""
    global NEXT_TRACK_ID
    
    # Get camera-specific parameters
    frame_skip = get_camera_config("frame_skip", camera_id, 1)
    iou_threshold = get_camera_config("iou_threshold", camera_id, 0.15)
    pose_conf = get_camera_config("pose_conf_threshold", camera_id, 0.15)
    fall_confirm_frames = get_camera_config("fall_confirm_frames", camera_id, 2)
    centre_distance_fallback=get_camera_config("centre_distance_fallback",camera_id,10)
    
    # Adjust thresholds based on frame skip
    if frame_skip >= 5:
        iou_threshold = 0.03
    
    fall_persistence_frames = get_camera_config("fall_confirm_frames", camera_id, 1)
    
    output = frame.copy()
    detections = detect_people(frame, camera_id)
    matches = []
    person_conf = get_camera_config("person_conf_threshold", camera_id, 0.35)
    
    # IoU tracking - match detections to existing tracks
    for det in detections:
        if det.get("conf", 0.0) < person_conf:
            continue
        
        bbox = det["bbox"]
        best_id = None
        best_iou = 0.0
        best_dist = float("inf")
        dist_id = None
        
        cx1, cy1 = bbox_center(bbox)
        
        for tid, state in tracked_persons.items():
            if state["last_bbox"] is None:
                continue
            
            prev_bbox = state["last_bbox"]
            
            # IoU matching
            val = iou(bbox, prev_bbox)
            if val > best_iou:
                best_iou = val
                best_id = tid
            
            # Center-distance fallback (critical for frame_skip=25)
            cx2, cy2 = bbox_center(prev_bbox)
            dx = cx1 - cx2
            dy = cy1 - cy2
            dist = math.sqrt(dx*dx + 2.5 * dy*dy)

            #dist = math.hypot(cx1 - cx2, cy1 - cy2)
            if dist < best_dist:
                best_dist = dist
                dist_id = tid

        used_track_ids = set()

        if best_iou >= iou_threshold and best_id not in used_track_ids:
            matches.append((best_id, det))
            used_track_ids.add(best_id)

# CHECK IF BEST_DIST is within permissible limit to consider as same bounding box. To accomodate the high frame_skip
        elif frame_skip >= 15 and best_dist <= centre_distance_fallback  and dist_id not in used_track_ids: 
            matches.append((dist_id, det))
            used_track_ids.add(dist_id)

        else:
            tid = NEXT_TRACK_ID
            NEXT_TRACK_ID += 1
            tracked_persons[tid] = create_person_state()
            matches.append((tid, det))
            used_track_ids.add(tid)

    
    # Update tracks and perform fall detection
    for tid, det in matches:
        state = tracked_persons[tid]

        if not det.get("bbox") or len(det["bbox"]) != 4:
            continue

        state["last_seen"] = frame_idx
        state["last_bbox"] = det["bbox"]
        
        x1, y1, x2, y2 = det["bbox"]
        
        # Get pose keypoints
        crop, offset = crop_person(frame, det["bbox"])
        pose = detect_keypoints_on_crop(
            crop,
            camera_id=camera_id,
            conf=pose_conf
        )
        
        # Fallback for high frame skip
        if pose is None and frame_skip >= 10:
            ankle_mid = ((x1 + x2) // 2, y2)
            body_mid = ((x1 + x2) // 2, (y1 + y2) // 2)
        elif pose is None:
            continue
        else:
            ox, oy = offset
            ankle_mid = (
                pose["ankle_mid"][0] + ox,
                pose["ankle_mid"][1] + oy
            )
            body_mid = (
                pose["body_mid"][0] + ox,
                pose["body_mid"][1] + oy
            )
        
        state["body_mid"] = body_mid
        fall_ref = body_mid

        # Initialization - assign person to rope
        if not state["initialized_above"]:
            # Get camera-specific rope assignment parameters
            rope_config = get_camera_config("rope_assignment", camera_id, {})
            
            # Handle both dict and non-dict formats
            if isinstance(rope_config, dict):
                h_margin = rope_config.get("horizontal_margin", 80)
                enable_debug = rope_config.get("enable_debug_logging", False)
                enable_validation = rope_config.get("enable_validation", False)
            else:
                h_margin = 80
                enable_debug = False
                enable_validation = False
            
            # Find nearest rope with horizontal priority
            rope_id = assign_rope_nearest_point_improved(
                ankle_mid[0],
                ankle_mid[1],
                rope_polylines,
                horizontal_margin=5.0
            )
            if rope_id is not None:
                rope_y = interpolate_rope_y(
                    ankle_mid[0],
                    rope_polylines[rope_id]
                )
                if rope_y <= ankle_mid[1]:
                    print(f"[INIT-WARN] Person {tid}: rope not below feet, skipping init")
                    continue

            if rope_id is None:
                print(f"[INIT] Person {tid}: No rope assigned (all ropes too far)")
                continue
            
            # Debug logging
            if enable_debug:
                log_rope_assignment(tid, ankle_mid[0], ankle_mid[1], rope_polylines, rope_id, verbose=False)
            
            # Validation
            if enable_validation:
                is_valid, reason = validate_rope_assignment(
                    ankle_mid[0], ankle_mid[1], rope_polylines, rope_id, h_margin
                )
                if not is_valid:
                    print(f"[WARN] Person {tid}: {reason}")
            
            rope_y = interpolate_rope_y(
                ankle_mid[0],
                rope_polylines[rope_id]
            )
            
            rope_side = classify_rope_side(
                ankle_mid[1],
                rope_y,
                ABOVE_ROPE_MARGIN_PX)
            
            # Initialization - assign person to rope
            state["initialized_above"] = True
            state["rope_id"] = rope_id
            state["rope_side"] = rope_side
            state["prev_rope_side"] = rope_side
            state["stabilization_frames"] = 1

            # 🔒 CRITICAL: lock safe baseline
            if rope_side in ["ABOVE", "NEAR"]:
                state["last_confirmed_side"] = "ABOVE"
                state["stable_side"] = "ABOVE"

            print(f"[INIT] Person {tid}: rope_id={rope_id}, initial_side={rope_side}, "
                f"confirmed={state['last_confirmed_side']}")

            continue  # Skip fall detection on initialization frame
        
        # Draw reference point
        cv2.circle(
            output,
            (int(fall_ref[0]), int(fall_ref[1])),
            6,
            (255, 0, 255),  # Magenta
            -1
        )
        
        # Fall detection
        rope_y = interpolate_rope_y(
            ankle_mid[0],
            rope_polylines[state["rope_id"]]
        )
        
        # Determine rope side
        rope_side = classify_rope_side(
            fall_ref[1],
            rope_y,
            FALL_THRESHOLD_PX
        )
        
        state["prev_rope_side"] = state["rope_side"]
        state["rope_side"] = rope_side

        print(
        f"[TRACK] frame={frame_idx} id={tid} "
        f"cx={int((x1+x2)/2)} cy={int((y1+y2)/2)} "
        f"side={rope_side} confirmed={state['last_confirmed_side']}")      

        # ===== IMPROVED STABILIZATION LOGIC =====
        # For high frame skip, be more lenient about side changes
        stabilization_threshold = 1 if frame_skip >= 15 else 3
        
        if rope_side == state["stable_side"]:
            # Same side as last stable state
            state["stabilization_frames"] += 1
            
            # Once stable, confirm this as the baseline
            if state["stabilization_frames"] >= stabilization_threshold:
                state["last_confirmed_side"] = state["stable_side"]
        else:
            # Different side - start counting towards new stable side
            state["stabilization_frames"] = 1
            
            # For very high frame skip, consider any side change as potential stabilization
            if frame_skip >= 20:
                # Immediately update if we see a clear change
                if rope_side == "ABOVE":
                    state["last_confirmed_side"] = "ABOVE"
                    state["last_confirmed_side"] = rope_side
                    state["stabilization_frames"] = stabilization_threshold
            elif state["stabilization_frames"] >= stabilization_threshold:
                # Update stable side after threshold
                state["stable_side"] = rope_side
                state["last_confirmed_side"] = rope_side
        
        # ===== FALL DETECTION =====
        # Detect transition: from ABOVE (confirmed) to BELOW (current)
        if (state["last_confirmed_side"] == "ABOVE" 
            and rope_side == "BELOW" ):
            state["transition_detected"] = True
            state["below_counter"] = 1
            print(f"[TRANSITION] Person {tid}: ABOVE->BELOW detected at frame {frame_idx}")
        
        elif rope_side == "BELOW" and state["transition_detected"]:
            state["below_counter"] += 1
        
        elif rope_side != "BELOW":
            # Reset if they go back above
            state["below_counter"] = 0
            state["transition_detected"] = False
        
        # Confirm fall
        if (
            not state["is_fallen"]
            and state["transition_detected"]
            and state["below_counter"] >= fall_persistence_frames
        ):
            state["is_fallen"] = True
            state["last_confirmed_side"] = "BELOW"  # ✅ ONLY HERE
            state["fall_logged"] = True
            snap = save_snapshot(output, camera_id, tid, tracked_persons)
            log_fall(camera_id, tid, state["rope_id"], snap)
            print(f"[FALL DETECTED] Camera={camera_id}, Person={tid}, Rope={state['rope_id']}, "
                  f"below_counter={state['below_counter']}, persist_frames={fall_persistence_frames}")
        
        # Draw bounding box (only if enabled)
        if draw_bounding_box and NEED_BOUNDING_BOX:
            color = (0, 0, 255) if state["is_fallen"] else (255, 255, 0)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"ID {tid} [{state['rope_side']}]"
            cv2.putText(
                output,
                label,
                (x1, max(0, y1 - 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )
            
            if state["is_fallen"]:
                cv2.putText(
                    output,
                    "Fall Detected",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )
    
    # Cleanup old tracks
    for tid in list(tracked_persons.keys()):
        if frame_idx - tracked_persons[tid]["last_seen"] > MAX_MISSED_FRAMES:
            del tracked_persons[tid]
    
    return output, tracked_persons
# =============================================================================
# PROCESS CAMERA SECTION
# =============================================================================

def draw_ropes(frame, rope_polylines):
    if not NEED_ROPE_IN_FRAME:
        return frame

    if not rope_polylines:
        return frame

    out = frame.copy()
    for rope in rope_polylines:
        pts = np.array(rope, np.int32)
        cv2.polylines(out, [pts], False, (0, 200, 0), 2)
        for (x, y) in pts:
            cv2.circle(out, (int(x), int(y)), 3, (0, 200, 0), 2)
    return out

def annotate_fall_text(frame, tracked_persons):
    """Add fall count text to frame"""
    fallen = sum(1 for s in tracked_persons.values() if s.get("is_fallen"))
    text = f"FALLS: {fallen}"
    cv2.putText(
        frame, text, (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (0, 0, 255), 2, cv2.LINE_AA
    )
    return frame

def run_video_loop(video_path, camera_id, display=True, max_frames=None, save_output_video=True, draw_bounding_box=True):
    """Main video processing loop"""
    print(f"[main_pipeline] Running camera_id = {camera_id} on video: {video_path}")
    
    # Get camera-specific frame skip
    frame_skip = get_camera_config("frame_skip", camera_id, 1)
    print(f"[main_pipeline] Frame skip for {camera_id}: {frame_skip}")
    
    # Load ropes
    rope_polylines = load_ropes_from_config(config, camera_id)
    print(f"[main_pipeline] Loaded {len(rope_polylines)} ropes for camera '{camera_id}'")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[main_pipeline] Video: FPS={fps:.2f}, size=({width}x{height}), frames={total_frames}")
    
    # Output video writer
    out_writer = None
    """if save_output_video:
        out_name = f"output_{camera_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(out_name, fourcc, fps, (width, height))
        if out_writer.isOpened():
            print(f"[main_pipeline] Writing output to: {out_name}")
        else:
            out_writer = None
    """
    tracked_persons = {}
    frame_idx = 0
    summary = {"frames": 0, "total_falls": 0}
    fall_events = []  # Track unique fall events
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("[main_pipeline] End of video.")
                break
            
            # Skip frames based on camera-specific frame_skip
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            
            if max_frames is not None and frame_idx >= max_frames:
                print(f"[main_pipeline] Reached max_frames={max_frames}")
                break
            
            # Process frame
            start = time.time()
            annotated_frame, tracked_persons = process_frame_for_falls(
                frame, camera_id, rope_polylines, tracked_persons, frame_idx, draw_bounding_box)
            end = time.time()
            inference_ms = (end - start) * 1000
            
            # Draw annotations
            annotated_frame = draw_ropes(annotated_frame, rope_polylines)
            annotated_frame = annotate_fall_text(annotated_frame, tracked_persons)
            
            # Track unique fall events (only when state changes from not fallen to fallen)
            for tid, state in tracked_persons.items():
                if state.get("is_fallen") and state.get("fall_logged"):
                    # This is a new fall event
                    fall_events.append({
                        "track_id": tid,
                        "frame": frame_idx,
                        "rope_id": state.get("rope_id")
                    })
                    state["fall_logged"] = False  # Reset flag
            
            summary["frames"] += 1
            
            # Display
            if display:
                cv2.imshow(f"Camera-{camera_id}", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[main_pipeline] Quit requested.")
                    break
            
            # Save output
            if out_writer is not None:
                if annotated_frame.shape[1] != width or annotated_frame.shape[0] != height:
                    annotated_frame = cv2.resize(annotated_frame, (width, height))
                out_writer.write(annotated_frame)
            
            frame_idx += 1
    
    finally:
        cap.release()
        if out_writer is not None:
            out_writer.release()
        if display:
            cv2.destroyAllWindows()
    
    summary["total_falls"] = len(fall_events)
    
    print("[main_pipeline] Done.")
    print(f"[main_pipeline] Processed frames: {summary['frames']}")
    print(f"[main_pipeline] Total unique falls detected: {summary['total_falls']}")
    print(f"[main_pipeline] Snapshots saved: {summary['total_falls']}")
    
    if fall_events:
        print("\n[main_pipeline] Fall Events:")
        for i, event in enumerate(fall_events, 1):
            print(f"  {i}. Track ID: {event['track_id']}, Frame: {event['frame']}, Rope: {event['rope_id']}")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point"""
    global config
    
    parser = argparse.ArgumentParser(description="Fall Detection Pipeline")
    parser.add_argument("--config", type=str, default="config.json", 
                       help="Path to config file (default: config.json)")
    parser.add_argument("--video", type=str, help="Path to video file or RTSP URL")
    parser.add_argument("--camera-id", type=str, required=True, 
                       help="Camera ID (e.g., cam1, cam5)")
    parser.add_argument("--no-display", action="store_true", help="Disable display")
    #parser.add_argument("--max-frames", type=int, help="Maximum frames to process")
    #parser.add_argument("--no-save", action="store_true", help="Don't save output video")
    parser.add_argument("--no-bounding-box", action="store_true", help="Don't draw bounding boxes")
    
    args = parser.parse_args()
    
    # Load config from specified file
    config = load_config(args.config)
    
    # Initialize models after config is loaded
    initialize_models()
    
    # Get video source from args or config
    if args.video:
        video_path = args.video
    else:
        # Try to get from camera_sources in config
        camera_sources = config.get("camera_sources", {})
        video_path = camera_sources.get(args.camera_id)
        if video_path is None:
            raise ValueError(f"No video source found for camera '{args.camera_id}'. "
                           "Provide --video argument or add to camera_sources in config.json")
    
    display = not args.no_display
    #save_output = not args.no_save
    draw_bounding_box = not args.no_bounding_box
    
    print(f"[main_pipeline] Starting fall detection for camera: {args.camera_id}")
    print(f"[main_pipeline] Video source: {video_path}")
    
    # Run pipeline
    run_video_loop(
        video_path,
        args.camera_id,
        display=display,
        draw_bounding_box=draw_bounding_box
    )

if __name__ == "__main__":
    main()
