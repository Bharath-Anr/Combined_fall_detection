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
    
    # Load constants from config
    constants = get_fixed_constants()
    ABOVE_ROPE_MARGIN_PX = constants['ABOVE_ROPE_MARGIN_PX']
    FALL_THRESHOLD_PX = constants['FALL_THRESHOLD_PX']
    MAX_MISSED_FRAMES = constants['MAX_MISSED_FRAMES']
    SNAPSHOT_ROOT = constants['SNAPSHOT_ROOT']
    CSV_PREFIX = constants['CSV_PREFIX']
    
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

def create_person_state():
    """Create new person tracking state"""
    return {
        "rope_id": None,
        "initialized_above": False,
        "above_counter": 0,
        "fall_counter": 0,
        "is_fallen": False,
        "fall_logged": False,  # Track if fall has been logged
        "last_bbox": None,
        "last_seen": 0
    }

def save_snapshot(frame, camera_id, track_id):
    """Save snapshot of person when fall is detected"""
    date = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")
    
    folder = os.path.join(SNAPSHOT_ROOT, date, camera_id)
    os.makedirs(folder, exist_ok=True)
    
    path = os.path.join(folder, f"{time_str}_person{track_id}.jpg")
    cv2.imwrite(path, frame)
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

def process_frame_for_falls(frame, camera_id, rope_polylines, tracked_persons, frame_idx, draw_bounding_box=True):
    """Main fall detection processing for a single frame"""
    global NEXT_TRACK_ID
    
    # Get camera-specific parameters
    frame_skip = get_camera_config("frame_skip", camera_id, 1)
    iou_threshold = get_camera_config("iou_threshold", camera_id, 0.15)
    pose_conf = get_camera_config("pose_conf_threshold", camera_id, 0.15)
    fall_confirm_frames = get_camera_config("fall_confirm_frames", camera_id, 2)
    
    
    # Adjust thresholds based on frame skip
    if frame_skip >= 5:
        iou_threshold = max(0.05, iou_threshold * 0.5)
    
    initial_confirm_frames = max(2, math.ceil(10 / frame_skip))
    fall_persistence_frames = max(fall_confirm_frames, math.ceil(10 / frame_skip))
    
    output = frame.copy()
    detections = detect_people(frame, camera_id)
    matches = []
    
    # IoU tracking - match detections to existing tracks
    for det in detections:
        person_conf = get_camera_config("person_conf_threshold", camera_id, 0.35)
        if det.get("conf", 0.0) < person_conf:
            continue

        bbox = det["bbox"]
        best_id = None
        best_iou = 0.0
        
        for tid, state in tracked_persons.items():
            if state["last_bbox"] is None:
                continue
            val = iou(bbox, state["last_bbox"])
            if val > best_iou:
                best_iou = val
                best_id = tid
        
        if best_iou >= iou_threshold:
            matches.append((best_id, det))
        else:
            tid = NEXT_TRACK_ID
            NEXT_TRACK_ID += 1
            tracked_persons[tid] = create_person_state()
            matches.append((tid, det))
    
    # Update tracks
    for tid, det in matches:
        state = tracked_persons[tid]
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
        pose = detect_keypoints_on_crop(crop, camera_id=camera_id, conf=pose_conf )
        
        # Fallback for high frame skip
        if pose is None and frame_skip >= 5:
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
        
        fall_ref = body_mid
        
        # Initialization - assign person to rope
        if not state["initialized_above"]:
            rope_id = assign_rope_below_point(
                ankle_mid[0],
                ankle_mid[1],
                rope_polylines
            )
            
            if rope_id is None:
                continue
            
            rope_y = interpolate_rope_y(
                ankle_mid[0],
                rope_polylines[rope_id]
            )
            
            if ankle_mid[1] < rope_y - ABOVE_ROPE_MARGIN_PX:
                state["above_counter"] += frame_skip
            else:
                state["fall_counter"] += frame_skip
            
            if frame_skip >= 5 and state["above_counter"] >= frame_skip:
                state["initialized_above"] = True
                state["rope_id"] = rope_id
            elif frame_skip < 5 and state["above_counter"] >= initial_confirm_frames:
                state["initialized_above"] = True
                state["rope_id"] = rope_id
            continue
        
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
        
        if fall_ref[1] > rope_y + FALL_THRESHOLD_PX:
            state["fall_counter"] += 1
        else:
            state["fall_counter"] = 0
        
        # Trigger fall event
        if not state["is_fallen"] and state["fall_counter"] >= fall_persistence_frames:
            state["is_fallen"] = True
            state["fall_logged"] = True  # Mark as logged
            snap = save_snapshot(output, camera_id, tid)
            log_fall(camera_id, tid, state["rope_id"], snap)
            print(f"[FALL DETECTED] Camera: {camera_id}, Person: {tid}, Rope: {state['rope_id']}")
        
        # Draw bounding box (only if enabled)
        if draw_bounding_box:
            color = (0, 0, 255) if state["is_fallen"] else (0, 255, 0)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
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
    """Draw rope polylines on the frame"""
    if not rope_polylines:
        return frame
    out = frame.copy()
    for rope in rope_polylines:
        if rope is None or len(rope) < 2:
            continue
        pts = np.array(rope, np.int32)
        cv2.polylines(out, [pts], False, (0, 200, 0), 2, lineType=cv2.LINE_AA)
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
                frame, camera_id, rope_polylines, tracked_persons, frame_idx, draw_bounding_box
            )
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
