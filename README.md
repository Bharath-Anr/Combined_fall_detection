config.json
   ->
pose_detector.py
   ->
(main pose + keypoint extraction)
   ->
main_pipeline.py
   ├─ rope loading & interpolation
   ├─ IoU-based tracking
   ├─ fall logic
   ├─ visualization
   └─ logging & snapshots

---

## 1. `main_pipeline.py`

### Purpose

This is the **main entry point** of the application.  
It orchestrates the entire video-processing pipeline including configuration loading, video ingestion, tracking, fall detection, visualization, and logging.

### Key Responsibilities

- Load and parse `config.json`
- Initialize models **once** for efficiency
- Read video streams using OpenCV
- Perform person detection per frame
- Maintain person identity using **IoU-based tracking**
- Invoke pose estimation on cropped person regions
- Apply fall detection logic based on:
  - Rope position
  - Body reference point
  - Temporal confirmation
- Draw visual overlays:
  - Bounding boxes (optional)
  - Rope lines (optional)
  - Body reference points
  - Fall alerts
- Save:
  - Snapshot images on fall detection
  - CSV logs per camera

### Important Details

- Entirely **configuration-driven**
- No hardcoded thresholds or paths
- Supports **multiple cameras**
- Acts as the **integration layer** for future extensions (alerts, RTSP, APIs)

---

## 2. `pose_detector.py`

### Purpose

Encapsulates **human pose estimation logic** using **YOLOv8 Pose**, exposing a clean interface to the main pipeline.

### Key Responsibilities

- Load YOLOv8 pose model
- Run pose inference on cropped person images
- Extract COCO-format keypoints
- Filter unreliable keypoints using confidence thresholds
- Compute derived reference points:
  - `ankle_mid`
  - `hip_mid`
  - `body_mid`

### Important Details

- Pose inference is **camera-aware**
- Separates:
  - Pose model confidence
  - Individual keypoint confidence
- Returns `None` when pose is unreliable to avoid false fall detection
- Keeps pose logic isolated from fall logic

---

## 3. `config.json`

### Purpose

Acts as the **single source of truth** for the entire system.  
All tuning and deployment changes should be made here.

### Key Responsibilities

- Define model paths (person detector, pose model)
- Configure **per-camera**:
  - Person confidence thresholds
  - Pose confidence thresholds
  - Keypoint confidence thresholds
  - IoU thresholds
  - Frame skip values
  - Fall confirmation frames
- Store rope coordinates for each camera
- Control visualization toggles:
  - Bounding boxes
  - Rope drawing
- Define input video sources
- Configure output locations (snapshots, CSV logs)

### Important Details

- Supports **multiple cameras**
- Each camera can have **multiple ropes**
- Enables easy tuning without modifying source code

---

## Fall Detection Logic (High Level)

- Person must first be **initialized above a rope**
- A fall is detected when:
  - Body reference point crosses below the rope
  - Condition persists for configurable frames
- Uses pose-based reference when available
- Falls back to bounding-box geometry when pose is unreliable

---

## Output Artifacts

### Snapshots

Saved only when a fall is detected:

