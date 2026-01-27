import cv2
import json
import os
import numpy as np

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
WINDOW_NAME = "Draw Ropes (Left-click points, 'n'=new rope, 's'=save, 'r'=reset, 'q'=quit)"

# ---------------------------------------------------------
# State variables
# ---------------------------------------------------------
current_rope = []
all_ropes = []

# ---------------------------------------------------------
# Utility to extract camera_id from filename
# ---------------------------------------------------------
"""def extract_camera_id(video_path):
    filename = os.path.basename(video_path)
    base = filename.split(".")[0]
    # Replace any non-alphanumeric separators (e.g. cam5.4 → cam5_4)
    return base.replace(".", "_")
    return""" 


# ---------------------------------------------------------
# Mouse callback to collect points
# ---------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global current_rope
    if event == cv2.EVENT_LBUTTONDOWN:
        current_rope.append((int(x), int(y)))


# ---------------------------------------------------------
# Save ropes into config.json
# ---------------------------------------------------------
def save_ropes_to_config(camera_id, rope_data):
    # Load config.json
    if not os.path.exists(CONFIG_PATH):
        raise RuntimeError(f"Config file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    # Ensure "ropes" exists
    if "ropes" not in config:
        config["ropes"] = {}

    # Assign new ropes to this camera
    config["ropes"][camera_id] = rope_data

    # Save back
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[SAVED] Ropes stored inside config.json under camera_id '{camera_id}'\n")


# ---------------------------------------------------------
# Main rope drawing function
# ---------------------------------------------------------
def draw_rope_for_camera(video_path,camera_id):
    global current_rope, all_ropes
    current_rope = []
    all_ropes = []

    #camera_id = extract_camera_id(video_path)
    

    print(f"\nCamera ID detected: {camera_id}")

    # Load first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Could not read first frame of the video")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("\n==============================================")
    print(f"Drawing MULTIPLE ropes for CAMERA: {camera_id}")
    print("Instructions:")
    print(" - Left-click to add points to the current rope")
    print(" - Press 'n' to finish current rope and start new rope")
    print(" - Press 's' to save all ropes into config.json")
    print(" - Press 'r' to reset everything")
    print(" - Press 'q' to quit without saving")
    print("==============================================\n")

    while True:
        display = frame.copy()

        # Draw finished ropes (green)
        for rope in all_ropes:
            if len(rope) > 1:
                pts = np.array(rope, np.int32)
                cv2.polylines(display, [pts], False, (0, 255, 0), 2)
            for p in rope:
                cv2.circle(display, p, 4, (0, 255, 0), -1)

        # Draw current rope (blue)
        if len(current_rope) > 1:
            pts = np.array(current_rope, np.int32)
            cv2.polylines(display, [pts], False, (255, 0, 0), 2)

        for p in current_rope:
            cv2.circle(display, p, 4, (0, 0, 255), -1)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(20) & 0xFF

        # Finalize rope
        if key == ord('n'):
            if len(current_rope) >= 2:
                all_ropes.append(current_rope.copy())
                print(f"Finished Rope #{len(all_ropes)}")
                current_rope = []
            else:
                print("Current rope must have at least 2 points.")

        # Save all ropes
        elif key == ord('s'):
            # Finalize current rope if needed
            if len(current_rope) >= 2:
                all_ropes.append(current_rope.copy())

            # Convert to JSON format
            rope_data = [
                [[x, y] for (x, y) in rope]
                for rope in all_ropes
            ]

            # Load config.json
            CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)

            # Insert/replace for this camera
            if "ropes" not in config:
                config["ropes"] = {}
            config["ropes"][camera_id] = rope_data

            # Save config.json
            with open(CONFIG_PATH, "w") as f:
                json.dump(config, f, indent=2)

            print(f"All ropes saved into config.json under camera '{camera_id}'")
            break


        # Reset
        elif key == ord('r'):
            print("Resetting all ropes...")
            current_rope = []
            all_ropes = []

        # Quit without saving
        elif key == ord('q'):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------
if __name__ == "__main__":
    draw_rope_for_camera("videos/IPPTZbeachv2.mp4","Ipptzcamera")
