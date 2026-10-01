"""Central settings for the Traffic Flow Optimization project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "data" / "videos"
FRAMES_DIR = ROOT / "data" / "frames"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

# --- Frame extraction / preprocessing ---
FRAME_STEP = 15              # keep every 15th frame
IMG_SIZE = (640, 640)        # resize target (width, height)
BLUR_KERNEL = (3, 3)         # noise reduction

# --- Vehicle detection (YOLO, COCO class ids) ---
YOLO_WEIGHTS = "yolov8n.pt"  # downloaded automatically by ultralytics
CONF_THRESHOLD = 0.35
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Approach zones as fractions of the frame: (x1, y1, x2, y2).
# Adjust these to match your camera view.
ZONES = {
    "N": (0.30, 0.00, 0.70, 0.40),
    "S": (0.30, 0.60, 0.70, 1.00),
    "W": (0.00, 0.30, 0.40, 0.70),
    "E": (0.60, 0.30, 1.00, 0.70),
}
COUNT_INTERVAL_S = 5         # aggregate counts over this many seconds

# --- Density classification (vehicles visible in a zone) ---
DENSITY_THRESHOLDS = (5, 12)  # <5 LOW, 5-11 MEDIUM, >=12 HIGH

# --- Simulation ---
SIM_DURATION_S = 3600
SAT_FLOW = 0.5               # vehicles / second discharged per green approach
LOST_TIME_S = 3              # yellow + all-red on every phase change
FIXED_GREEN_S = 30
MIN_GREEN_S = 10
MAX_GREEN_S = 60
IDLE_CO2_G_PER_S = 0.7       # rough idling emission proxy per queued vehicle
