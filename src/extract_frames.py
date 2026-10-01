"""Step 3: extract frames from traffic video."""
from pathlib import Path
import cv2
import config


def extract_frames(video_path, out_dir=None, step=config.FRAME_STEP):
    video_path = Path(video_path)
    out_dir = Path(out_dir or config.FRAMES_DIR / video_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    i = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            cv2.imwrite(str(out_dir / f"frame_{i:06d}.jpg"), frame)
            saved += 1
        i += 1
    cap.release()
    print(f"Saved {saved} frames from {video_path.name} -> {out_dir}")
    return out_dir
