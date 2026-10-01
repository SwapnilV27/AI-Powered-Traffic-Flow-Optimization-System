"""Step 4: resize, denoise and normalize frames."""
from pathlib import Path
import cv2
import numpy as np
import config


def preprocess_frame(frame, size=config.IMG_SIZE, blur=config.BLUR_KERNEL):
    """Resize + Gaussian noise reduction. Returns uint8 BGR image."""
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(frame, blur, 0)


def normalize(frame):
    """Scale to float32 in [0, 1] (for custom TensorFlow models)."""
    return frame.astype(np.float32) / 255.0


def preprocess_folder(in_dir, out_dir):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(in_dir.glob("*.jpg")):
        img = cv2.imread(str(p))
        if img is not None:
            cv2.imwrite(str(out_dir / p.name), preprocess_frame(img))
            n += 1
    print(f"Preprocessed {n} images -> {out_dir}")
