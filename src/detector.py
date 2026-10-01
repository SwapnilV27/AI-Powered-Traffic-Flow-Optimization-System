"""Step 5: YOLO vehicle detection + per-approach counting + density labels."""
from pathlib import Path
import cv2
import pandas as pd
import config
from src.preprocess import preprocess_frame
from src.density import classify_density


def _zone_of(cx, cy, w, h):
    for name, (x1, y1, x2, y2) in config.ZONES.items():
        if x1 * w <= cx <= x2 * w and y1 * h <= cy <= y2 * h:
            return name
    return None


def process_video(video_path, out_csv=None, save_video=False, show=False):
    from ultralytics import YOLO  # lazy import so other modules work without it

    video_path = Path(video_path)
    out_csv = Path(out_csv or config.OUTPUT_DIR / f"{video_path.stem}_counts.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(config.YOLO_WEIGHTS)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = config.IMG_SIZE
    writer = None
    if save_video:
        writer = cv2.VideoWriter(str(out_csv.with_suffix(".mp4")),
                                 cv2.VideoWriter_fourcc(*"mp4v"), max(fps / config.FRAME_STEP, 1), (W, H))

    zone_names = list(config.ZONES)
    samples, rows, i = [], [], 0
    next_flush = config.COUNT_INTERVAL_S
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        if i % config.FRAME_STEP == 0:
            frame = preprocess_frame(frame)
            res = model.predict(frame, conf=config.CONF_THRESHOLD,
                                classes=list(config.VEHICLE_CLASSES), verbose=False)[0]
            counts = dict.fromkeys(zone_names, 0)
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                z = _zone_of((x1 + x2) / 2, (y1 + y2) / 2, W, H)
                if z:
                    counts[z] += 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            samples.append(counts)
            for name, (zx1, zy1, zx2, zy2) in config.ZONES.items():
                cv2.rectangle(frame, (int(zx1 * W), int(zy1 * H)), (int(zx2 * W), int(zy2 * H)), (255, 128, 0), 1)
                cv2.putText(frame, f"{name}:{counts[name]} {classify_density(counts[name])}",
                            (int(zx1 * W) + 4, int(zy1 * H) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            if writer:
                writer.write(frame)
            if show:
                cv2.imshow("Traffic", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        if t >= next_flush and samples:
            row = {"time_s": round(next_flush, 1)}
            for z in zone_names:
                avg = sum(s[z] for s in samples) / len(samples)
                row[z] = round(avg, 2)
                row[f"{z}_density"] = classify_density(avg)
            rows.append(row)
            samples = []
            next_flush += config.COUNT_INTERVAL_S
        i += 1
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {len(rows)} intervals -> {out_csv}")
    return out_csv
