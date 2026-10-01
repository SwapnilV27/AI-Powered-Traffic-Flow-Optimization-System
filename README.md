# AI-Powered Traffic Flow Optimization System (Major Project)

Detects vehicles in traffic-camera video with YOLO, classifies traffic density per approach,
predicts congestion with a TensorFlow LSTM, and adjusts signal timing dynamically. The control
logic is evaluated in a software intersection simulation against a fixed-time signal.

## Pipeline (maps to the project PDF)
| Step | File |
|---|---|
| 1. Define objectives (cut wait time, queue length, idling emissions) | `config.py` |
| 2. Collect video data (Kaggle datasets) | put videos in `data/videos/` |
| 3. Extract frames | `src/extract_frames.py` |
| 4. Preprocess (resize, denoise, normalize) | `src/preprocess.py` |
| 5. Vehicle detection + density classification | `src/detector.py`, `src/density.py` |
| 6. Congestion prediction | `src/predictor.py` |
| 7. Signal timing logic | `src/signal_control.py` |
| 8. Simulation | `src/simulation.py` |
| 9. Evaluation (traffic flow metrics) | `src/evaluate.py` |

## Setup (Windows / VS Code)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Run commands from this folder.

## Usage
```
python main.py simulate                              # works with only numpy/pandas/matplotlib
python main.py train                                 # trains the LSTM (needs tensorflow)
python main.py simulate                              # now uses the trained LSTM automatically
python main.py extract data/videos/traffic.mp4       # frames -> data/frames/
python main.py detect data/videos/traffic.mp4 --save-video
python main.py train --csv outputs/traffic_counts.csv
```
Outputs go to `outputs/` (`simulation_summary.csv`, `comparison.png`, detection CSV/video).

## Datasets
Download from Kaggle (links in the PDF): *Traffic Video Analysis* and *Vehicle Detection* datasets.
Place video files in `data/videos/`.

## Before you use real video
Edit `ZONES` in `config.py` so the N/S/E/W rectangles cover the approaches in your camera view.
Check with `--save-video` and tune `DENSITY_THRESHOLDS` to your scene.

## Notes and limits
- The simulation is a simplified single 4-way intersection with two phases and Poisson arrivals
  with a demand shift halfway through. Results show relative improvement under those assumptions,
  not real-world guarantees. Emission figures are a rough idling proxy.
- `detect` counts vehicles visible per zone (occupancy). `train` without `--csv` learns from
  simulated arrival counts, which is what the simulator's predictive controller expects.
- If TensorFlow is not installed or no trained model exists, the predictive controller falls back
  to a moving-average predictor.
- YOLO weights (`yolov8n.pt`) download automatically on first `detect` run.
