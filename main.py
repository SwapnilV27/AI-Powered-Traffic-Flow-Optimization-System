"""AI-Powered Traffic Flow Optimization System - command line entry point.

Examples (run from this folder):
  python main.py simulate                       # compare fixed vs adaptive signals
  python main.py train                          # train LSTM congestion predictor
  python main.py extract data/videos/traffic.mp4
  python main.py detect  data/videos/traffic.mp4 --save-video
"""
import argparse
import config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("extract", help="extract frames from a video")
    s.add_argument("video")
    s = sub.add_parser("detect", help="detect vehicles, count per approach, classify density")
    s.add_argument("video")
    s.add_argument("--save-video", action="store_true")
    s.add_argument("--show", action="store_true")
    s = sub.add_parser("train", help="train the LSTM congestion predictor")
    s.add_argument("--csv", help="optional counts CSV from 'detect' (default: synthetic demand)")
    s.add_argument("--epochs", type=int, default=15)
    s = sub.add_parser("simulate", help="run the signal-control simulation and evaluation")
    s.add_argument("--seeds", type=int, default=5)

    a = p.parse_args()
    if a.cmd == "extract":
        from src.extract_frames import extract_frames
        from src.preprocess import preprocess_folder
        out = extract_frames(a.video)
        preprocess_folder(out, str(out) + "_processed")
    elif a.cmd == "detect":
        from src.detector import process_video
        process_video(a.video, save_video=a.save_video, show=a.show)
    elif a.cmd == "train":
        from src.predictor import CongestionPredictor
        pred = CongestionPredictor()
        if a.csv:
            import pandas as pd
            df = pd.read_csv(a.csv)
            pred.fit(binned_series=df[list(config.ZONES)[:4]].to_numpy(), epochs=a.epochs)
        else:
            from src.simulation import generate_arrivals
            pred.fit(per_second_arrivals=generate_arrivals(duration=20000, seed=123), epochs=a.epochs)
        pred.save()
    elif a.cmd == "simulate":
        from src.evaluate import run_evaluation
        run_evaluation(seeds=range(a.seeds))


if __name__ == "__main__":
    main()
