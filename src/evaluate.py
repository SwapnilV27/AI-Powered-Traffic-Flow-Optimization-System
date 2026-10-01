"""Step 9: compare controllers over several random demand scenarios."""
import pandas as pd
import config
from src.simulation import generate_arrivals, simulate
from src.signal_control import FixedTimeController, AdaptiveController
from src.predictor import EMAPredictor


def get_predictor():
    """Use the trained LSTM if available, otherwise the EMA fallback."""
    path = config.MODEL_DIR / "congestion_lstm.keras"
    if path.exists():
        try:
            from src.predictor import CongestionPredictor
            print("Using trained LSTM predictor.")
            return CongestionPredictor.load(path)
        except Exception as e:  # TensorFlow missing or model unreadable
            print(f"LSTM unavailable ({e}); using EMA predictor.")
    print("Using EMA predictor.")
    return EMAPredictor()


def run_evaluation(seeds=range(5), plot=True):
    predictor = get_predictor()
    rows = []
    for seed in seeds:
        arr = generate_arrivals(seed=seed)
        for ctrl in (FixedTimeController(),
                     AdaptiveController(name="Adaptive"),
                     AdaptiveController(predictor=predictor, name="Adaptive+Prediction")):
            r = simulate(ctrl, arr)
            r["seed"] = seed
            rows.append(r)
    df = pd.DataFrame(rows)
    summary = df.groupby("controller").mean(numeric_only=True).drop(columns="seed").round(2)
    summary = summary.loc[["Fixed-time", "Adaptive", "Adaptive+Prediction"]]
    base = summary.loc["Fixed-time", "avg_wait_s"]
    summary["wait_reduction_%"] = ((base - summary["avg_wait_s"]) / base * 100).round(1)

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(config.OUTPUT_DIR / "simulation_runs.csv", index=False)
    summary.to_csv(config.OUTPUT_DIR / "simulation_summary.csv")
    print("\n" + summary.to_string() + "\n")
    if plot:
        _plot(summary)
    return summary


def _plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for a, col, title in zip(ax, ["avg_wait_s", "avg_queue", "co2_idle_kg"],
                             ["Average wait (s)", "Average queue (vehicles)", "Idling CO2 proxy (kg)"]):
        a.bar(summary.index, summary[col], color=["#9aa5b1", "#2b8cc4", "#0b3d62"])
        a.set_title(title)
        a.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out = config.OUTPUT_DIR / "comparison.png"
    fig.savefig(out, dpi=150)
    print(f"Saved chart -> {out}")
