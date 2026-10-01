"""Step 6: congestion prediction.

CongestionPredictor - small TensorFlow LSTM forecasting next-interval
vehicle counts per approach (N, S, E, W).
EMAPredictor        - dependency-free fallback (exponential moving average).
Both expose predict_rate(history) -> per-second arrival rate for 4 approaches.
"""
import numpy as np
import config


class EMAPredictor:
    def __init__(self, alpha=0.05, lookback=120):
        self.alpha, self.lookback = alpha, lookback

    def predict_rate(self, history):
        h = np.asarray(history[-self.lookback:], dtype=float)
        if len(h) == 0:
            return np.zeros(4)
        ema = h[0].copy()
        for row in h[1:]:
            ema = self.alpha * row + (1 - self.alpha) * ema
        return ema


class CongestionPredictor:
    def __init__(self, bin_s=5, window=12):
        self.bin_s, self.window = bin_s, window
        self.scale = 10.0
        self.model = None

    def _bin(self, arrivals):
        a = np.asarray(arrivals, dtype=float)
        n = len(a) // self.bin_s
        return a[: n * self.bin_s].reshape(n, self.bin_s, -1).sum(axis=1)

    def _windows(self, series):
        X = np.stack([series[i:i + self.window] for i in range(len(series) - self.window)])
        y = series[self.window:]
        return X / self.scale, y / self.scale

    def build(self, n_zones=4):
        import tensorflow as tf
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.window, n_zones)),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(n_zones),
        ])
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return self.model

    def fit(self, per_second_arrivals=None, binned_series=None, epochs=15):
        series = np.asarray(binned_series, float) if binned_series is not None else self._bin(per_second_arrivals)
        X, y = self._windows(series)
        if self.model is None:
            self.build(series.shape[1])
        split = int(len(X) * 0.8)
        return self.model.fit(X[:split], y[:split], validation_data=(X[split:], y[split:]),
                              epochs=epochs, batch_size=32, verbose=2)

    def predict_rate(self, history):
        h = np.asarray(history, dtype=float)
        series = self._bin(h) if len(h) >= self.bin_s else np.zeros((0, 4))
        if len(series) < self.window or self.model is None:
            return h.mean(axis=0) if len(h) else np.zeros(4)
        x = series[-self.window:][None] / self.scale
        pred = self.model.predict(x, verbose=0)[0] * self.scale  # vehicles per bin
        return np.clip(pred, 0, None) / self.bin_s

    def save(self, path=config.MODEL_DIR / "congestion_lstm.keras"):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        print(f"Saved model -> {path}")

    @classmethod
    def load(cls, path=config.MODEL_DIR / "congestion_lstm.keras"):
        import tensorflow as tf
        obj = cls()
        obj.model = tf.keras.models.load_model(path)
        return obj
