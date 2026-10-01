"""Step 7: signal timing logic.

Two phases: 0 = North/South green, 1 = East/West green.
Controllers receive a state dict and return the phase they want next.
"""
import config

PHASE_APPROACHES = {0: [0, 1], 1: [2, 3]}   # indices into [N, S, E, W]


class FixedTimeController:
    name = "Fixed-time"

    def __init__(self, green=config.FIXED_GREEN_S):
        self.green = green

    def decide(self, s):
        return 1 - s["phase"] if s["elapsed"] >= self.green else s["phase"]


class AdaptiveController:
    """Queue-pressure controller; optionally adds predicted arrivals."""

    def __init__(self, predictor=None, name="Adaptive", min_green=config.MIN_GREEN_S,
                 max_green=config.MAX_GREEN_S, ratio=1.3, horizon=20, refresh=5):
        self.predictor, self.name = predictor, name
        self.min_green, self.max_green = min_green, max_green
        self.ratio, self.horizon, self.refresh = ratio, horizon, refresh
        self._rates, self._last_t = None, -10**9

    def _pressure(self, s, phase):
        idx = PHASE_APPROACHES[phase]
        p = sum(s["queues"][i] for i in idx)
        if self.predictor is not None:
            if s["t"] - self._last_t >= self.refresh:
                self._rates = self.predictor.predict_rate(s["history"])
                self._last_t = s["t"]
            p += self.horizon * sum(self._rates[i] for i in idx)
        return p

    def decide(self, s):
        cur, other = s["phase"], 1 - s["phase"]
        if s["elapsed"] < self.min_green:
            return cur
        if s["elapsed"] >= self.max_green:
            return other
        p_cur, p_other = self._pressure(s, cur), self._pressure(s, other)
        cur_empty = sum(s["queues"][i] for i in PHASE_APPROACHES[cur]) == 0
        if p_other > 0 and (cur_empty or p_other > p_cur * self.ratio):
            return other
        return cur
