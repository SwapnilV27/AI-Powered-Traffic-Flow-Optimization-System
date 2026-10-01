"""Step 8: software simulation of a 4-way signalised intersection."""
from collections import deque
import numpy as np
import config
from src.signal_control import PHASE_APPROACHES


def generate_arrivals(duration=config.SIM_DURATION_S, seed=0):
    """Poisson arrivals per second for [N, S, E, W] with shifting demand."""
    rng = np.random.default_rng(seed)
    t = np.arange(duration)
    wave = 1 + 0.3 * np.sin(2 * np.pi * t / 900)
    heavy, light = 0.16, 0.07
    first = t < duration // 2          # NS heavy first half, EW heavy second half
    ns = np.where(first, heavy, light) * wave
    ew = np.where(first, light, heavy) * wave
    rates = np.stack([ns, ns, ew, ew], axis=1)
    return rng.poisson(rates)


def simulate(controller, arrivals, sat_flow=config.SAT_FLOW, lost_time=config.LOST_TIME_S):
    T = len(arrivals)
    queues = [deque() for _ in range(4)]
    credit = [0.0] * 4
    phase, elapsed, lost = 0, 0, 0
    waits, q_sum, max_q, departed = [], 0, 0, 0
    history = []
    for t in range(T):
        for a in range(4):
            queues[a].extend([t] * int(arrivals[t, a]))
        history.append(arrivals[t])
        if lost > 0:
            lost -= 1
        else:
            state = {"t": t, "phase": phase, "elapsed": elapsed,
                     "queues": [len(q) for q in queues], "history": history}
            want = controller.decide(state)
            if want != phase:
                phase, elapsed, lost = want, 0, lost_time
            else:
                for a in PHASE_APPROACHES[phase]:
                    credit[a] += sat_flow
                    while credit[a] >= 1 and queues[a]:
                        waits.append(t - queues[a].popleft())
                        credit[a] -= 1
                        departed += 1
                    if not queues[a]:
                        credit[a] = 0.0
                elapsed += 1
        total_q = sum(len(q) for q in queues)
        q_sum += total_q
        max_q = max(max_q, total_q)
    waits.extend(T - ta for q in queues for ta in q)   # still waiting at the end
    return {
        "controller": controller.name,
        "avg_wait_s": float(np.mean(waits)) if waits else 0.0,
        "avg_queue": q_sum / T,
        "max_queue": max_q,
        "throughput": departed,
        "co2_idle_kg": q_sum * config.IDLE_CO2_G_PER_S / 1000,
    }
