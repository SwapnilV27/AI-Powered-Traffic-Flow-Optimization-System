"""Traffic density classification from vehicle counts."""
import config


def classify_density(count, thresholds=config.DENSITY_THRESHOLDS):
    low, high = thresholds
    if count < low:
        return "LOW"
    if count < high:
        return "MEDIUM"
    return "HIGH"
