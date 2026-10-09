import numpy as np
def tiou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0
def overlaps(a, b): return min(a[1], b[1]) > max(a[0], b[0])
def pct(xs, p): return float(np.percentile(xs, p)) if len(xs) else 0.0
