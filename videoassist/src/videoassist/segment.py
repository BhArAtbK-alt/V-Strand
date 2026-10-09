"""Temporal event segmentation: change-point on visual-evidence similarity (+ max length / time gap)."""
from .schema import Segment

def segment_video(video_id, evs, emb, thr=0.35, max_len=30.0, gap=3.0):
    base = sorted([e for e in evs if e.modality == "visual"], key=lambda e: e.t0) or sorted(evs, key=lambda e: e.t0)
    groups = [[base[0]]]
    for e in base[1:]:
        g, prev = groups[-1], groups[-1][-1]
        sim = float(emb.one(prev.text) @ emb.one(e.text))
        if sim < thr or e.t1 - g[0].t0 > max_len or e.t0 - prev.t1 > gap:
            groups.append([e])
        else:
            g.append(e)
    segs = [Segment(video_id, i, g[0].t0, g[-1].t1) for i, g in enumerate(groups)]
    for e in evs:                                   # attach every modality to its segment
        mid = (e.t0 + e.t1) / 2
        s = min(segs, key=lambda s: 0 if s.start <= mid <= s.end else min(abs(mid - s.start), abs(mid - s.end)))
        s.evidence.append(e)
    return segs
