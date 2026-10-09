"""Ablations A-E (+E-noV for RQ3) with Recall@K, MRR, tIoU, grounded-answer accuracy, unsupported rate, latency."""
import numpy as np
from .pipeline import VideoAssist
from .schema import Evidence
from .metrics import tiou, overlaps, pct

def evaluate(videos, cfg):
    sysm = VideoAssist(cfg)
    for v in videos:
        sysm.ingest(v["video_id"], [Evidence(**e) for e in v["evidence"]])       # system never sees ground truth
    results = {}
    for name, m in cfg["modes"].items():
        r = dict(r1=[], r3=[], rr=[], iou=[], hit50=[], acc=[], fabst=[], cabst=[], unsup=[], lat=[])
        for v in videos:
            for q in v["gt"]["queries"]:
                a = sysm.query(v["video_id"], q["q"], m["modalities"], m["full"], m["verify"])
                r["lat"].append(a.latency_ms)
                if q["answerable"]:
                    gt = (q["start"], q["end"])
                    rank = next((i + 1 for i, s in enumerate(a.ranked) if overlaps(s, gt)), None)
                    r["r1"].append(rank == 1); r["r3"].append(rank is not None); r["rr"].append(1 / rank if rank else 0)
                    u = 0.0 if a.abstained else tiou((a.start, a.end), gt)
                    r["iou"].append(u); r["hit50"].append(u >= 0.5)
                    r["acc"].append(u >= 0.5 and (not q["key"] or q["key"] in a.answer))
                    r["fabst"].append(a.abstained)
                else:
                    r["cabst"].append(a.abstained)
                if not a.abstained:                      # unsupported = answered an unanswerable query, or zero overlap with truth
                    r["unsup"].append((not q["answerable"]) or not overlaps((a.start, a.end), (q["start"], q["end"])))
        mean = lambda x: round(float(np.mean(x)), 3) if len(x) else None
        results[name] = {"Recall@1": mean(r["r1"]), "Recall@3": mean(r["r3"]), "MRR": mean(r["rr"]),
                         "mean_tIoU": mean(r["iou"]), "tIoU>=0.5": mean(r["hit50"]), "grounded_answer_acc": mean(r["acc"]),
                         "unsupported_rate": mean(r["unsup"]), "false_abstain": mean(r["fabst"]),
                         "correct_abstain": mean(r["cabst"]),
                         "p50_ms": round(pct(r["lat"], 50), 2), "p95_ms": round(pct(r["lat"], 95), 2)}
    return results

def to_markdown(res):
    cols = list(next(iter(res.values())))
    rows = ["| mode | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    rows += [f"| {n} | " + " | ".join(str(v[c]) for c in cols) + " |" for n, v in res.items()]
    return "\n".join(rows)
