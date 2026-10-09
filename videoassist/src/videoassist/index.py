"""Multimodal index: one embedding row per (segment, modality). numpy now; FAISS is the drop-in at scale."""
import numpy as np
MODS = ("visual", "speech", "ocr")

class Index:
    def __init__(self, emb):
        self.emb, self.segs, self.mats = emb, {}, {}

    def add_video(self, video_id, segs):
        def row(s, m):
            txt = " ".join(e.text for e in s.evidence if e.modality == m)
            return self.emb.one(txt) if txt else np.zeros(self.emb.dim, np.float32)
        self.segs[video_id] = segs
        self.mats[video_id] = {m: np.stack([row(s, m) for s in segs]) for m in MODS}

    def search(self, video_id, text, modalities, k):
        q = self.emb.one(text)
        sc = sum(self.mats[video_id][m] @ q for m in modalities) / len(modalities)
        return [(self.segs[video_id][i], float(sc[i])) for i in np.argsort(-sc)[:k]]
