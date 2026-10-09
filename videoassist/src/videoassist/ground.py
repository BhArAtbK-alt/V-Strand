"""Fine temporal grounding: keep evidence in the segment that matches the query; interval = their span."""
def ground(seg, text, emb, modalities, rel_thr=0.6, min_abs=0.15):
    q = emb.one(text)
    sc = [(e, float(emb.one(e.text) @ q)) for e in seg.evidence if e.modality in modalities]
    if not sc:
        return seg.start, seg.end, []
    best = max(s for _, s in sc)
    if best < min_abs:                      # nothing matches: fall back to the segment; verification will decide
        return seg.start, seg.end, []
    keep = [e for e, s in sc if s >= max(rel_thr * best, min_abs)]
    return min(e.t0 for e in keep), max(e.t1 for e in keep), keep
