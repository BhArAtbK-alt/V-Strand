"""VideoAssist: retrieval -> temporal grounding -> relation reasoning -> verification -> grounded answer."""
import time
from .text import Embedder
from .segment import segment_video
from .index import Index, MODS
from .query import parse
from .ground import ground
from .verify import coverage, find_code
from .qa import extractive, llm_answer
from .schema import Answer
from .media import cut_clip, extract_frames

ABSTAIN = "Insufficient evidence in the video to answer this question."

class VideoAssist:
    def __init__(self, cfg):
        self.cfg = cfg
        self.emb = Embedder(cfg["embedding"]["dim"], cfg["embedding"]["backend"])
        self.index, self.media = Index(self.emb), {}

    def ingest(self, video_id, evidence, video_path=None):
        segs = segment_video(video_id, evidence, self.emb, **self.cfg["segmentation"])
        self.index.add_video(video_id, segs)
        if video_path:
            self.media[video_id] = video_path
        return {"video_id": video_id, "segments": len(segs)}

    def query(self, video_id, text, modalities=MODS, full=True, verify=True, k=3, generate=None):
        t0 = time.perf_counter()
        c, plan, segs = self.cfg, parse(text), self.index.segs[video_id]
        mods, rel_label = tuple(modalities), None
        if full and plan.relation:                                   # temporal relation reasoning
            top = self.index.search(video_id, plan.anchor, mods, k)
            aseg, score = top[0]
            a0, a1, a_ev = ground(aseg, plan.anchor, self.emb, mods, **c["grounding"])
            rel_label = f"{plan.relation}({plan.anchor} @ {a0:.0f}-{a1:.0f}s)"
            if plan.relation == "during":
                seg, iv = aseg, (a0, a1)
            else:
                j = aseg.idx + (1 if plan.relation == "after" else -1)
                seg = segs[j] if 0 <= j < len(segs) else None
                iv = (seg.start, seg.end) if seg else None
            cov_text, cov_ev = plan.anchor, a_ev
        else:
            top = self.index.search(video_id, text, mods, k)
            seg, score = top[0]
            if full:                                                  # fine temporal grounding
                g0, g1, _ = ground(seg, text, self.emb, mods, **c["grounding"])
                iv = (g0, g1)
            else:                                                     # segment-level only
                iv = (seg.start, seg.end)
            cov_text = text
            cov_ev = None
        ev = [e for e in seg.evidence if e.modality in mods and e.t0 < iv[1] and e.t1 > iv[0]] if seg else []
        cov = coverage(cov_text, cov_ev if cov_ev is not None else ev) if ev else 0.0
        ranked = ([(seg.start, seg.end)] if seg else []) + [(s.start, s.end) for s, _ in top if s is not seg]

        ok = seg is not None and bool(ev)
        if verify:                                                   # evidence verification
            ok = ok and cov >= c["verification"]["min_coverage"] and (plan.wants != "code" or find_code(ev))
        ans = None
        if ok:
            ans = llm_answer(plan, ev, generate) if generate else extractive(plan, ev, strict=verify)
        conf = round(0.6 * cov + 0.4 * min(1.0, score / 0.5), 2)
        if ans is None and (verify or not ev):
            out = Answer(ABSTAIN if (verify or not iv) else "(no evidence text)", None, None, bool(verify or not iv), min(conf, 0.3),
                         rel_label, [], ranked=ranked)
            if not out.abstained:
                out.start, out.end = iv if iv else (None, None)
        else:
            out = Answer(ans, iv[0], iv[1], False, conf, rel_label, [vars(e) for e in ev], ranked=ranked)
            if video_id in self.media and c.get("media", {}).get("cut_clips"):                                # clip + supporting frames
                o = c["paths"]["outputs"]
                out.clip_path = cut_clip(self.media[video_id], iv[0], iv[1], f"{o}/{video_id}_{iv[0]:.0f}.mp4")
                out.frames = extract_frames(self.media[video_id], [iv[0], sum(iv) / 2, max(iv[0], iv[1] - 0.1)], o)
        out.latency_ms = (time.perf_counter() - t0) * 1000
        return out
