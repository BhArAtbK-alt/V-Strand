import json, os, pathlib
from dataclasses import asdict
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from .config import ROOT, load_config
from .pipeline import VideoAssist
from .schema import Evidence

app, cfg = FastAPI(title="VideoAssist"), load_config()
va, meta, STATIC = VideoAssist(cfg), {}, pathlib.Path(__file__).parent / "static"
DATA_DIRS = [ROOT / d for d in os.getenv("VIDEOASSIST_DATA_DIRS", "data/demo,data/staged").split(",")]

def register(vid, evidence, operation=None, mp4=None):
    info = va.ingest(vid, evidence, str(mp4) if mp4 else None)
    meta[vid] = {"video_id": vid, "operation": operation, "segments": info["segments"],
                 "duration": max(e.t1 for e in evidence), "video_url": f"/media/{vid}.mp4" if mp4 else None}
    return info

def load_dirs():                                     # auto-load <id>.json (+ <id>.mp4 if present) at startup
    for d in DATA_DIRS:
        for f in sorted(d.glob("*.json")):
            j = json.loads(f.read_text())
            mp4 = d / f"{j['video_id']}.mp4"
            register(j["video_id"], [Evidence(**e) for e in j["evidence"]], j.get("operation"), mp4 if mp4.exists() else None)
load_dirs()

class IngestEvidence(BaseModel):
    video_id: str
    evidence: list[dict]
    operation: str | None = None

class IngestVideo(BaseModel):
    video_id: str
    video_path: str

class Query(BaseModel):
    video_id: str
    query: str
    mode: str = "E"

@app.get("/health")
def health(): return {"ok": True, "videos": len(meta)}

@app.get("/videos")
def videos(): return list(meta.values())

@app.get("/videos/{vid}/timeline")
def timeline(vid: str):
    if vid not in va.index.segs: raise HTTPException(404, "unknown video")
    segs = va.index.segs[vid]
    return {**meta[vid], "segments": [{"idx": s.idx, "start": s.start, "end": s.end} for s in segs],
            "evidence": [vars(e) for s in segs for e in s.evidence if e.modality != "visual"]}

@app.get("/media/{name}")
def media(name: str):
    for d in DATA_DIRS:
        p = d / name
        if p.suffix == ".mp4" and p.is_file() and p.resolve().parent == d.resolve():
            return FileResponse(p, media_type="video/mp4")         # supports HTTP Range -> seeking works
    raise HTTPException(404)

@app.post("/ingest/evidence")                 # pre-extracted / staged evidence
def ingest_evidence(r: IngestEvidence):
    return register(r.video_id, [Evidence(**{**e, "video_id": r.video_id}) for e in r.evidence], r.operation)

@app.post("/ingest/video")                    # real extractors (Whisper, PaddleOCR, YOLO): local/GPU use, see extractors.py
def ingest_video(r: IngestVideo):
    from .extractors import extract_video
    return register(r.video_id, extract_video(r.video_id, r.video_path))

@app.post("/query")
def query(r: Query):
    if r.video_id not in va.index.segs: raise HTTPException(404, "video not ingested")
    if r.mode not in cfg["modes"]: raise HTTPException(400, f"mode must be one of {list(cfg['modes'])}")
    return asdict(va.query(r.video_id, r.query, **cfg["modes"][r.mode]))

@app.get("/")
def ui(): return FileResponse(STATIC / "index.html")
