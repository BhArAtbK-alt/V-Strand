"""REAL extraction for actual videos (needs: ffmpeg, openai-whisper, paddleocr, ultralytics, transformers).
Lazy imports; produces the same Evidence records the staged annotations provide.
NOT yet exercised in this repo's tests -- verify on real footage. Library APIs vary by version."""
import subprocess, pathlib, tempfile
from .schema import Evidence

def sample_frames(video, fps=1, outdir=None):
    outdir = outdir or tempfile.mkdtemp()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vf", f"fps={fps}", f"{outdir}/%05d.jpg"], check=True)
    return sorted(pathlib.Path(outdir).glob("*.jpg"))

def transcribe(video_id, video, model="small"):
    import whisper
    r = whisper.load_model(model).transcribe(video)
    return [Evidence(video_id, "speech", s["start"], s["end"], s["text"].strip()) for s in r["segments"]]

def ocr_frames(video_id, frames, fps=1):
    from paddleocr import PaddleOCR
    ocr, out = PaddleOCR(lang="en"), []
    for i, f in enumerate(frames):
        res = ocr.ocr(str(f)) or [[]]
        txt = " ".join(line[1][0] for line in (res[0] or []))
        if txt: out.append(Evidence(video_id, "ocr", i / fps, (i + 1) / fps, txt))
    return out

def visual_evidence(video_id, frames, fps=1):
    """YOLO-class detector gives equipment labels; a pretrained video-capable VLM gives per-window captions."""
    from ultralytics import YOLO          # use weights fine-tuned/checked for routers, cables, LEDs
    det = YOLO("yolov8n.pt")
    out = []
    for i, f in enumerate(frames):
        names = {det.names[int(b.cls)] for b in det(str(f), verbose=False)[0].boxes}
        # TODO: caption = vlm_caption(frames[i:i+2])  (pretrained VLM, e.g. via transformers)
        out.append(Evidence(video_id, "visual", i / fps, (i + 1) / fps, "objects: " + ", ".join(sorted(names))))
    return out

def extract_video(video_id, video, fps=1):
    frames = sample_frames(video, fps)
    return transcribe(video_id, video) + ocr_frames(video_id, frames, fps) + visual_evidence(video_id, frames, fps)
