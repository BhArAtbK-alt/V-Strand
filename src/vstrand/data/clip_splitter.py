"""Partition a long video into sequential short clips."""
from .synthetic import Clip

def split_video(path, clip_seconds=10.0, frames_per_clip=8):
    try:
        from decord import VideoReader, cpu
    except ImportError as e:
        raise ImportError("pip install decord to read real videos") from e
    vr = VideoReader(path, ctx=cpu(0))
    fps, total = vr.get_avg_fps(), len(vr)
    step = int(fps * clip_seconds)
    clips = []
    for i, s in enumerate(range(0, total, step)):
        e = min(s + step, total)
        idx = [int(s + (e - s) * k / frames_per_clip) for k in range(frames_per_clip)]
        clips.append(Clip(i, s / fps, e / fps, caption="", frames=vr.get_batch(idx).asnumpy()))
    return clips
