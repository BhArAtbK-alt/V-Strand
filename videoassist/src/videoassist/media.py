"""FFmpeg helpers: relevant clip + supporting frames. No-op when the source video file is unavailable."""
import subprocess, pathlib

def cut_clip(video, t0, t1, out):
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.2f}", "-to", f"{t1:.2f}", "-i", video,
                    "-c", "copy", out], check=True)
    return out

def extract_frames(video, times, outdir):
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    paths = []
    for i, t in enumerate(times):
        p = f"{outdir}/frame_{i}_{t:.1f}.jpg"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1", p], check=True)
        paths.append(p)
    return paths
