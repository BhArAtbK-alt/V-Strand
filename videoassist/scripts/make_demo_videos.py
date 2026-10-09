"""python scripts/make_demo_videos.py  (needs: pip install imageio imageio-ffmpeg pillow)
Renders 3 small PLACEHOLDER videos (install / troubleshoot / replace) whose on-screen text mirrors the evidence records,
so the UI can be demoed end to end. Replace with real footage: put <video_id>.mp4 + <video_id>.json in data/demo/."""
import sys, json, pathlib, textwrap
import numpy as np, imageio
from PIL import Image, ImageDraw, ImageFont
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from videoassist.data.synthetic import generate
W, H, FPS = 480, 270, 4
def font(s):
    try: return ImageFont.load_default(size=s)
    except TypeError: return ImageFont.load_default()
F1, F2 = font(15), font(12)
for v in generate(3):
    dur = int(max(e["t1"] for e in v["evidence"])) + 2
    path = ROOT / "data" / "demo" / f"{v['video_id']}.mp4"
    w = imageio.get_writer(str(path), fps=FPS, codec="libx264", quality=6, macro_block_size=None, pixelformat="yuv420p")
    for f in range(dur * FPS):
        t = f / FPS
        img = Image.new("RGB", (W, H), (24, 28, 38)); d = ImageDraw.Draw(img)
        d.text((10, 8), f"{v['video_id']}  t={t:5.1f}s  [placeholder video]", fill=(150, 160, 180), font=F2)
        act = {m: next((e for e in v["evidence"] if e["modality"] == m and e["t0"] <= t < e["t1"]), None) for m in ("visual", "speech", "ocr")}
        if act["visual"]:
            d.multiline_text((14, 50), "\n".join(textwrap.wrap(act["visual"]["text"], 38)), fill=(235, 235, 240), font=F1, spacing=6)
        if act["ocr"]:
            d.rectangle((14, 150, W - 14, 185), outline=(80, 220, 120)); d.text((22, 158), "SCREEN: " + act["ocr"]["text"], fill=(80, 220, 120), font=F1)
        if act["speech"]:
            d.text((14, 235), '"' + act["speech"]["text"] + '"', fill=(250, 210, 90), font=F1)
        w.append_data(np.asarray(img))
    w.close()
    (ROOT / "data" / "demo" / f"{v['video_id']}.json").write_text(json.dumps({k: v[k] for k in ("video_id", "operation", "evidence")}))
    print(path.name, path.stat().st_size // 1024, "KB")
