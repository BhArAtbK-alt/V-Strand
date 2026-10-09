"""python scripts/make_dataset.py [n_videos]  -> data/staged/*.json (synthetic stand-in for the 50-100 staged videos)"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from videoassist.data.synthetic import generate
out = pathlib.Path(__file__).resolve().parents[1] / "data" / "staged"
for v in generate(int(sys.argv[1]) if len(sys.argv) > 1 else 60):
    (out / f"{v['video_id']}.json").write_text(json.dumps(v, indent=1))
print("written to", out)
