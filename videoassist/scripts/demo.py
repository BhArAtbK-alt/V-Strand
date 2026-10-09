"""python scripts/demo.py -> one video, three queries, full system output."""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from videoassist.config import load_config
from videoassist.pipeline import VideoAssist
from videoassist.schema import Evidence
from videoassist.data.synthetic import generate
v = generate(3)[1]                                   # a troubleshooting video
va = VideoAssist(load_config()); print(va.ingest(v["video_id"], [Evidence(**e) for e in v["evidence"]]))
for q in ["What error code is shown on the router?", "What does the technician do after they replace the faulty ethernet cable?",
          "Did the technician configure a VLAN?"]:
    a = va.query(v["video_id"], q)
    print(f"\nQ: {q}\n A: {a.answer}\n   [{a.start}-{a.end}s] relation={a.relation} conf={a.confidence} abstained={a.abstained}")
    for e in a.evidence[:3]: print(f"   evidence {e['modality']:6s} {e['t0']:.0f}-{e['t1']:.0f}s: {e['text']}")
