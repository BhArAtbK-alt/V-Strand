"""End-to-end investor/industry demo (offline, CPU, ~seconds).
Shows: offline graph build -> retrieval (with hard negatives) -> structured reasoning filter
       -> neural KG training -> 3D aligner shapes -> edge 1-bit compression."""
import sys, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from vstrand.pipeline import VideoGraphRAG
from vstrand.data.synthetic import make_demo_video, DEMO_QUERIES
from vstrand.perception.encoder import MockLanguageBind
from vstrand.perception.projector import SharedProjector
from vstrand.spatial.aligner import SpatioTemporalAligner
from vstrand.training import train_kg, train_approxbit

def hr(t): print(f"\n{'='*8} {t} {'='*8}")

hr("1. Offline GraphRAG index (query-independent)")
rag = VideoGraphRAG(); clips = make_demo_video()
print("graph:", rag.index(clips))

hr("2. Retrieval + Structured Reasoning")
for q in DEMO_QUERIES:
    r = rag.ask(q)
    print(f"\nQ: {q}\n  sub-queries : {r['subqueries']}\n  retrieved   : clips {r['retrieved']}")
    print(f"  verified    : clips {r['verified']}   <- hard negatives removed: {sorted(set(r['retrieved'])-set(r['verified']))}")
    for c in r["verified"]: print(f"    clip {c} @ {clips[c].start:.0f}s: {rag.graph.descriptions[c]}")

hr("3. Alignment-before-projection (shapes)")
enc, proj = MockLanguageBind(), SharedProjector(1024, 512)
v = enc.encode_video(torch.zeros(1, 8, 3, 224, 224)); i = enc.encode_image(torch.zeros(1, 3, 224, 224))
print("video", tuple(v.shape), "+ image", tuple(i.shape), "->", tuple(proj(i, v).shape), "(LLM tokens)")

hr("4. Neural KG extraction (triples + ontology rule)")
_, s0, o0 = train_kg.train(use_ontology=False, verbose=False)
_, s1, o1 = train_kg.train(use_ontology=True, verbose=False)
print(f"rule violation  without ontology loss: {o0:.4f}   with: {o1:.4f}")

hr("5. 3D spatio-temporal grounding (shapes)")
al = SpatioTemporalAligner(1024, 256, 128, 4)
out, w = al(torch.randn(1, 32, 1024), torch.randn(1, 200, 256), torch.randn(1, 200, 3))
print("fused", tuple(out.shape), "grounding attention", tuple(w.shape))

hr("6. Edge->cloud 1-bit offloading")
_, m = train_approxbit.train(verbose=False)
print(f"payload reduction vs fp32: {m['compression_x']:.0f}x | recon MSE {m['mse']:.3f} (zero-baseline {m['baseline_mse']:.3f})")
