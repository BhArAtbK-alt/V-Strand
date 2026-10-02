import torch
from vstrand.pipeline import VideoGraphRAG
from vstrand.data.synthetic import make_demo_video
from vstrand.edge.approxbit import ApproxBit, compression_ratio
from vstrand.perception.projector import SharedProjector
from vstrand.training import train_kg

def test_structured_reasoning_removes_hard_negatives():
    rag = VideoGraphRAG(); rag.index(make_demo_video())
    r = rag.ask("Did I open the laptop?")
    assert 2 in r["verified"]                    # the clip where the laptop is opened
    assert 1 not in r["verified"] and 6 not in r["verified"]   # 'closed laptop' hard negatives
    assert len(r["retrieved"]) > len(r["verified"])

def test_entity_merging():
    rag = VideoGraphRAG(); st = rag.index(make_demo_video())
    total_mentions = sum(len(rag.lvlm.describe_clip(c)["entities"]) for c in make_demo_video())
    assert st["entity_prototypes"] < total_mentions

def test_projector_shapes():
    p = SharedProjector(32, 64)
    assert p(torch.randn(1, 4, 32), torch.randn(1, 10, 32)).shape == (1, 14, 64)

def test_approxbit_is_binary_and_compresses():
    x = torch.randn(2, 64, 4, 8, 8); rec, code = ApproxBit()(x)
    assert set(code.unique().tolist()) <= {0.0, 1.0} and rec.shape == x.shape
    assert compression_ratio(x, code) == 16 * 32

def test_ontology_reduces_violation():
    _, _, o_off = train_kg.train(use_ontology=False, epochs=40, verbose=False)
    _, _, o_on = train_kg.train(use_ontology=True, epochs=40, verbose=False)
    assert o_on <= o_off + 1e-6
