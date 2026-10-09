import pytest
from videoassist.config import load_config
from videoassist.data.synthetic import generate
from videoassist.evaluate import evaluate
from videoassist.metrics import tiou
from videoassist.pipeline import VideoAssist
from videoassist.query import parse
from videoassist.schema import Evidence

@pytest.fixture(scope="module")
def videos(): return generate(12)

def test_parse_relation_and_type():
    p = parse("What does the technician do after they connect the power cable?")
    assert p.relation == "after" and "power cable" in p.anchor
    assert parse("What error code is shown?").wants == "code" and parse("When does it start?").wants == "time"

def test_tiou(): assert tiou((0, 10), (5, 15)) == pytest.approx(1 / 3)

def test_grounded_answer_and_abstention(videos):
    v = videos[1]; va = VideoAssist(load_config())
    va.ingest(v["video_id"], [Evidence(**e) for e in v["evidence"]])
    a = va.query(v["video_id"], "What error code is shown on the router?")
    ev = v["gt"]["events"][1]
    assert not a.abstained and ev["key"] in a.answer and tiou((a.start, a.end), (ev["start"], ev["end"])) > 0.5
    assert va.query(v["video_id"], "Did the technician configure a VLAN?").abstained

def test_during_relation(videos):
    v = videos[0]; va = VideoAssist(load_config())
    va.ingest(v["video_id"], [Evidence(**e) for e in v["evidence"]])
    a = va.query(v["video_id"], "What is said while they connect the power cable to the router?")
    assert a.relation.startswith("during") and a.start is not None

def test_ablation_trends(videos):
    r = evaluate(videos, load_config())
    assert r["E"]["unsupported_rate"] < r["E-noV"]["unsupported_rate"]          # RQ3 direction
    assert r["E-noV"]["mean_tIoU"] > r["D"]["mean_tIoU"]                         # RQ2 direction
    assert r["D"]["grounded_answer_acc"] >= r["A"]["grounded_answer_acc"]        # RQ1 direction
