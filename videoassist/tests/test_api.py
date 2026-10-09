from fastapi.testclient import TestClient
from videoassist.api import app
c = TestClient(app)

def test_ui_and_videos():
    assert "VideoAssist" in c.get("/").text
    vs = c.get("/videos").json()
    assert len(vs) >= 3 and all(v["video_url"] for v in vs)

def test_media_supports_range():
    r = c.get("/media/rou_001.mp4", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100
    assert c.get("/media/../../etc/passwd").status_code in (404, 422)

def test_timeline_and_query():
    t = c.get("/videos/rou_001/timeline").json()
    assert t["segments"] and t["duration"] > 30
    a = c.post("/query", json={"video_id": "rou_001", "query": "What error code is shown on the router?"}).json()
    assert not a["abstained"] and a["start"] is not None
    assert c.post("/query", json={"video_id": "rou_001", "query": "Did the technician configure a VLAN?"}).json()["abstained"]
