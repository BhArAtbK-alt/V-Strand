from dataclasses import dataclass, field

@dataclass
class Evidence:                      # one timestamped observation from one modality
    video_id: str
    modality: str                    # visual | speech | ocr
    t0: float
    t1: float
    text: str

@dataclass
class Segment:                       # temporal event candidate
    video_id: str
    idx: int
    start: float
    end: float
    evidence: list = field(default_factory=list)

@dataclass
class QueryPlan:
    raw: str
    target: str
    relation: str | None             # before | after | during
    anchor: str | None
    wants: str                       # time | what | code | yesno

@dataclass
class Answer:
    answer: str
    start: float | None
    end: float | None
    abstained: bool
    confidence: float                # heuristic, NOT calibrated
    relation: str | None = None
    evidence: list = field(default_factory=list)
    clip_path: str | None = None
    frames: list = field(default_factory=list)
    ranked: list = field(default_factory=list)       # top-K (start, end) candidates
    latency_ms: float = 0.0
