"""LVLM backbone interface. The pipeline only talks to BaseLVLM, so swapping
mock -> Video-LLaVA -> Qwen2.5-VL changes nothing downstream."""
from abc import ABC, abstractmethod
import re
from ..reasoning.types import SubQuery

class BaseLVLM(ABC):
    @abstractmethod
    def describe_clip(self, clip) -> dict:
        """-> {'description': str, 'entities': list[str]} from frames + transcript."""
    @abstractmethod
    def verify(self, clip_description: str, sub: SubQuery) -> bool:
        """Answer a binary sub-query about a clip."""
    @abstractmethod
    def decompose(self, query: str) -> list:
        """Split a user query into binary/numeric sub-queries."""

STOP = {"the","a","an","and","on","in","at","to","of","while","beside","into","it","its","is","this","that","with"}
def _tok(t): return [w for w in re.findall(r"[a-z]+", t.lower()) if w not in STOP]
def _stem(w): return w[:4] if len(w) > 4 else w

class MockLVLM(BaseLVLM):
    """Rule-based stand-in so the demo is deterministic and runs offline."""
    def describe_clip(self, clip):
        text = f"{clip.caption} {clip.transcript}".strip()
        return {"description": text, "entities": sorted({w for w in _tok(clip.caption) if len(w) > 3})}

    def verify(self, desc, sub):
        stems = {_stem(w) for w in _tok(desc)}
        return all(_stem(k) in stems for k in sub.keywords)

    def decompose(self, query):
        m = re.match(r"did (?:i|he|she|they) (\w+) (?:the |a )?(\w+)(?: (?:into|in|on) the (\w+))?\??$", query.lower().strip())
        if not m:
            return [SubQuery(query, "binary", _tok(query))]
        verb, obj, extra = m.groups()
        return [SubQuery(f"Does the clip show the person {verb} the {obj}?", "binary",
                         [verb, obj] + ([extra] if extra else []))]

class HFVideoLLaVA(BaseLVLM):
    """Real backbone (GPU + `pip install transformers accelerate decord`).
    The three prompts below are the three LVLM calls a Vgent-style pipeline needs."""
    DESCRIBE = ("List key entities (people, objects, actions) in this clip and describe what happens. "
                "Speech: {transcript}\nReturn JSON {{\"description\":..., \"entities\":[...]}}")
    VERIFY = "Clip description: {desc}\nQuestion: {q}\nAnswer strictly yes or no."
    DECOMPOSE = ("Break this question into the smallest yes/no or numeric sub-questions needed to verify "
                 "a clip answers it. Return a JSON list of strings.\nQuestion: {q}")
    def __init__(self, hf_id="LanguageBind/Video-LLaVA-7B-hf", dtype="bfloat16"):
        import torch
        from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
        self.processor = VideoLlavaProcessor.from_pretrained(hf_id)
        self.model = VideoLlavaForConditionalGeneration.from_pretrained(
            hf_id, torch_dtype=getattr(torch, dtype), device_map="auto")
    def describe_clip(self, clip): raise NotImplementedError("TODO: processor(text,videos) -> generate -> parse JSON")
    def verify(self, d, s): raise NotImplementedError("TODO: text-only generate with VERIFY prompt")
    def decompose(self, q): raise NotImplementedError("TODO: text-only generate with DECOMPOSE prompt")

def build_lvlm(cfg):
    name = cfg["backbone"]["name"]
    if name == "mock": return MockLVLM()
    if name == "video_llava": return HFVideoLLaVA(cfg["backbone"]["hf_id"], cfg["backbone"]["dtype"])
    raise ValueError(name)
