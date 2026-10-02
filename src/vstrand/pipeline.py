from .config import load_config
from .perception.lvlm import build_lvlm
from .graph.builder import VideoGraph
from .graph.retriever import retrieve
from .reasoning.structured import structured_reasoning

class VideoGraphRAG:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        self.lvlm = build_lvlm(self.cfg)
        self.graph = None

    def index(self, clips):                                   # offline, once per video
        g = self.cfg["graph"]
        self.graph = VideoGraph.build(clips, self.lvlm, g["embed_dim"], g["merge_threshold"])
        return self.graph.stats()

    def ask(self, query):                                     # online, per query
        r = self.cfg["retrieval"]
        cands, _ = retrieve(self.graph, query, r["top_n"], r["graph_hops"])
        kept, subs, trace = structured_reasoning(
            query, cands, self.graph.descriptions, self.lvlm, self.cfg["reasoning"]["min_pass_ratio"])
        return {"query": query, "retrieved": sorted(cands), "verified": sorted(kept),
                "subqueries": [s.text for s in subs], "trace": trace}
