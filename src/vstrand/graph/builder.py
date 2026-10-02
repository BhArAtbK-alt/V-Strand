"""Pillar 2 - Offline, query-independent GraphRAG memory.
Nodes: clips + global entity prototypes.  Edges: clip--entity (mentions) and clip--clip (shared entity).
Built ONCE; every later query only traverses the graph, never re-tokenizes raw video."""
import numpy as np, networkx as nx
from .embedder import HashEmbedder, cosine

class VideoGraph:
    def __init__(self, embed_dim=256, merge_threshold=0.8):
        self.g = nx.Graph()
        self.embed = HashEmbedder(embed_dim)
        self.thr = merge_threshold
        self.protos = {}          # proto_id -> (centroid, count, canonical name)
        self.descriptions = {}    # clip_id -> text

    def _assign_proto(self, name):
        v = self.embed([name])[0]
        best, best_s = None, -1
        for pid, (c, n, _) in self.protos.items():
            s = cosine(v, c)
            if s > best_s: best, best_s = pid, s
        if best is not None and best_s >= self.thr:
            c, n, nm = self.protos[best]
            self.protos[best] = ((c * n + v) / (n + 1), n + 1, nm)     # running centroid
            return best
        pid = f"E{len(self.protos)}"
        self.protos[pid] = (v, 1, name)
        return pid

    def add_clip(self, clip, info):
        cid = f"C{clip.clip_id}"
        self.descriptions[clip.clip_id] = info["description"]
        self.g.add_node(cid, kind="clip", start=clip.start, end=clip.end, text=info["description"])
        for ent in info["entities"]:
            pid = self._assign_proto(ent)
            self.g.add_node(pid, kind="entity", name=self.protos[pid][2])
            self.g.add_edge(cid, pid, kind="mentions")
        return cid

    def link_clips(self):
        """Clip--clip edges weighted by number of shared entity prototypes."""
        clips = [n for n, d in self.g.nodes(data=True) if d["kind"] == "clip"]
        for i, a in enumerate(clips):
            for b in clips[i + 1:]:
                shared = set(self.g[a]) & set(self.g[b])
                if shared: self.g.add_edge(a, b, kind="shares", weight=len(shared))

    @classmethod
    def build(cls, clips, lvlm, embed_dim=256, merge_threshold=0.8):
        G = cls(embed_dim, merge_threshold)
        for c in clips: G.add_clip(c, lvlm.describe_clip(c))
        G.link_clips()
        return G

    def stats(self):
        k = [d["kind"] for _, d in self.g.nodes(data=True)]
        return {"clips": k.count("clip"), "entity_prototypes": k.count("entity"), "edges": self.g.number_of_edges()}
