"""Graph retrieval: score entity prototypes by similarity to the query, score clips by the
entities they mention, then expand through clip--clip edges (graph_hops)."""
import numpy as np
from .embedder import cosine

def retrieve(G, query, top_n=4, hops=1):
    qwords = [w for w in query.lower().replace("?", "").split() if len(w) > 2]
    qv = G.embed(qwords)
    scores = {}
    for pid, (c, _, _) in G.protos.items():
        s = max(cosine(q, c) for q in qv)
        if s < 0.5: continue
        for nb in G.g[pid]:
            scores[nb] = scores.get(nb, 0) + s
    frontier = dict(scores)
    for _ in range(hops):                         # light spreading activation
        for cid, s in list(frontier.items()):
            for nb, e in G.g[cid].items():
                if e["kind"] == "shares":
                    scores[nb] = scores.get(nb, 0) + 0.1 * s * e["weight"] / 3
    ranked = sorted((c for c in scores if c.startswith("C")), key=lambda c: -scores[c])[:top_n]
    return [int(c[1:]) for c in ranked], scores
