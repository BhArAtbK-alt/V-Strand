import re, hashlib
import numpy as np

STOP = set("a an the and or of to in on at for with from by is are was were be been do does did what when where "
           "which who how that this it its into onto as then while after before during technician they".split())

def tokens(t):
    return [w for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", t.lower()) if w not in STOP]

def stem(w):
    return w[:5]

class Embedder:
    """Offline hashing embedder (default) or sentence-transformers (backend='st')."""
    def __init__(self, dim=512, backend="hash"):
        self.dim, self.cache, self.st = dim, {}, None
        if backend == "st":
            from sentence_transformers import SentenceTransformer
            self.st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.dim = self.st.get_sentence_embedding_dimension()

    def _hash(self, t):
        v = np.zeros(self.dim, np.float32)
        ts = [stem(w) for w in tokens(t)]
        for f in ts + [a + "_" + b for a, b in zip(ts, ts[1:])]:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += 1 if (h >> 20) & 1 else -1
        n = np.linalg.norm(v)
        return v / n if n else v

    def one(self, t):
        if t not in self.cache:
            self.cache[t] = self.st.encode(t, normalize_embeddings=True) if self.st else self._hash(t)
        return self.cache[t]
