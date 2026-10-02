"""Text embedder for entity matching. Hashing char-n-gram fallback (offline, deterministic);
swap for sentence-transformers in real runs."""
import hashlib, numpy as np

class HashEmbedder:
    def __init__(self, dim=256, n=3):
        self.dim, self.n = dim, n
    def _one(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        t = f"#{text.lower()}#"
        for i in range(len(t) - self.n + 1):
            h = int(hashlib.md5(t[i:i + self.n].encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        nrm = np.linalg.norm(v)
        return v / nrm if nrm else v
    def __call__(self, texts):
        return np.stack([self._one(t) for t in texts])

def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
