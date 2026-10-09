CREATE TABLE videos   (video_id TEXT PRIMARY KEY, operation TEXT, duration REAL, path TEXT);
CREATE TABLE segments (video_id TEXT REFERENCES videos, idx INT, start_s REAL, end_s REAL, PRIMARY KEY (video_id, idx));
CREATE TABLE evidence (id SERIAL PRIMARY KEY, video_id TEXT REFERENCES videos, modality TEXT CHECK (modality IN ('visual','speech','ocr')),
                       t0 REAL, t1 REAL, text TEXT);
CREATE TABLE queries_log (id SERIAL PRIMARY KEY, video_id TEXT, query TEXT, answer TEXT, start_s REAL, end_s REAL,
                          abstained BOOL, confidence REAL, latency_ms REAL, ts TIMESTAMPTZ DEFAULT now());
-- embeddings live in FAISS (per-modality indices); add pgvector later only if needed
