# VideoAssist
**Multimodal Temporal Event Retrieval and Evidence-Grounded Question Answering for IT Field-Service Videos**
Domain: IT / network field service (router installation · troubleshooting · device replacement). User: field technician / supervisor.

Given a video and a natural-language query, VideoAssist finds the relevant event, localizes its time interval, resolves
before/after/during relations, and answers **only from visual, speech and OCR evidence** — or abstains.

**Design principle:** retrieval → temporal grounding → evidence verification → answer. The answerer never sees anything but retrieved evidence (`qa.py: llm_answer`).

## Output (per query)
answer · start/end timestamp · clip · supporting frames · transcript/OCR evidence · temporal relation · confidence · abstention flag
(`schema.Answer`; clip/frames are produced with FFmpeg when the source video file is registered.)

## Pipeline → code
| Stage | Module |
|---|---|
| preprocessing, speech, OCR, visual (YOLO + pretrained VLM) | `extractors.py` (real; untested here) / `data/synthetic.py` (staged stand-in) |
| temporal event segmentation | `segment.py` |
| multimodal indexing | `index.py` (numpy; FAISS is the drop-in) |
| query understanding | `query.py` |
| coarse retrieval | `index.search` |
| fine temporal grounding | `ground.py` |
| temporal relation reasoning | `pipeline.py` (before / after / during) |
| evidence verification + abstention | `verify.py` |
| grounded QA | `qa.py` |
| API | `api.py` (FastAPI) |

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt && pip install -e .
uvicorn videoassist.api:app --reload                   # or: make api
```
Open **http://localhost:8000** -> the UI: pick a video, ask a question (or click an example), the player jumps to the
grounded interval, the timeline highlights it, and the evidence list is clickable. Abstentions are shown explicitly.
Other commands: `make data eval` (ablation table), `make demo` (CLI), `make test`. Docker: `docker compose up --build api`.

## Where videos live
`data/demo/<video_id>.mp4` + `data/demo/<video_id>.json` (evidence records; `operation` optional). Everything in
`data/demo/` and `data/staged/` is auto-loaded at startup and streamed from `/media/<id>.mp4`.
The 3 included videos are **generated placeholders** (`scripts/make_demo_videos.py`). To use real footage:
extract evidence locally (`POST /ingest/video`, needs Whisper/PaddleOCR/YOLO) or write the JSON yourself, save the JSON,
drop the mp4 next to it, restart. Keep files < 100 MB for GitHub (use Git LFS or external storage otherwise).

## Deploy to Render (via GitHub)
See **DEPLOY.md**. Short version: push to GitHub -> Render "New > Blueprint" -> select repo -> Deploy (`render.yaml` is included).
Free tier = demo mode only (spins down after 15 min idle, no persistent disk, small RAM -> no Whisper/OCR/VLM there).

## Evaluation (RQ1–RQ4)
Ablations (`configs/default.yaml`): **A** transcript · **B** visual · **C** visual+transcript · **D** +OCR (A–D: segment-level) ·
**E-noV** +fine grounding & relation reasoning · **E** full (+ verification).
Metrics: Recall@1/3, MRR (per-video retrieval, hit = overlap with ground truth), mean tIoU, tIoU≥0.5, timestamp-grounded answer accuracy
(tIoU≥0.5 and correct value, e.g. error code), unsupported-answer rate (= answered an unanswerable query, or zero overlap with truth),
false/correct abstention, p50/p95 latency.
RQ1: A→D · RQ2: D→E-noV · RQ3: E-noV→E · RQ4: accuracy vs latency columns (+ false-abstain cost).

## Honest status
* **All numbers from this repo are on synthetic staged data with a hashing embedder** — they validate the pipeline and the *direction* of each effect, not real-world performance. Ablation gaps partly reflect how the synthetic captions were written. Do not cite them as results.
* Latency here excludes Whisper/OCR/VLM; real p50/p95 will be dominated by them (index-time vs query-time must be reported separately).
* `confidence` is a heuristic (coverage + retrieval score), not calibrated.
* Not yet done: running `extractors.py` on real footage; VLM visual captions (TODO in `visual_evidence`); PostgreSQL persistence (schema in `db/`, API is in-memory); FAISS; React/Next.js UI; optional timeline/event-graph/summaries.
* Next: collect 50–100 de-identified/staged videos with human-verified event timestamps; swap in `sentence-transformers` (`embedding.backend: st`) and re-tune thresholds on a dev split.

## Scope boundaries
Out of scope (and not implemented): face recognition, person identification, CCTV / city-scale / multi-camera surveillance, autonomous troubleshooting or fault diagnosis, foundation-model training, unsupported real-time claims. Uses pretrained models only. De-identify footage (faces, customer data, IPs/serials in OCR) before use.
