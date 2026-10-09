# Deploy VideoAssist to Render from GitHub

## 1. Push to GitHub
```bash
cd videoassist
git init -b main
git add . && git commit -m "VideoAssist: initial commit"
# A) GitHub CLI
gh auth login && gh repo create videoassist --private --source=. --remote=origin --push
# B) or web: github.com/new -> create EMPTY repo "videoassist", then
git remote add origin https://github.com/<you>/videoassist.git && git push -u origin main
```
Check that `data/demo/*.mp4` was committed (`git ls-files data/demo`) — Render builds from the repo, so the demo videos must be in it.

## 2. Create the service on Render
1. Sign in at https://render.com with GitHub and authorize access to the repo.
2. **New +  ->  Blueprint** -> pick the `videoassist` repo -> Render reads `render.yaml` -> **Apply / Deploy**.
   (Manual alternative: **New + -> Web Service** -> repo -> Language/Runtime: **Docker** -> Instance type: **Free** -> Health check path `/health`.)
3. Wait for the Docker build (a few minutes). When it shows **Live**, open `https://videoassist-xxxx.onrender.com`.
4. Check `/health` returns `{"ok": true, "videos": 3}` and the UI loads with a playable video.

## 3. Update
`git push` to `main` -> Render redeploys automatically (private repos need the Render GitHub app installed on that repo).

## Free-tier limits (verified in Render docs)
* Spins down after ~15 min idle -> first request afterwards is slow (cold start). Open the URL a minute before a demo.
* No persistent disk -> uploads/changes made at runtime are lost on restart/redeploy. Put data in the repo.
* Small RAM/CPU -> demo mode only (staged evidence + pre-made videos). Run Whisper/PaddleOCR/YOLO/VLM extraction **locally or on a GPU box**, then commit the resulting JSON + mp4 (or use a paid plan + disk).
* The service is public once deployed: do not publish real customer footage. De-identify first.

## Troubleshooting
* Build fails -> read the Render build log; test locally first: `docker build -t va . && docker run -p 8000:8000 va`.
* Port errors -> the Dockerfile binds to `$PORT` (Render sets it); don't hard-code a port.
* Video won't seek/play -> use H.264 (`libx264`, `yuv420p`) mp4; `scripts/make_demo_videos.py` shows how.
