---
title: CarnaticSearch
emoji: 🎵
colorFrom: red
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: BM25 + LGBMRanker search over Carnatic kritis
---

# CarnaticSearch (HF Space deployment)

Self-contained production copy of CarnaticSearch, ready to deploy as a Hugging Face Space using the **Docker SDK**.

## What's in here

```
prod/
├── Dockerfile          # Docker SDK config
├── requirements.txt    # pinned Python deps
├── README.md           # this file (HF metadata at top)
├── app.py              # FastAPI backend
├── search.py           # BM25 + LGBMRanker pipeline
├── song_data/
│   └── data.json       # corpus
├── static/             # frontend (HTML/CSS/JS)
└── state/              # writable: ranker_v3.pkl + feedback_log.jsonl
```

## Deploy to a new HF Space

1. Create a new Space at https://huggingface.co/new-space — pick **Docker** as the SDK.
2. Clone the empty Space repo locally:
   ```bash
   git clone https://huggingface.co/spaces/<your-name>/<space-name>
   cd <space-name>
   ```
3. Copy the contents of this `prod/` directory into the cloned Space repo.
4. Commit and push:
   ```bash
   git add .
   git commit -m "Initial deploy"
   git push
   ```

The build takes a few minutes (most of it is `lightgbm` + `scikit-learn`). Once it finishes, the Space serves the UI at `https://huggingface.co/spaces/<you>/<name>`.

## Persistent storage (optional)

By default, `state/` lives inside the container — `feedback_log.jsonl` and `ranker_v3.pkl` reset each time the Space restarts. To make them survive restarts:

1. In the Space **Settings → Persistent Storage**, enable a persistent disk (paid feature, ~$5/mo).
2. The disk mounts at `/data`.
3. Set the env var in the Space settings: `CARNATIC_STATE_DIR=/data`.

The code writes ranker + feedback to whatever directory `CARNATIC_STATE_DIR` points at (default `./state`).

## Run locally before deploying

```bash
cd prod
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
# open http://localhost:7860
```

Same UI, same endpoints as the Space.
