"""FastAPI service and web page (laptop, CPU).

  python -m uvicorn cap.api:app --port 8015   ->   http://localhost:8015   (interactive API docs: /docs)

  POST /caption?model=finetuned|zero_shot|both&beams=   upload a photo -> caption(s)
  GET  /samples             held-out test photos with their 5 human captions and every model's caption
  GET  /samples/{file}      one demo photo
  GET  /model-info          model card and all results
  GET  /health              status and model version
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .infer import DISCLAIMER, Service

ART = Path(os.environ.get("CAP_ARTIFACTS", "artifacts"))
UI = Path(__file__).parent / "static" / "index.html"
_cache: dict = {}


def svc() -> Service:
    if "s" not in _cache:
        try:
            _cache["s"] = Service(ART)
        except (FileNotFoundError, RuntimeError) as e:
            raise HTTPException(503, str(e))
    return _cache["s"]


@asynccontextmanager
async def lifespan(_app):
    try:
        s = svc()
        print(f"model {s.card['name']} loaded - http://localhost:8015", flush=True)
    except Exception as e:
        print(f"model not loaded: {getattr(e, 'detail', e)}", flush=True)
    yield


app = FastAPI(title="Image captioning (Flickr8k, BLIP)", version="1.0", description=DISCLAIMER, lifespan=lifespan)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui():
    return HTMLResponse(UI.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    s = _cache.get("s")
    return {"status": "ready" if s else "model not loaded", "model_version": s.card["name"] if s else None}


@app.get("/model-info")
def model_info():
    s = svc()
    return {"version": s.card, "metrics": {k: v for k, v in s.metrics.items() if k not in ("history",)}}


@app.get("/samples")
def samples():
    return svc().samples


@app.get("/samples/{name}", include_in_schema=False)
def sample_file(name: str):
    d = (svc().dir / "samples").resolve()
    p = (d / name).resolve()
    if p.parent != d or not p.exists() or p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        raise HTTPException(404, "not found")
    return FileResponse(p)


@app.post("/caption")
async def caption(file: UploadFile = File(..., description="a photo (JPEG or PNG)"),
                  model: str = Query("finetuned", description="finetuned, zero_shot or both"),
                  beams: int | None = Query(None, ge=1, le=10, description="beam size (default: chosen on validation)")):
    data = await file.read()
    try:
        return svc().caption(data, model, beams)
    except ValueError as e:
        raise HTTPException(415 if "not an image" in str(e) else 422, str(e))
