"""
OCR Luminar web backend.

Design notes:
- `engine` below is instantiated ONCE at module load time and shared by every
  request/thread. There is no per-job or per-thread engine, session, or model
  duplication -- every page transcription call, across every concurrent job,
  goes through this single GlmOcrEngine (one requests.Session / connection
  pool). The model itself lives once in Ollama's process, kept resident via
  `keep_alive` in engine.py.
- Page-level parallelism is handled per-job with a ThreadPoolExecutor sized
  by PAGE_WORKERS -- tune this up on stronger GPUs. All pages of a job pull
  from the same shared `engine`, so raising PAGE_WORKERS increases concurrent
  *requests* against Ollama, not the number of models loaded.
- Job state lives in an in-memory dict guarded by a lock. Fine for a single-
  user local tool; swap for Redis/DB if this ever needs multiple clients.
"""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr_luminar.convert import image_to_png_bytes, is_supported, load_pages  # noqa: E402
from ocr_luminar.engine import GlmOcrEngine, OcrEngineError  # noqa: E402
from ocr_luminar.writer import write_json, write_markdown  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PAGE_WORKERS = 4  # concurrent page requests per job -- raise on stronger GPUs

# --- single shared engine instance, see module docstring ---
engine = GlmOcrEngine(num_ctx=8192)

app = FastAPI(title="OCR Luminar")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()
job_pool = ThreadPoolExecutor(max_workers=4)  # concurrent jobs, independent of PAGE_WORKERS


def _update(job_id: str, **kwargs) -> None:
    with jobs_lock:
        jobs[job_id].update(kwargs)


def _run_job(job_id: str, file_path: Path, fmt: str, dpi: int) -> None:
    try:
        _update(job_id, status="reading", started_at=time.time())
        pages = load_pages(file_path, dpi=dpi)
        total = len(pages)
        _update(job_id, status="processing", total_pages=total, done_pages=0)

        page_markdowns: list[str | None] = [None] * total

        def do_page(page):
            png = image_to_png_bytes(page.image)
            markdown = engine.transcribe_image(png)  # shared engine, called concurrently
            return page.index, markdown

        with ThreadPoolExecutor(max_workers=PAGE_WORKERS) as pool:
            futures = [pool.submit(do_page, p) for p in pages]
            for future in as_completed(futures):
                idx, markdown = future.result()
                page_markdowns[idx] = markdown
                with jobs_lock:
                    jobs[job_id]["done_pages"] += 1

        doc_name = file_path.stem
        outputs: dict[str, str] = {}
        if fmt in ("md", "both"):
            outputs["md"] = str(write_markdown(OUTPUT_DIR, doc_name, page_markdowns))
        if fmt in ("json", "both"):
            outputs["json"] = str(write_json(OUTPUT_DIR, doc_name, file_path, page_markdowns, engine.model))

        _update(job_id, status="done", outputs=outputs, finished_at=time.time())
    except OcrEngineError as e:
        _update(job_id, status="error", error=str(e))
    except Exception as e:  # noqa: BLE001
        _update(job_id, status="error", error=f"Unexpected error: {e}")


@app.get("/api/health")
def health() -> dict:
    try:
        engine.check_available()
        return {"ok": True, "model": engine.model}
    except OcrEngineError as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), format: str = "both", dpi: int = 200) -> dict:
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    if not is_supported(dest):
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"Unsupported file type: {dest.suffix}")

    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "filename": file.filename,
            "status": "queued",
            "done_pages": 0,
            "total_pages": 0,
        }

    job_pool.submit(_run_job, job_id, dest, format, dpi)
    return {"job_id": job_id}


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    with jobs_lock:
        return sorted(jobs.values(), key=lambda j: j.get("started_at", 0), reverse=True)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/jobs/{job_id}/download/{fmt}")
def download(job_id: str, fmt: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "Result not ready")
    path = job.get("outputs", {}).get(fmt)
    if not path:
        raise HTTPException(404, f"Format '{fmt}' not available for this job")
    return FileResponse(path, filename=Path(path).name)


app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")