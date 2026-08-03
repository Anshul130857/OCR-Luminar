"""
OCR Luminar web backend.

Nothing this app processes is written to disk. Uploaded files are read into
memory (bytes), pages are rendered to in-memory PIL images, and the final
Markdown/JSON are held as strings in the in-memory job store -- served
straight back to the browser on download. Restarting the server clears
everything, by design.

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
- Job state (including the in-memory output strings) lives in an in-memory
  dict guarded by a lock. Fine for a single-user local tool; swap for
  Redis/a DB if this ever needs to survive restarts or serve multiple people.
"""

from __future__ import annotations

import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr_luminar.convert import image_to_png_bytes, is_supported, load_pages_from_bytes  # noqa: E402
from ocr_luminar.engine import GlmOcrEngine, OcrEngineError  # noqa: E402
from ocr_luminar.writer import build_json, build_markdown  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent

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


def _run_job(job_id: str, data: bytes, filename: str, fmt: str, dpi: int) -> None:
    try:
        _update(job_id, status="reading", started_at=time.time())
        pages = load_pages_from_bytes(data, filename, dpi=dpi)
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

        results: dict[str, str] = {}
        if fmt in ("md", "both"):
            results["md"] = build_markdown(page_markdowns)
        if fmt in ("json", "both"):
            results["json"] = build_json(filename, page_markdowns, engine.model)

        # `results` holds the final strings in memory only -- nothing is
        # written to disk here or anywhere else in this request's lifetime.
        _update(job_id, status="done", results=results, finished_at=time.time())
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
    if not is_supported(Path(file.filename)):
        raise HTTPException(400, f"Unsupported file type: {Path(file.filename).suffix}")

    data = await file.read()  # held in memory only, never written to disk

    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "filename": file.filename,
            "status": "queued",
            "done_pages": 0,
            "total_pages": 0,
        }

    job_pool.submit(_run_job, job_id, data, file.filename, format, dpi)
    return {"job_id": job_id}


def _public_view(job: dict[str, Any]) -> dict[str, Any]:
    """Status info safe to send to the browser -- never includes the actual
    OCR content, only which formats are ready to download."""
    view = {k: v for k, v in job.items() if k != "results"}
    if "results" in job:
        view["available_formats"] = list(job["results"].keys())
    return view


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    with jobs_lock:
        return [_public_view(j) for j in sorted(jobs.values(), key=lambda j: j.get("started_at", 0), reverse=True)]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return _public_view(job)


@app.get("/api/jobs/{job_id}/download/{fmt}")
def download(job_id: str, fmt: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "Result not ready")
    content = job.get("results", {}).get(fmt)
    if content is None:
        raise HTTPException(404, f"Format '{fmt}' not available for this job")

    doc_name = Path(job["filename"]).stem
    media_type = "application/json" if fmt == "json" else "text/markdown"
    ext = "json" if fmt == "json" else "md"
    return PlainTextResponse(
        content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{doc_name}.{ext}"'},
    )


app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")