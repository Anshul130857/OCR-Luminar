"""
Thin client around a local Ollama instance running the `glm-ocr` model.

GLM-OCR is a vision-language model that natively emits structured, semantic
Markdown (headings, tables, fenced code blocks, LaTeX-style formulas) rather
than flat text -- so we lean on that instead of trying to reinvent layout
detection ourselves. We just prompt it per-page and collect the Markdown.
"""

from __future__ import annotations

import base64
import time

import requests

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "glm-ocr"

# Kept deliberately explicit about the output contract: GLM-OCR is capable of
# code, tables, formulas and prose in one pass, so we ask for all of it in a
# single structured Markdown response rather than making multiple calls.
OCR_PROMPT = (
    "You are an OCR and document-structure engine. Transcribe this page image "
    "completely and faithfully into Markdown.\n"
    "Rules:\n"
    "- Preserve reading order and heading levels (#, ##, ###).\n"
    "- Reproduce tables as GitHub-flavored Markdown tables.\n"
    "- Reproduce any source code exactly, inside fenced code blocks with the "
    "correct language tag if identifiable.\n"
    "- Reproduce mathematical formulas as LaTeX inside $...$ or $$...$$.\n"
    "- For figures/photos/diagrams that contain no legible text, insert a short "
    "bracketed description like [Figure: bar chart of quarterly revenue].\n"
    "- Do not summarize, translate, or omit content. Do not add commentary "
    "outside the transcription.\n"
    "- Output ONLY the Markdown transcription, nothing else."
)


class OcrEngineError(RuntimeError):
    pass


class GlmOcrEngine:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: int = 300,
        max_retries: int = 2,
        num_ctx: int = 8192,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.num_ctx = num_ctx
        # One shared session/connection pool for the whole process. Threads
        # calling transcribe_image() concurrently reuse this -- there is no
        # per-thread or per-call model/client duplication. The model itself
        # lives once in Ollama's process (kept resident via keep_alive); this
        # object is just an HTTP client wrapper around that single instance.
        self.session = requests.Session()

    def check_available(self) -> None:
        """Raise a clear error early if Ollama or the model isn't reachable."""
        try:
            resp = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise OcrEngineError(
                f"Could not reach Ollama at {self.base_url}. "
                f"Is it running? (`ollama serve`). Underlying error: {e}"
            ) from e

        names = {m.get("name", "").split(":")[0] for m in resp.json().get("models", [])}
        if self.model not in names:
            raise OcrEngineError(
                f"Model '{self.model}' not found in Ollama. Pull it first with:\n"
                f"  ollama pull {self.model}"
            )

    def transcribe_image(self, png_bytes: bytes) -> str:
        """Send one page image to GLM-OCR and return Markdown text."""
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": OCR_PROMPT, "images": [b64]},
            ],
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0.0, "num_ctx": self.num_ctx},
        }

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                resp = self.session.post(
                    f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
                )
                if not resp.ok:
                    raise OcrEngineError(
                        f"{resp.status_code} {resp.reason} from Ollama: {resp.text[:500]}"
                    )
                data = resp.json()
                content = data.get("message", {}).get("content", "").strip()
                if not content:
                    raise OcrEngineError("Model returned empty content.")
                return content
            except (requests.RequestException, OcrEngineError) as e:
                last_err = e
                if attempt <= self.max_retries:
                    time.sleep(1.5 * attempt)
                    continue
        raise OcrEngineError(f"OCR call failed after retries: {last_err}") from last_err