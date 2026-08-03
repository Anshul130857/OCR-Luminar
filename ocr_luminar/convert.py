"""
Converts input documents (PDF or raster images) into a list of in-memory
page images ready to hand to the OCR engine.

Supported inputs:
  - .pdf            -> rendered to one image per page via pdf2image (needs poppler)
  - .png/.jpg/.jpeg/.tif/.tiff/.bmp/.webp -> loaded directly as a single "page"
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXTS = {".pdf"}
SUPPORTED_EXTS = IMAGE_EXTS | PDF_EXTS


@dataclass
class Page:
    index: int          # 0-based page number
    image: Image.Image  # PIL image for this page


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTS


def load_pages(path: Path, dpi: int = 200) -> list[Page]:
    """Load a document into a list of Page objects (1 page for images, N for PDFs)."""
    ext = path.suffix.lower()

    if ext in IMAGE_EXTS:
        img = Image.open(path).convert("RGB")
        return [Page(index=0, image=img)]

    if ext in PDF_EXTS:
        # Imported lazily so the CLI still works on machines without poppler
        # installed if the user is only OCR'ing images.
        from pdf2image import convert_from_path

        images = convert_from_path(str(path), dpi=dpi)
        return [Page(index=i, image=img.convert("RGB")) for i, img in enumerate(images)]

    raise ValueError(f"Unsupported file type: {ext}. Supported: {sorted(SUPPORTED_EXTS)}")


def image_to_png_bytes(img: Image.Image, max_dimension: int = 2200) -> bytes:
    """Downscale (if needed) and serialize a PIL image to PNG bytes for the OCR model."""
    w, h = img.size
    scale = min(1.0, max_dimension / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()