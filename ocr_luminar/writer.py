from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .structure import parse_markdown_to_elements


def build_markdown(page_markdowns: list[str]) -> str:
    """Same joining logic as write_markdown, but returns the string directly
    instead of touching disk. Used by the web app.
    """
    parts = []
    for i, md in enumerate(page_markdowns):
        if len(page_markdowns) > 1:
            parts.append(f"<!-- page {i + 1} -->\n\n{md.strip()}")
        else:
            parts.append(md.strip())
    return "\n\n---\n\n".join(parts) + "\n"


def build_json(source_name: str, page_markdowns: list[str], model: str) -> str:
    """Same document shape as write_json, but returns the JSON string directly
    instead of touching disk. Used by the web app.
    """
    pages: list[dict[str, Any]] = []
    for i, md in enumerate(page_markdowns):
        pages.append(
            {
                "page_number": i + 1,
                "markdown": md,
                "elements": parse_markdown_to_elements(md),
            }
        )

    doc = {
        "source_file": source_name,
        "document_name": Path(source_name).stem,
        "model": model,
        "page_count": len(pages),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "pages": pages,
    }
    return json.dumps(doc, indent=2, ensure_ascii=False)


def write_markdown(out_dir: Path, doc_name: str, page_markdowns: list[str]) -> Path:
    out_path = out_dir / f"{doc_name}.md"
    parts = []
    for i, md in enumerate(page_markdowns):
        if len(page_markdowns) > 1:
            parts.append(f"<!-- page {i + 1} -->\n\n{md.strip()}")
        else:
            parts.append(md.strip())
    out_path.write_text("\n\n---\n\n".join(parts) + "\n", encoding="utf-8")
    return out_path


def write_json(out_dir: Path, doc_name: str, source_path: Path, page_markdowns: list[str], model: str) -> Path:
    out_path = out_dir / f"{doc_name}.json"
    pages: list[dict[str, Any]] = []
    for i, md in enumerate(page_markdowns):
        pages.append(
            {
                "page_number": i + 1,
                "markdown": md,
                "elements": parse_markdown_to_elements(md),
            }
        )

    doc = {
        "source_file": str(source_path),
        "document_name": doc_name,
        "model": model,
        "page_count": len(pages),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "pages": pages,
    }
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path