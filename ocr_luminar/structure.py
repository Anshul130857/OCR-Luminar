"""
Parses GLM-OCR's Markdown output into a structured element list for JSON
output. Kept as lightweight regex/line-scanning rather than a full Markdown
AST library, since GLM-OCR's output is already clean, constrained Markdown
(we control the prompt contract in engine.py).
"""

from __future__ import annotations

import re
from typing import Any

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_CODE_FENCE_RE = re.compile(r"^```(\S*)\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
_FIGURE_RE = re.compile(r"^\[Figure:\s*(.*)\]$", re.IGNORECASE)


def parse_markdown_to_elements(markdown: str) -> list[dict[str, Any]]:
    """Turn a Markdown string into an ordered list of typed elements:
    heading, paragraph, code, table, figure.
    """
    lines = markdown.splitlines()
    elements: list[dict[str, Any]] = []
    i = 0
    buffer: list[str] = []

    def flush_paragraph():
        text = "\n".join(buffer).strip()
        if text:
            elements.append({"type": "paragraph", "text": text})
        buffer.clear()

    while i < len(lines):
        line = lines[i]

        # Fenced code block
        fence = _CODE_FENCE_RE.match(line)
        if fence:
            flush_paragraph()
            lang = fence.group(1) or None
            code_lines = []
            i += 1
            while i < len(lines) and not _CODE_FENCE_RE.match(lines[i]):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            elements.append({"type": "code", "language": lang, "text": "\n".join(code_lines)})
            continue

        # Heading
        heading = _HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            elements.append({"type": "heading", "level": len(heading.group(1)), "text": heading.group(2).strip()})
            i += 1
            continue

        # Figure placeholder
        figure = _FIGURE_RE.match(line.strip())
        if figure:
            flush_paragraph()
            elements.append({"type": "figure", "description": figure.group(1).strip()})
            i += 1
            continue

        # Table (GFM): a row line followed by a separator line
        if _TABLE_ROW_RE.match(line) and i + 1 < len(lines) and _TABLE_SEP_RE.match(lines[i + 1]):
            flush_paragraph()
            table_lines = [line, lines[i + 1]]
            i += 2
            while i < len(lines) and _TABLE_ROW_RE.match(lines[i]):
                table_lines.append(lines[i])
                i += 1
            elements.append({"type": "table", "rows": _parse_table(table_lines), "markdown": "\n".join(table_lines)})
            continue

        # Blank line -> paragraph break
        if not line.strip():
            flush_paragraph()
            i += 1
            continue

        buffer.append(line)
        i += 1

    flush_paragraph()
    return elements


def _parse_table(table_lines: list[str]) -> list[list[str]]:
    rows = []
    for idx, raw in enumerate(table_lines):
        if idx == 1:
            continue  # skip the --- separator row
        cells = [c.strip() for c in raw.strip().strip("|").split("|")]
        rows.append(cells)
    return rows