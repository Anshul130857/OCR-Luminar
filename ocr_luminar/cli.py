"""
OCR Luminar CLI

Usage:
    python -m ocr_luminar.cli --input DOC_OR_FOLDER --output-dir OUT [options]

Examples:
    python -m ocr_luminar.cli --input invoice.pdf --output-dir out
    python -m ocr_luminar.cli --input ./scans/ --output-dir out --format json
    python -m ocr_luminar.cli --input notes.png --output-dir out --model glm-ocr
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .convert import SUPPORTED_EXTS, image_to_png_bytes, is_supported, load_pages
from .engine import GlmOcrEngine, OcrEngineError
from .writer import write_json, write_markdown


def collect_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if not is_supported(input_path):
            raise SystemExit(f"Unsupported file type: {input_path.suffix}. Supported: {sorted(SUPPORTED_EXTS)}")
        return [input_path]

    if input_path.is_dir():
        files = sorted(p for p in input_path.rglob("*") if p.is_file() and is_supported(p))
        if not files:
            raise SystemExit(f"No supported documents found under {input_path}")
        return files

    raise SystemExit(f"Input path does not exist: {input_path}")


def process_document(doc_path: Path, engine: GlmOcrEngine, out_dir: Path, fmt: str, dpi: int, workers: int) -> None:
    print(f"\n[+] {doc_path.name}")
    pages = load_pages(doc_path, dpi=dpi)
    page_markdowns: list[str | None] = [None] * len(pages)

    def do_page(page):
        t0 = time.time()
        png_bytes = image_to_png_bytes(page.image)
        # Same `engine` object is shared across every thread here -- one
        # HTTP session, one model instance in Ollama. Threads only add
        # concurrent *requests*, never a duplicate model or client.
        markdown = engine.transcribe_image(png_bytes)
        return page.index, markdown, time.time() - t0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(do_page, page) for page in pages]
        completed = 0
        for future in as_completed(futures):
            idx, markdown, elapsed = future.result()
            page_markdowns[idx] = markdown
            completed += 1
            print(f"    page {idx + 1}/{len(pages)} done in {elapsed:.1f}s  ({completed}/{len(pages)} complete)")

    doc_name = doc_path.stem
    if fmt in ("md", "both"):
        md_path = write_markdown(out_dir, doc_name, page_markdowns)
        print(f"    -> {md_path}")
    if fmt in ("json", "both"):
        json_path = write_json(out_dir, doc_name, doc_path, page_markdowns, engine.model)
        print(f"    -> {json_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OCR Luminar - PDF/image -> Markdown/JSON via GLM-OCR")
    parser.add_argument("--input", required=True, type=Path, help="File or folder of PDFs/images")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write outputs")
    parser.add_argument("--format", choices=["md", "json", "both"], default="both", help="Output format(s)")
    parser.add_argument("--model", default="glm-ocr", help="Ollama model name (default: glm-ocr)")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--dpi", type=int, default=200, help="Rasterization DPI for PDF pages")
    parser.add_argument("--num-ctx", type=int, default=8192, help="Ollama context window size (tokens)")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent page requests against Ollama (raise on stronger GPUs)")
    parser.add_argument("--skip-check", action="store_true", help="Skip the Ollama/model availability check")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    engine = GlmOcrEngine(model=args.model, base_url=args.ollama_url, num_ctx=args.num_ctx)

    if not args.skip_check:
        try:
            engine.check_available()
        except OcrEngineError as e:
            print(f"[!] {e}", file=sys.stderr)
            return 1

    inputs = collect_inputs(args.input)
    print(f"Found {len(inputs)} document(s) to process.")

    failures = []
    for doc_path in inputs:
        try:
            process_document(doc_path, engine, args.output_dir, args.format, args.dpi, args.workers)
        except Exception as e:  # noqa: BLE001 - keep batch runs going
            print(f"    [!] failed: {e}", file=sys.stderr)
            failures.append((doc_path, str(e)))

    print(f"\nDone. {len(inputs) - len(failures)}/{len(inputs)} succeeded.")
    if failures:
        print("Failures:")
        for p, err in failures:
            print(f"  - {p}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())