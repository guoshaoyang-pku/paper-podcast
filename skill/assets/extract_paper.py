#!/usr/bin/env python3
"""Extract full text and images from a paper PDF into extracted/<slug>/.

Usage:
    python3 extract_paper.py <path-to-pdf> [<slug>]

If slug is omitted it is derived from the filename (everything before the
first '_').
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = ROOT / "extracted"


def slug_from_filename(name: str) -> str:
    base = Path(name).stem
    # take part before first underscore if it looks like arxiv id, else whole
    return base.split("_", 1)[0] if "_" in base else base


def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "paper"


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    pdf = Path(sys.argv[1]).resolve()
    if not pdf.exists():
        sys.exit(f"PDF not found: {pdf}")
    slug = sys.argv[2] if len(sys.argv) > 2 else slug_from_filename(pdf.name)
    slug = slugify(slug)
    out_dir = EXTRACTED / slug
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf)
    full_text_parts: list[str] = []
    img_index = 0
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    for i, page in enumerate(doc, 1):
        text = page.get_text("text")
        full_text_parts.append(f"\n\n===== PAGE {i} =====\n{text}")
        # render whole page to PNG (for Figure references)
        try:
            pix = page.get_pixmap(dpi=150)
            pix.save((pages_dir / f"p{i:02d}.png").as_posix())
        except Exception as e:  # pragma: no cover
            print(f"  page render skip p{i}: {e}", file=sys.stderr)
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha >= 4:  # CMYK -> RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_index += 1
                img_path = out_dir / "images" / f"p{i:02d}_{img_index:03d}.png"
                pix.save(img_path.as_posix())
                pix = None
            except Exception as e:  # pragma: no cover
                print(f"  image skip p{i}: {e}", file=sys.stderr)

    (out_dir / "full_text.txt").write_text("".join(full_text_parts), encoding="utf-8")
    meta = {
        "source_pdf": str(pdf),
        "pages": doc.page_count,
        "images_extracted": img_index,
    }
    (out_dir / "meta.json").write_text(
        __import__("json").dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"OK {slug}: {doc.page_count} pages, {img_index} images -> {out_dir}")


if __name__ == "__main__":
    main()
