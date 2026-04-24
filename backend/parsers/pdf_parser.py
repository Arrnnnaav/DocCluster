"""PDF parser using PyMuPDF. Headings detected via font size > 13."""
from __future__ import annotations

from pathlib import Path

import fitz

HEADING_FONT_THRESHOLD = 13.0


def parse_pdf(filepath: str | Path, source: str) -> list[dict]:
    """Extract heading/text sections from a PDF.

    Walks blocks top-to-bottom. Text spans with max font size > 13 start a new
    section; everything after them (until the next heading) becomes that
    section's body text.
    """
    path = Path(filepath)
    sections: list[dict] = []
    current_heading = ""
    current_text: list[str] = []
    current_page = 1

    def flush(page: int) -> None:
        body = "\n".join(current_text).strip()
        if current_heading or body:
            sections.append(
                {
                    "heading": current_heading,
                    "text": body,
                    "page": page,
                    "source": source,
                }
            )

    with fitz.open(path) as doc:
        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("dict").get("blocks", [])
            for block in blocks:
                if block.get("type", 0) != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    line_text = "".join(s.get("text", "") for s in spans).strip()
                    if not line_text:
                        continue
                    max_size = max(s.get("size", 0.0) for s in spans)
                    if max_size > HEADING_FONT_THRESHOLD:
                        flush(current_page)
                        current_heading = line_text
                        current_text = []
                        current_page = page_num
                    else:
                        current_text.append(line_text)

    flush(current_page)
    return sections
