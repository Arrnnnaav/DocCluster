"""DOCX parser using python-docx. Headings detected via paragraph style name."""
from __future__ import annotations

from pathlib import Path

from docx import Document


def parse_docx(filepath: str | Path, source: str) -> list[dict]:
    """Extract heading/text sections from a DOCX.

    Any paragraph whose style name contains "Heading" (case-insensitive) starts
    a new section. All subsequent paragraphs accumulate as body text until the
    next heading.
    """
    doc = Document(str(filepath))
    sections: list[dict] = []
    current_heading = ""
    current_text: list[str] = []

    def flush() -> None:
        body = "\n".join(current_text).strip()
        if current_heading or body:
            sections.append(
                {
                    "heading": current_heading,
                    "text": body,
                    "page": 1,
                    "source": source,
                }
            )

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = (para.style.name or "") if para.style else ""
        if "heading" in style_name.lower():
            flush()
            current_heading = text
            current_text = []
        else:
            current_text.append(text)

    flush()
    return sections
