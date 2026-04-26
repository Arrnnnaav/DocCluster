"""Chunker: heading-primary split, paragraph fallback, small-chunk merge.

Also exposes `parse_file`, the single entry point that dispatches to the right
parser based on file extension and returns finished chunks.
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path

MAX_WORDS = 500
MIN_WORDS = 20


def _word_count(text: str) -> int:
    return len(text.split())


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_by_words(text: str, max_words: int) -> list[str]:
    """Hard split on word boundaries when paragraph markers are absent (e.g. raw PDF text)."""
    words = text.split()
    return [
        " ".join(words[i : i + max_words])
        for i in range(0, len(words), max_words)
        if words[i : i + max_words]
    ]


def _make_chunk(text: str, heading: str, source: str, page: int) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "text": text,
        "heading": heading,
        "source": source,
        "page": page,
        "word_count": _word_count(text),
    }


def chunk_sections(sections: list[dict]) -> list[dict]:
    """Turn parser output (list of {heading,text,page,source}) into chunks."""
    intermediate: list[dict] = []

    for sec in sections:
        heading = sec.get("heading", "") or ""
        text = (sec.get("text", "") or "").strip()
        source = sec.get("source", "")
        page = sec.get("page", 1)
        if not text and not heading:
            continue

        combined = text if not heading else (f"{heading}\n{text}" if text else heading)

        if _word_count(combined) <= MAX_WORDS:
            intermediate.append(_make_chunk(combined, heading, source, page))
            continue

        # Oversized section — split by paragraph; fall back to word-based split when
        # the parser (e.g. PDF) emits lines joined with \n rather than \n\n.
        paragraphs = _split_paragraphs(text)
        if len(paragraphs) <= 1:
            paragraphs = _split_by_words(text or combined, MAX_WORDS) or [combined]
        buffer: list[str] = []
        buffer_words = 0
        first = True
        for para in paragraphs:
            para_words = _word_count(para)
            if buffer and buffer_words + para_words > MAX_WORDS:
                body = "\n\n".join(buffer)
                if first and heading:
                    body = f"{heading}\n{body}"
                    first = False
                intermediate.append(_make_chunk(body, heading, source, page))
                buffer = [para]
                buffer_words = para_words
            else:
                buffer.append(para)
                buffer_words += para_words
        if buffer:
            body = "\n\n".join(buffer)
            if first and heading:
                body = f"{heading}\n{body}"
            intermediate.append(_make_chunk(body, heading, source, page))

    # Merge tiny chunks forward.
    merged: list[dict] = []
    i = 0
    while i < len(intermediate):
        chunk = intermediate[i]
        while chunk["word_count"] < MIN_WORDS and i + 1 < len(intermediate):
            nxt = intermediate[i + 1]
            chunk = _make_chunk(
                chunk["text"] + "\n\n" + nxt["text"],
                chunk["heading"] or nxt["heading"],
                chunk["source"],
                chunk["page"],
            )
            i += 1
        merged.append(chunk)
        i += 1

    return merged


def _is_code_chunk(text: str) -> bool:
    """Heuristic: return True when a chunk is mostly code, not prose."""
    t = text.strip()
    signals = [
        t.startswith("import "),
        t.startswith("from "),
        t.startswith("def "),
        t.startswith("class "),
        t.count("\n") > 3 and "    " in t,   # indented block
        t.count("(") + t.count(")") > 8,
    ]
    return sum(signals) >= 2


def parse_file(filepath: str | Path, filename: str) -> list[dict]:
    """Auto-detect file type, parse, and chunk. Returns list of chunk dicts."""
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext == ".pdf":
        from .pdf_parser import parse_pdf
        sections = parse_pdf(path, filename)
    elif ext == ".docx":
        from .docx_parser import parse_docx
        sections = parse_docx(path, filename)
    elif ext in {".txt", ".md", ".markdown"}:
        from .txt_parser import parse_txt
        sections = parse_txt(path, filename)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    result = chunk_sections(sections)
    # Tag code-heavy chunks so topic_modeler can skip Flan-T5 for them
    for c in result:
        c["is_code"] = _is_code_chunk(c["text"])
    return result
