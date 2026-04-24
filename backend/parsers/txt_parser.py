"""Plain-text / Markdown parser. Headings detected by ALL CAPS lines (<80 chars)
or Markdown `#` prefixes."""
from __future__ import annotations

from pathlib import Path


def _is_heading(line: str, next_line: str | None) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    if (
        len(stripped) < 80
        and stripped.isupper()
        and any(c.isalpha() for c in stripped)
    ):
        return True
    if next_line is not None and next_line.strip() == "" and len(stripped) < 80:
        # Line followed by blank — treat as heading only if it looks like a title
        # (no trailing punctuation that implies sentence).
        if not stripped.endswith((".", "?", "!", ",", ";", ":")):
            return True
    return False


def parse_txt(filepath: str | Path, source: str) -> list[dict]:
    """Extract heading/text sections from a .txt or .md file."""
    path = Path(filepath)
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()

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

    for i, line in enumerate(lines):
        next_line = lines[i + 1] if i + 1 < len(lines) else None
        if _is_heading(line, next_line):
            flush()
            current_heading = line.strip().lstrip("#").strip()
            current_text = []
        else:
            current_text.append(line)

    flush()
    return sections
