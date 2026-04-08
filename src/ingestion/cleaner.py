from __future__ import annotations

import re


WHITESPACE_PATTERN = re.compile(r"[ \t]+")
MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    lines = [WHITESPACE_PATTERN.sub(" ", line).strip() for line in text.replace("\r\n", "\n").split("\n")]
    cleaned = "\n".join(lines)
    cleaned = MULTI_NEWLINE_PATTERN.sub("\n\n", cleaned)
    return cleaned.strip()
