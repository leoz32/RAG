from __future__ import annotations

import hashlib
from datetime import datetime, timezone


def make_experiment_id(prefix: str = "exp") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}"


def make_chunk_id(source_file: str, logical_page: int, index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    safe_source = source_file.replace("/", "_")
    return f"chunk_{safe_source}_{logical_page}_{index}_{digest}"
