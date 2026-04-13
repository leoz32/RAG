from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, write_json


DEFAULT_INDEX_URL = "https://openstax.org/books/us-history/pages/index"
DEFAULT_OUT_DIR = "data/raw/openstax_us_history_original"
USER_AGENT = "rag-hallucination-eval/0.1 (research skeleton; contact project owner)"


@dataclass
class PageRecord:
    title: str
    slug: str
    url: str


def _fetch_html(url: str, timeout: int = 30) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip().lower())
    return slug.strip("._-") or "untitled"


def _natural_sort_key(value: str) -> list[tuple[int, int | str]]:
    parts = re.split(r"(\d+)", value)
    key: list[tuple[int, int | str]] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part))
    return key


def _discover_page_links(index_html: str, index_url: str) -> list[PageRecord]:
    soup = BeautifulSoup(index_html, "html.parser")
    seen: set[str] = set()
    pages: list[PageRecord] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        absolute_url = urljoin(index_url, href)
        parsed = urlparse(absolute_url)
        if parsed.netloc != "openstax.org":
            continue
        if "/books/us-history/pages/" not in parsed.path:
            continue
        tail = parsed.path.rsplit("/", 1)[-1]
        if tail in {"index", "introduction"}:
            continue
        title = " ".join(anchor.get_text(" ", strip=True).split())
        if not title:
            continue
        if absolute_url in seen:
            continue
        seen.add(absolute_url)
        pages.append(PageRecord(title=title, slug=_slugify(tail), url=absolute_url))
    pages.sort(key=lambda page: _natural_sort_key(page.slug))
    return pages


def _normalize_text(text: str) -> str:
    text = unescape(text)
    text = text.replace("\xa0", " ")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _looks_like_body_heading(text: str) -> bool:
    if not text:
        return False
    if text in {"Contents", "Highlights", "Print", "Search", "Close"}:
        return False
    return True


def _extract_markdown_from_page(page_html: str, page_url: str) -> str:
    soup = BeautifulSoup(page_html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    lines: list[str] = []
    seen_intro = False
    stop_markers = {"Citation/Attribution", "OpenStax’s mission is to make an amazing education accessible for all."}

    for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = " ".join(tag.get_text(" ", strip=True).split())
        if not text:
            continue
        if text in stop_markers:
            break
        if not seen_intro:
            if tag.name in {"h1", "h2"} and _looks_like_body_heading(text):
                seen_intro = True
            else:
                continue
        if tag.name == "h1":
            lines.append(f"# {text}")
        elif tag.name == "h2":
            lines.append(f"## {text}")
        elif tag.name == "h3":
            lines.append(f"### {text}")
        elif tag.name == "h4":
            lines.append(f"#### {text}")
        elif tag.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)

    body = _normalize_text("\n\n".join(lines))
    if not body:
        raise ValueError(f"Failed to extract body text from page: {page_url}")

    header = [
        "<!--",
        "Source: OpenStax U.S. History",
        f"Section URL: {page_url}",
        "NOTE: OpenStax pages include licensing and AI-usage restrictions.",
        "Only ingest this content if you have confirmed your usage is permitted.",
        "-->",
        "",
    ]
    return "\n".join(header) + body + "\n"


def _write_manifest(out_dir: Path, pages: Iterable[PageRecord]) -> Path:
    manifest_path = out_dir / "manifest.json"
    write_json(
        manifest_path,
        {
            "source": "OpenStax U.S. History",
            "index_url": DEFAULT_INDEX_URL,
            "page_count": len(list(pages)),
            "pages": [page.__dict__ for page in pages],
        },
    )
    return manifest_path


def _load_fetch_state(path: Path) -> dict:
    if not path.exists():
        return {"completed": [], "failed": {}}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_fetch_state(path: Path, payload: dict) -> None:
    write_json(path, payload)


def _require_permission(args: argparse.Namespace) -> None:
    if args.discover_only:
        return
    if args.i_have_permission:
        return
    raise SystemExit(
        "Refusing to fetch textbook body content without explicit confirmation. "
        "Re-run with --i-have-permission only if you have confirmed your OpenStax usage is allowed."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Discover OpenStax U.S. History section links and, with explicit permission confirmation, "
            "download section pages into Markdown files suitable for later ingestion."
        )
    )
    parser.add_argument("--index-url", default=DEFAULT_INDEX_URL)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N discovered pages.")
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only build a manifest of section URLs without downloading page bodies.",
    )
    parser.add_argument(
        "--i-have-permission",
        action="store_true",
        help="Required to fetch and write textbook body content.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing Markdown files instead of skipping them.",
    )
    args = parser.parse_args()

    _require_permission(args)

    out_dir = ensure_dir(ROOT / args.out_dir)
    html = _fetch_html(args.index_url)
    pages = _discover_page_links(html, args.index_url)
    if args.limit > 0:
        pages = pages[: args.limit]

    manifest_path = _write_manifest(out_dir, pages)
    state_path = out_dir / "fetch_state.json"
    state = _load_fetch_state(state_path)
    completed = set(state.get("completed", []))
    failed = dict(state.get("failed", {}))
    print(f"Discovered {len(pages)} section pages")
    print(f"Manifest written to {manifest_path}")

    if args.discover_only:
        print("Discovery only mode enabled; no page content downloaded.")
        return

    for page in pages:
        target_path = out_dir / f"{page.slug}.md"
        if target_path.exists() and not args.force:
            completed.add(page.slug)
            print(f"Skip existing {target_path}")
            continue

        try:
            page_html = _fetch_html(page.url)
            markdown = _extract_markdown_from_page(page_html, page.url)
            target_path.write_text(markdown, encoding="utf-8")
            completed.add(page.slug)
            failed.pop(page.slug, None)
            _save_fetch_state(
                state_path,
                {
                    "completed": sorted(completed, key=_natural_sort_key),
                    "failed": failed,
                },
            )
            print(f"Wrote {target_path}")
        except Exception as exc:  # pragma: no cover
            failed[page.slug] = {
                "url": page.url,
                "error": str(exc),
            }
            _save_fetch_state(
                state_path,
                {
                    "completed": sorted(completed, key=_natural_sort_key),
                    "failed": failed,
                },
            )
            print(f"Failed {page.url}: {exc}")


if __name__ == "__main__":
    main()
