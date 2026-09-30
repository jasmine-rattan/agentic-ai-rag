"""Ingestion pipeline: chunk sources, generate wiki pages, update index."""

import json
import re
import uuid
from pathlib import Path

from harness.model import call_gemma
from harness.retrieval import build_index

VAULT_WIKI = Path(__file__).parent.parent / "vault" / "wiki"
VAULT_INDEX = Path(__file__).parent.parent / "vault" / "index.md"
CHUNKS_PATH = Path(__file__).parent.parent / "data" / "chunks.json"

CHUNK_WORDS = 400
OVERLAP_WORDS = 50


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _read_file(path: Path) -> str:
    """Return plain text from .txt, .md, or .pdf files."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        try:
            from pdfminer.high_level import extract_text
            return extract_text(str(path))
        except ImportError:
            raise ImportError(
                "pdfminer.six is required for PDF support: pip install pdfminer.six"
            )
    raise ValueError(f"Unsupported file type: {suffix}")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _detect_section(text: str, char_offset: int) -> str:
    """Find the nearest Markdown heading before char_offset."""
    snippet = text[:char_offset]
    headings = re.findall(r"^#{1,6}\s+(.+)", snippet, re.MULTILINE)
    return headings[-1].strip() if headings else "Introduction"


def _chunk_text(text: str, source_path: str) -> list[dict]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_WORDS, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        # rough char offset for section detection
        char_offset = len(" ".join(words[:start]))
        section = _detect_section(text, char_offset)
        chunks.append({
            "id": str(uuid.uuid4()),
            "source_path": source_path,
            "section": section,
            "text": chunk_text,
        })
        if end == len(words):
            break
        start += CHUNK_WORDS - OVERLAP_WORDS
    return chunks


# ---------------------------------------------------------------------------
# Wiki page generation
# ---------------------------------------------------------------------------

_WIKI_PROMPT = """\
You are a knowledge-base builder. Given the source text below, write a concise \
wiki page in Markdown.

Rules:
- First line must be a heading: # <2-6 word natural topic title>
- Include: Summary section, Key Concepts section, a source reference line \
  formatted exactly as: [[source: {source_path}]]
- End with a Related Notes section listing 2-3 related topic names as \
  [[Topic Name]] wiki links.
- Be factual and neutral. Do not invent information not present in the source.
- Keep total length under 600 words.

Source text:
---
{text}
---
"""

_FILENAME_PROMPT = """\
Given this wiki page heading, produce a short (2-6 word) filename with title \
case, no special characters, no hashes. Reply with ONLY the filename (no \
extension).

Heading: {heading}
"""


def _generate_wiki_page(text: str, source_path: str) -> tuple[str, str]:
    """Call Gemma to generate wiki page content. Returns (filename, content)."""
    prompt = _WIKI_PROMPT.format(source_path=source_path, text=text[:6000])
    content = call_gemma(prompt)

    # extract heading from first line
    first_line = content.split("\n")[0].strip()
    heading = re.sub(r"^#+\s*", "", first_line)

    # ask model for clean filename
    fname_prompt = _FILENAME_PROMPT.format(heading=heading)
    filename = call_gemma(fname_prompt).strip()
    # sanitize just in case
    filename = re.sub(r"[^\w\s\-]", "", filename).strip()
    filename = re.sub(r"\s+", " ", filename)
    if not filename:
        filename = heading[:50]

    return filename, content


# ---------------------------------------------------------------------------
# Index update
# ---------------------------------------------------------------------------

def _update_index(entries: list[dict]) -> None:
    """Rewrite vault/index.md with all wiki page links grouped by source."""
    lines = ["# Wiki Index\n"]
    by_source: dict[str, list[str]] = {}
    for e in entries:
        src = Path(e["source_path"]).name
        by_source.setdefault(src, []).append(e["wiki_file"])

    for src, pages in sorted(by_source.items()):
        lines.append(f"\n## {src}\n")
        for page in sorted(pages):
            stem = Path(page).stem
            lines.append(f"- [[{stem}]]\n")

    VAULT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    VAULT_INDEX.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def ingest(source_dir: str) -> None:
    """Ingest all supported files from source_dir into the wiki + BM25 index."""
    src_path = Path(source_dir)
    if not src_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    supported = list(src_path.glob("*.md")) + list(src_path.glob("*.txt")) + list(src_path.glob("*.pdf"))
    if not supported:
        print(f"No supported files (.md, .txt, .pdf) found in {source_dir}")
        return

    # Load existing chunks, keyed by source_path for deduplication
    existing_chunks: list[dict] = []
    if CHUNKS_PATH.exists():
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            existing_chunks = json.load(f)
    existing_sources = {c["source_path"] for c in existing_chunks}

    new_chunks: list[dict] = list(existing_chunks)
    wiki_entries: list[dict] = []

    # Load existing wiki entries from index
    if VAULT_INDEX.exists():
        content = VAULT_INDEX.read_text(encoding="utf-8")
        for match in re.finditer(r"\[\[(.+?)\]\]", content):
            title = match.group(1)
            wiki_entries.append({
                "source_path": "unknown",
                "wiki_file": f"{title}.md",
            })

    VAULT_WIKI.mkdir(parents=True, exist_ok=True)

    for file_path in sorted(supported):
        rel_path = str(file_path)
        print(f"  Ingesting: {file_path.name}")

        text = _read_file(file_path)

        # chunk (always re-chunk to keep index fresh)
        file_chunks = _chunk_text(text, rel_path)

        # remove old chunks for this source
        new_chunks = [c for c in new_chunks if c["source_path"] != rel_path]
        new_chunks.extend(file_chunks)

        # generate wiki page
        print(f"    Generating wiki page via Gemma...")
        try:
            filename, wiki_content = _generate_wiki_page(text, rel_path)
        except RuntimeError as e:
            print(f"    [ERROR] {e}")
            continue

        wiki_file = VAULT_WIKI / f"{filename}.md"
        wiki_file.write_text(wiki_content, encoding="utf-8")
        print(f"    Saved: vault/wiki/{filename}.md")

        wiki_entries.append({
            "source_path": rel_path,
            "wiki_file": f"{filename}.md",
        })

    build_index(new_chunks)
    print(f"\n  Index updated: {len(new_chunks)} chunks from {len(supported)} file(s)")

    _update_index(wiki_entries)
    print("  vault/index.md updated")
