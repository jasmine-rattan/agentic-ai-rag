"""BM25 retrieval index — fully offline, no embedding API."""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

CHUNKS_PATH = Path(__file__).parent.parent / "data" / "chunks.json"


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def build_index(chunks: list[dict]) -> None:
    """Persist chunks to data/chunks.json."""
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


def load_index() -> tuple[BM25Okapi, list[dict]]:
    """Load chunks from disk and build a BM25 index. Returns (bm25, chunks)."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "No index found. Run: python wiki.py ingest ./vault/raw"
        )
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    if not chunks:
        raise ValueError("Index is empty. Re-run ingest.")
    tokenized = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25, chunks


def search(query: str, top_k: int = 3) -> list[dict]:
    """Return top_k chunks matching the query, ranked by BM25 score."""
    bm25, chunks = load_index()
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in ranked[:top_k]:
        if score > 0:
            entry = dict(chunks[idx])
            entry["score"] = round(float(score), 4)
            results.append(entry)
    return results
