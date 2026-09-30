"""Search mode: BM25 passage retrieval, no model call."""

from harness.retrieval import search


def run_search(query: str, top_k: int = 5) -> None:
    """Print top-k BM25 passages matching query. No model call."""
    print(f"\nSearch: \"{query}\"\n{'─' * 60}")

    results = search(query, top_k=top_k)
    if not results:
        print("No matching passages found.")
        return

    for i, chunk in enumerate(results, 1):
        print(f"\n[{i}] Score: {chunk['score']:.4f}")
        print(f"    Source: {chunk['source_path']}")
        print(f"    Section: {chunk['section']}")
        print(f"    {'─' * 50}")
        # show up to 300 chars of passage
        preview = chunk["text"][:300].strip()
        if len(chunk["text"]) > 300:
            preview += "..."
        print(f"    {preview}")
    print()
