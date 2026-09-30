"""Ask mode: RAG + citations + evidence cards."""

import json
import re
from datetime import datetime
from pathlib import Path

from harness.model import call_gemma
from harness.retrieval import search

INSTRUCTIONS_PATH = Path(__file__).parent.parent / "instructions" / "wiki-instructions.md"
EVIDENCE_DIR = Path(__file__).parent.parent / "evidence"


def _load_instructions() -> str:
    if INSTRUCTIONS_PATH.exists():
        return INSTRUCTIONS_PATH.read_text(encoding="utf-8").strip()
    return "Answer the question using only the provided sources. Cite sources inline."


def _build_prompt(question: str, passages: list[dict]) -> str:
    context_lines = []
    for i, p in enumerate(passages, 1):
        context_lines.append(
            f"[{i}] Source: {p['source_path']} | Section: {p['section']}\n{p['text']}"
        )
    context = "\n\n".join(context_lines)
    return (
        f"Question: {question}\n\n"
        f"Retrieved passages:\n{context}\n\n"
        "Answer the question using only the passages above. "
        "Cite sources inline using [1], [2], [3] notation. "
        "If the passages do not contain sufficient evidence, reply with: "
        "\"Insufficient evidence in the available notes to answer this question.\""
    )


def _extract_citations(answer: str, passages: list[dict]) -> list[str]:
    """Extract [N] citations from answer and map to source paths."""
    nums = re.findall(r"\[(\d+)\]", answer)
    cited_sources = []
    for n in nums:
        idx = int(n) - 1
        if 0 <= idx < len(passages):
            src = passages[idx]["source_path"]
            if src not in cited_sources:
                cited_sources.append(src)
    return cited_sources


def _save_evidence(question: str, passages: list[dict], answer: str, citations: list[str]) -> Path:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^\w]+", "_", question.lower())[:40].strip("_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = EVIDENCE_DIR / f"ask_{timestamp}_{slug}.json"
    card = {
        "timestamp": timestamp,
        "question": question,
        "retrieved_passages": passages,
        "answer": answer,
        "citations": citations,
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, ensure_ascii=False)
    return filename


def run_ask(question: str, top_k: int = 3) -> None:
    """Retrieve passages, call Gemma, display answer with citations, save evidence card."""
    print(f"\nQuestion: {question}\n{'─' * 60}")

    system = _load_instructions()
    passages = search(question, top_k=top_k)

    if not passages:
        answer = "Insufficient evidence in the available notes to answer this question."
        print(f"\nAnswer:\n{answer}\n")
        _save_evidence(question, [], answer, [])
        return

    prompt = _build_prompt(question, passages)
    print("Retrieving and generating answer...\n")
    answer = call_gemma(prompt, system=system)

    citations = _extract_citations(answer, passages)

    print(f"Answer:\n{answer}\n")
    if citations:
        print(f"Sources cited: {', '.join(citations)}")

    card_path = _save_evidence(question, passages, answer, citations)
    print(f"\nEvidence card saved: {card_path.name}")
