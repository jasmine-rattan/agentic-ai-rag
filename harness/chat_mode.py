"""Chat mode: personal assistant with conversation history and selective retrieval."""

import json
import re
from pathlib import Path

from harness.model import call_gemma
from harness.retrieval import search

PERSONA_PATH = Path(__file__).parent.parent / "instructions" / "persona.md"
HISTORY_PATH = Path(__file__).parent.parent / "chat_history.json"
MAX_HISTORY_TURNS = 10

# Keywords that trigger note retrieval
_FACTUAL_KEYWORDS = {"what", "where", "who", "when", "how many", "which", "define",
                     "explain", "describe", "is there", "does", "did", "was", "were"}


def _load_persona() -> str:
    if PERSONA_PATH.exists():
        return PERSONA_PATH.read_text(encoding="utf-8").strip()
    return "You are a helpful personal assistant for Agentic AI coursework."


def _load_history() -> list[dict]:
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, encoding="utf-8") as f:
            history = json.load(f)
        return history[-MAX_HISTORY_TURNS * 2:]  # keep last N turns (2 entries per turn)
    return []


def _save_history(history: list[dict]) -> None:
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def _is_factual(message: str) -> bool:
    """Heuristic: does this message warrant note retrieval?"""
    lower = message.lower()
    for kw in _FACTUAL_KEYWORDS:
        if kw in lower:
            return True
    return False


def _build_prompt(persona: str, history: list[dict], message: str, passages: list[dict]) -> str:
    lines = [persona, ""]

    if history:
        lines.append("Conversation so far:")
        for turn in history:
            role = turn.get("role", "user").capitalize()
            lines.append(f"{role}: {turn['content']}")
        lines.append("")

    if passages:
        lines.append("Relevant notes (cite as [source_path] when referencing):")
        for i, p in enumerate(passages, 1):
            lines.append(f"[{i}] {p['source_path']} — {p['section']}: {p['text'][:300]}")
        lines.append("")
        lines.append(
            "Use the notes above to ground factual claims. Label any suggestions "
            "or plans that are not from the notes as 'Suggestion:'."
        )
        lines.append("")

    lines.append(f"User: {message}")
    lines.append("Assistant:")
    return "\n".join(lines)


def run_chat() -> None:
    """Interactive chat session. Type 'exit' or 'quit' to end."""
    persona = _load_persona()
    history = _load_history()

    print("\nChat mode — personal assistant for your Agentic AI notes.")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEnding chat session.")
            break

        if not message:
            continue
        if message.lower() in {"exit", "quit"}:
            print("Ending chat session.")
            break

        # selective retrieval
        passages = []
        if _is_factual(message):
            try:
                passages = search(message, top_k=3)
            except FileNotFoundError:
                pass  # index not built yet; continue without retrieval

        prompt = _build_prompt(persona, history, message, passages)
        response = call_gemma(prompt)

        print(f"\nAssistant: {response}\n")

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        _save_history(history)
