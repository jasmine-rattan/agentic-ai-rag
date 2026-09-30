#!/usr/bin/env python3
"""Personal Wiki CLI — talk to your Agentic AI notes using local Gemma + RAG."""

import argparse
import sys


BANNER = """
╔══════════════════════════════════════════════════════╗
║  Personal Wiki — Agentic AI Notes                   ║
║  Model : gemma3:4b (Ollama, localhost:11434)         ║
║  Retrieval: BM25 (offline, no embedding API)         ║
╚══════════════════════════════════════════════════════╝
"""


def cmd_ingest(args):
    from harness.ingest import ingest
    print(BANNER)
    print(f"Ingesting sources from: {args.source_dir}\n")
    ingest(args.source_dir)


def cmd_search(args):
    from harness.search_mode import run_search
    run_search(args.query, top_k=args.top_k)


def cmd_ask(args):
    from harness.ask_mode import run_ask
    print(BANNER)
    run_ask(args.question, top_k=3)


def cmd_chat(args):
    from harness.chat_mode import run_chat
    print(BANNER)
    run_chat()


def main():
    parser = argparse.ArgumentParser(
        prog="wiki",
        description="Personal Wiki: local Gemma + BM25 RAG over your Agentic AI notes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  ingest <dir>   Chunk sources, generate wiki pages, build BM25 index
  search <query> Retrieve matching passages (no model call)
  ask <question> RAG: retrieve + Gemma answer + citations
  chat           Interactive personal assistant session

Examples:
  python wiki.py ingest ./vault/raw
  python wiki.py search "agent memory"
  python wiki.py ask "What is retrieval augmented generation?"
  python wiki.py chat

Setup (one-time):
  brew install ollama && ollama pull gemma3:4b
  pip install -r requirements.txt
  python wiki.py ingest ./vault/raw
""",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="command")
    subparsers.required = True

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest sources and build index")
    p_ingest.add_argument("source_dir", help="Directory containing source files (vault/raw)")
    p_ingest.set_defaults(func=cmd_ingest)

    # search
    p_search = subparsers.add_parser("search", help="Search passages (no model call)")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")
    p_search.set_defaults(func=cmd_search)

    # ask
    p_ask = subparsers.add_parser("ask", help="RAG: retrieve + Gemma + citations")
    p_ask.add_argument("question", help="Question to answer")
    p_ask.add_argument(
        "--mode", choices=["local"], default="local",
        help="Inference mode (default: local via Ollama)"
    )
    p_ask.set_defaults(func=cmd_ask)

    # chat
    p_chat = subparsers.add_parser("chat", help="Interactive personal assistant")
    p_chat.set_defaults(func=cmd_chat)

    args = parser.parse_args()

    try:
        args.func(args)
    except RuntimeError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
