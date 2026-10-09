"""
CLI — interactive terminal chatbot interface.
"""

from campusmate.rag_pipeline import (
    load_or_build_index, build_full_index,
    answer_question, get_indexed_sources,
)


HELP_TEXT = """
Commands:
  /help     — Show this help message
  /sources  — List indexed PDF files
  /reindex  — Rebuild the PDF index
  /exit     — Exit CampusMate
"""


def print_banner():
    """Print the startup banner."""
    print()
    print("=" * 50)
    print("  CampusMate — College Assistant")
    print("=" * 50)
    print()


def format_sources(chunks):
    """Format source citations from retrieved chunks."""
    seen = set()
    lines = []
    for i, chunk in enumerate(chunks, 1):
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            seen.add(key)
            lines.append(f"  [{i}] {chunk['source']}, Page {chunk['page']}")
    return "\n".join(lines)


def run_cli():
    """Main CLI loop."""
    print_banner()

    # Load embedding model & index (no LLM call here)
    print("Loading embedding model...")
    index, metadata = load_or_build_index()
    print()
    print("CampusMate is ready! Type /help for commands.")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() == "/exit":
            print("Goodbye!")
            break

        elif user_input.lower() == "/help":
            print(HELP_TEXT)

        elif user_input.lower() == "/sources":
            sources = get_indexed_sources(metadata)
            if sources:
                print("\nAvailable PDFs:")
                for i, src in enumerate(sources, 1):
                    print(f"  {i}. {src}")
            else:
                print("  No PDFs indexed yet.")
            print()

        elif user_input.lower() == "/reindex":
            print("Rebuilding the PDF index...")
            index, metadata = build_full_index()
            print("Done!\n")

        else:
            # Answer the question
            print()
            answer, chunks = answer_question(user_input, index, metadata)
            print(f"CampusMate:\n{answer}")
            if chunks:
                print(f"\nSources:\n{format_sources(chunks)}")
            print()
