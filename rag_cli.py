"""
CLI version of the RAG chatbot — no Streamlit needed.
Usage:
    python rag_cli.py --pdf path/to/file.pdf [--embed huggingface] [--llm groq]
"""

import argparse
import os
from app import build_qa_chain


def main():
    parser = argparse.ArgumentParser(description="RAG Chatbot CLI")
    parser.add_argument("--pdf",   required=True,             help="Path to PDF file")
    parser.add_argument("--embed", default="huggingface",     help="Embedding provider: huggingface | openai")
    parser.add_argument("--llm",   default="groq",            help="LLM provider: groq | openai | gemini")
    args = parser.parse_args()

    print(f"\nLoading '{args.pdf}' with {args.embed} embeddings + {args.llm} LLM…")
    qa = build_qa_chain(args.pdf, args.embed, args.llm)
    print("Index ready. Type 'quit' to exit.\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        result = qa.invoke({"query": q})
        print(f"Bot: {result['result']}\n")


if __name__ == "__main__":
    main()
