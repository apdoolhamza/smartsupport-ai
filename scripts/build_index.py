"""
SmartSupport AI - Index Builder

Run this once after placing your documents in the documents/ folder.

Usage:
    python scripts/build_index.py

Supported document formats:
    .txt   Plain text
    .md    Markdown
    .pdf   PDF (text-based, not scanned)
    .csv   CSV (each row treated as a document entry)
"""

import sys
import logging
from pathlib import Path

# Allow imports from backend/
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from rag_engine import RAGEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("build_index")

DOCUMENTS_DIR = Path(__file__).parent.parent / "documents"
SUPPORTED     = {".txt", ".md", ".pdf", ".csv"}


def read_file(path: Path, engine: RAGEngine) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return engine.extract_pdf(path)
    return path.read_text(encoding="utf-8", errors="replace")


def main():
    print()
    print("=" * 55)
    print("  SmartSupport AI - Index Builder")
    print("=" * 55)

    if not DOCUMENTS_DIR.exists() or not any(DOCUMENTS_DIR.iterdir()):
        print()
        print("  ERROR: No documents found.")
        print()
        print("  Place your documents in:")
        print("    documents/faq.txt")
        print("    documents/products.txt")
        print("    documents/policies.pdf")
        print("    (any .txt .md .pdf .csv files)")
        print()
        sys.exit(1)

    files = [
        f for f in DOCUMENTS_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED
    ]

    if not files:
        print()
        print(f"  ERROR: No supported files found in documents/")
        print(f"  Supported formats: .txt .md .pdf .csv")
        print()
        sys.exit(1)

    print()
    print(f"  Found {len(files)} document(s):")
    for f in files:
        print(f"    - {f.name}")
    print()

    engine = RAGEngine()
    docs   = []

    for f in files:
        print(f"  Reading: {f.name} ...", end=" ", flush=True)
        text = read_file(f, engine)
        if text.strip():
            docs.append({"text": text, "source": f.name})
            print(f"OK ({len(text):,} characters)")
        else:
            print("SKIPPED (no readable text)")

    if not docs:
        print()
        print("  ERROR: No readable content found in any document.")
        sys.exit(1)

    print()
    print("  Building index...")
    engine.index_documents(docs)
    engine.save()

    stats = engine.stats()
    print()
    print("  Index built successfully.")
    print()
    print(f"  Documents : {len(docs)}")
    print(f"  Chunks    : {stats['total_chunks']}")
    print(f"  Vectors   : {stats['index_vectors']}")
    print()
    print("  You can now start the server:")
    print("    python backend/main.py")
    print()
    print("=" * 55)
    print()

if __name__ == "__main__":
    main()
