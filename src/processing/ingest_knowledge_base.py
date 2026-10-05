"""
CLI script to ingest knowledge base documents into the local FAISS vector store.
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.services.rag_service import RAGService


def main():
    print("=" * 60)
    print("PCB Knowledge Base Ingestion Pipeline")
    print("=" * 60)

    rag = RAGService(use_offline_fallback=False)
    print(f"Scanning knowledge base directory: {rag.kb_path}")
    print(f"Vector store destination: {rag.vectorstore_dir}")

    result = rag.ingest_knowledge_base(force_rebuild=True)
    print("\nIngestion Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    stats = rag.get_collection_stats()
    print("\nVector Store Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Run verification queries
    test_queries = [
        "What are possible causes of a broken copper PCB track?",
        "What is the recommended rework procedure for a missing component?",
        "LM1117 regulator capacitor ESR and stability requirements",
    ]

    print("\n" + "=" * 60)
    print("Testing Semantic Retrieval Queries")
    print("=" * 60)

    for query in test_queries:
        print(f"\n[QUERY]: {query}")
        results = rag.query(query, top_k=2)
        print(f"Retrieved {len(results)} chunks:")
        for idx, r in enumerate(results, 1):
            snippet = r['text'][:140].replace('\n', ' ')
            print(f"  [{idx}] Source: {r['source']} ({r['category']}) | Score: {r['similarity_score']}")
            print(f"      Title: {r['title']}")
            print(f"      Snippet: {snippet}...")


if __name__ == "__main__":
    main()
