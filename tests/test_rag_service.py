"""
Unit and integration tests for RAGService, document ingestion, and FAISS retrieval.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from src.services.rag_service import (
    DOCUMENT_CATALOG,
    DocumentChunk,
    RAGService,
    chunk_text,
    clean_text,
    extract_text_from_file,
)


class TestRAGService(unittest.TestCase):
    """Test suite for RAG document loader, chunker, indexer, and retriever."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.kb_dir = Path(__file__).resolve().parent.parent / "knowledge_base"

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_clean_text(self):
        """Test text cleaning function."""
        raw = "Line 1\x00\x08   \n\n\n\nLine 2   with   extra  spaces.\n"
        cleaned = clean_text(raw)
        self.assertNotIn("\x00", cleaned)
        self.assertEqual(cleaned, "Line 1\n\nLine 2 with extra spaces.")

    def test_chunk_text_basic(self):
        """Test text chunking with paragraph and sentence boundaries."""
        sample_text = (
            "Section 1: PCB trace repair procedures.\n\n"
            "Paragraph 1 discusses conductor restoration using jumper wires according to IPC-7721 standards. "
            "It requires cleaning the surface, pre-tinning the pads, and securing the wire with epoxy adhesive.\n\n"
            "Paragraph 2 discusses thermal stress and trace blowout under high current conditions."
        )
        chunks = chunk_text(sample_text, chunk_size=150, chunk_overlap=30)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIsInstance(chunk, str)
            self.assertGreater(len(chunk), 0)

    def test_extract_text_from_markdown(self):
        """Test extracting text from markdown file in knowledge base."""
        md_file = self.kb_dir / "repair_guides" / "ipc_7721_procedure_4_2_3_jumper_wire_reference.md"
        self.assertTrue(md_file.is_file(), f"File missing: {md_file}")
        text = extract_text_from_file(md_file)
        self.assertIn("IPC-7721", text)
        self.assertIn("Procedure 4.2.3", text)

    def test_extract_text_from_pdf(self):
        """Test extracting text from sample PDF in knowledge base."""
        pdf_file = self.kb_dir / "datasheets" / "ti_lm1117_datasheet.pdf"
        self.assertTrue(pdf_file.is_file(), f"PDF missing: {pdf_file}")
        text = extract_text_from_file(pdf_file)
        self.assertGreater(len(text), 1000)
        self.assertIn("LM1117", text)

    def test_load_and_chunk_all_kb_documents(self):
        """Test loading and chunking all 11 knowledge base documents."""
        rag = RAGService(
            vectorstore_dir=str(self.test_dir / "vectorstore"),
            kb_path=str(self.kb_dir),
            use_offline_fallback=True,
        )
        chunks = rag.load_and_chunk_documents()
        self.assertGreaterEqual(len(chunks), 50)

        # Verify all 11 cataloged documents are represented
        sources_found = set(c.source for c in chunks)
        for doc_name in DOCUMENT_CATALOG.keys():
            self.assertIn(
                doc_name,
                sources_found,
                f"Document {doc_name} was not found in loaded chunks",
            )

        # Verify chunk metadata integrity
        first_chunk = chunks[0]
        self.assertIsNotNone(first_chunk.title)
        self.assertIsNotNone(first_chunk.category)
        self.assertIsNotNone(first_chunk.publisher)
        self.assertIsNotNone(first_chunk.doc_type)

    def test_vectorstore_ingestion_and_persistence(self):
        """Test building FAISS index, saving, and reloading from disk."""
        vs_dir = self.test_dir / "vs_test"
        rag = RAGService(
            vectorstore_dir=str(vs_dir),
            kb_path=str(self.kb_dir),
            use_offline_fallback=True,
        )

        res = rag.ingest_knowledge_base(force_rebuild=True)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["indexed_documents"], 11)
        self.assertGreater(res["total_chunks"], 100)

        stats = rag.get_collection_stats()
        self.assertEqual(stats["unique_documents"], 11)
        self.assertGreater(stats["total_chunks"], 0)

        # Verify disk persistence
        index_file = vs_dir / "faiss_index.bin"
        meta_file = vs_dir / "chunks_metadata.json"
        self.assertTrue(index_file.is_file())
        self.assertTrue(meta_file.is_file())

        # Test reloading from disk
        rag_reloaded = RAGService(
            vectorstore_dir=str(vs_dir),
            kb_path=str(self.kb_dir),
            use_offline_fallback=True,
        )
        self.assertEqual(len(rag_reloaded.chunks), res["total_chunks"])
        self.assertIsNotNone(rag_reloaded.index)

    def test_semantic_query_retrieval(self):
        """Test query retrieval and similarity scoring."""
        vs_dir = self.test_dir / "vs_query_test"
        rag = RAGService(
            vectorstore_dir=str(vs_dir),
            kb_path=str(self.kb_dir),
            use_offline_fallback=True,
        )
        rag.ingest_knowledge_base(force_rebuild=True)

        # Query 1: Track repair
        results = rag.query("jumper wire repair for broken copper trace IPC-7721", top_k=3)
        self.assertGreater(len(results), 0)
        top_match = results[0]
        self.assertIn("similarity_score", top_match)
        self.assertIn("text", top_match)
        self.assertIn("source", top_match)
        self.assertGreater(top_match["similarity_score"], 0.0)

        # Query 2: Category filtering
        results_filtered = rag.query("regulator capacitor", top_k=3, category="datasheets")
        for r in results_filtered:
            self.assertEqual(r["category"], "datasheets")


if __name__ == "__main__":
    unittest.main()
