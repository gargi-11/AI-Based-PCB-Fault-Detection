"""
RAG (Retrieval-Augmented Generation) Service for PCB Knowledge Base.
Handles document loading, text extraction, semantic chunking, metadata enrichment,
vector embeddings (Google GenAI models/gemini-embedding-001 & local offline vectorizer),
and FAISS similarity search.
"""

import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Known Document Catalog mapping filename to verified technical metadata
DOCUMENT_CATALOG: Dict[str, Dict[str, str]] = {
    "ti_lm1117_datasheet.pdf": {
        "title": "LM1117 800-mA Low-Dropout Linear Regulator Datasheet",
        "category": "datasheets",
        "publisher": "Texas Instruments Incorporated",
        "doc_type": "Component Datasheet (SNOS412N)",
        "source_url": "https://www.ti.com/lit/ds/symlink/lm1117.pdf",
    },
    "ti_ne555_datasheet.pdf": {
        "title": "NE555 Precision Timers Datasheet",
        "category": "datasheets",
        "publisher": "Texas Instruments Incorporated",
        "doc_type": "Component Datasheet (SLFS022I)",
        "source_url": "https://www.ti.com/lit/ds/symlink/ne555.pdf",
    },
    "ti_lm7805_datasheet.pdf": {
        "title": "LM340, LM340A, and LM7805 Series 3-Terminal Positive Regulators",
        "category": "datasheets",
        "publisher": "Texas Instruments Incorporated",
        "doc_type": "Component Datasheet (SNOSBD0J)",
        "source_url": "https://www.ti.com/lit/ds/symlink/lm340.pdf",
    },
    "ti_snva558_thermal_pcb_design.pdf": {
        "title": "AN-2020 Thermal Design By Insight, Not Hindsight",
        "category": "pcb_faults",
        "publisher": "Texas Instruments Incorporated",
        "doc_type": "Application Report (SNVA558B)",
        "source_url": "https://www.ti.com/lit/an/snva558b/snva558b.pdf",
    },
    "ti_slva951_power_layout_faults.pdf": {
        "title": "Layout Guidelines for Sound and Vibration Power Supplies",
        "category": "pcb_faults",
        "publisher": "Texas Instruments Incorporated",
        "doc_type": "Application Report (SLVA951)",
        "source_url": "https://www.ti.com/lit/an/slva951/slva951.pdf",
    },
    "ti_snva021_pcb_layout_guidelines.pdf": {
        "title": "AN-1149 Layout Guidelines for Switching Regulators",
        "category": "pcb_faults",
        "publisher": "Texas Instruments Incorporated",
        "doc_type": "Application Report (SNVA021C)",
        "source_url": "https://www.ti.com/lit/an/snva021c/snva021c.pdf",
    },
    "ti_snoa405_smt_rework_guidelines.pdf": {
        "title": "AN-1187 Leadless Package User's Guide and Rework Guidelines",
        "category": "repair_guides",
        "publisher": "Texas Instruments Incorporated",
        "doc_type": "Application Report (SNOA405A)",
        "source_url": "https://www.ti.com/lit/an/snoa405a/snoa405a.pdf",
    },
    "ipc_7721_procedure_4_2_3_jumper_wire_reference.md": {
        "title": "IPC-7721 Procedure 4.2.3: Jumper Wire Modification & Conductor Repair Reference",
        "category": "repair_guides",
        "publisher": "IPC Engineering Reference",
        "doc_type": "Engineering Standard Procedure Summary",
        "source_url": "Standard IPC-7711/7721 Rework Specification",
    },
    "microchip_an2519_hardware_design.pdf": {
        "title": "AN2519: AVR Microcontroller Hardware Design Considerations",
        "category": "component_reference",
        "publisher": "Microchip Technology Incorporated",
        "doc_type": "Application Note (DS00002519B)",
        "source_url": "https://ww1.microchip.com/downloads/en/Appnotes/AN2519-AVR-Microcontroller-Hardware-Design-Considerations-00002519B.pdf",
    },
    "ti_slva079_ldo_basics.pdf": {
        "title": "Understanding the Terms and Definitions of LDO Voltage Regulators",
        "category": "component_reference",
        "publisher": "Texas Instruments Incorporated",
        "doc_type": "Application Report (SLVA079)",
        "source_url": "https://www.ti.com/lit/an/slva079/slva079.pdf",
    },
    "ipc_a_610_component_acceptance_reference.md": {
        "title": "IPC-A-610: Electronic Assembly Inspection & Acceptance Criteria Reference",
        "category": "component_reference",
        "publisher": "IPC Engineering Reference",
        "doc_type": "Engineering Standard Summary",
        "source_url": "Standard IPC-A-610 Inspection Criteria",
    },
}

DEFAULT_VECTORSTORE_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "data" / "vectorstore"
)
DEFAULT_KB_PATH = str(Path(__file__).resolve().parent.parent.parent / "knowledge_base")
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"


@dataclass
class DocumentChunk:
    """Represents a text chunk with complete provenance metadata."""

    chunk_id: str
    text: str
    source: str
    title: str
    category: str
    publisher: str
    doc_type: str
    source_url: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def clean_text(raw_text: str) -> str:
    """
    Clean extracted text by removing control characters, excess whitespace,
    and normalizing line breaks while preserving punctuation and numbers.
    """
    if not raw_text:
        return ""

    # Replace null bytes and non-printable control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", raw_text)

    # Normalize multiple consecutive spaces/tabs to a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize multiple line breaks to max 2
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Strip leading and trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    cleaned = "\n".join(lines).strip()
    return cleaned


def extract_text_from_file(file_path: Path) -> str:
    """
    Extract raw text from PDF or Markdown/text file.

    Args:
        file_path: Path to the target file.

    Returns:
        Extracted string content.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        extracted_pages: List[str] = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_pages.append(page_text.strip())
        return "\n\n".join(extracted_pages)

    elif suffix in [".md", ".txt", ".rst"]:
        return file_path.read_text(encoding="utf-8", errors="ignore")

    else:
        raise ValueError(f"Unsupported file format: {suffix} for {file_path}")


def chunk_text(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> List[str]:
    """
    Split text into overlapping semantic chunks prioritizing paragraph, sentence,
    and whitespace boundaries.

    Args:
        text: Input text string.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Target character overlap between consecutive chunks.

    Returns:
        List of chunk strings.
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > chunk_size:
            sentences = re.split(r"(?<=[.?!])\s+", para)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue

                if len(current_chunk) + len(sent) + 1 <= chunk_size:
                    current_chunk = f"{current_chunk} {sent}".strip()
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    if len(sent) > chunk_size:
                        start = 0
                        while start < len(sent):
                            end = start + chunk_size
                            chunks.append(sent[start:end])
                            start = end - chunk_overlap
                        current_chunk = ""
                    else:
                        current_chunk = sent
        else:
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk = f"{current_chunk}\n\n{para}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    if chunk_overlap > 0 and len(chunks) > 1:
        refined_chunks: List[str] = []
        for i, chunk in enumerate(chunks):
            if i > 0 and chunk_overlap > 0:
                prev_overlap = chunks[i - 1][-chunk_overlap:]
                combined = f"... {prev_overlap} {chunk}".strip()
                refined_chunks.append(combined)
            else:
                refined_chunks.append(chunk)
        return refined_chunks

    return chunks


class RAGService:
    """
    Manages knowledge base document ingestion, local vector database persistence (FAISS),
    and semantic context retrieval for PCB fault diagnosis.
    """

    def __init__(
        self,
        vectorstore_dir: Optional[str] = None,
        kb_path: Optional[str] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        api_key: Optional[str] = None,
        use_offline_fallback: bool = False,
    ):
        """
        Initialize the RAGService.

        Args:
            vectorstore_dir: Directory path for FAISS and metadata storage.
            kb_path: Directory path for knowledge_base files.
            embedding_model: Gemini embedding model name (default: models/gemini-embedding-001).
            api_key: Optional Gemini API key.
            use_offline_fallback: If True, forces local TF-IDF vectorizer (useful for tests).
        """
        self.vectorstore_dir = Path(vectorstore_dir or DEFAULT_VECTORSTORE_PATH)
        self.kb_path = Path(kb_path or DEFAULT_KB_PATH)
        self.embedding_model = embedding_model
        self.use_offline_fallback = use_offline_fallback

        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.vectorstore_dir / "faiss_index.bin"
        self.metadata_file = self.vectorstore_dir / "chunks_metadata.json"
        self.config_file = self.vectorstore_dir / "vectorstore_config.json"

        # Load API key
        if not api_key:
            default_env = Path(__file__).resolve().parent.parent.parent / ".env"
            if default_env.is_file():
                load_dotenv(dotenv_path=default_env)
            else:
                load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")

        self.api_key = api_key.strip() if api_key else None
        self.genai_client = None
        if self.api_key and not self.use_offline_fallback:
            try:
                self.genai_client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize GenAI client: {e}")
                self.genai_client = None

        self.chunks: List[DocumentChunk] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self.offline_vectorizer: Optional[TfidfVectorizer] = None
        self.active_embedding_type: str = "offline_tfidf"

        # Try to load existing index if present
        self.load_index()

    def load_and_chunk_documents(self, kb_dir: Optional[str] = None) -> List[DocumentChunk]:
        """
        Scan knowledge base directory, extract text, clean, and chunk all documents.

        Args:
            kb_dir: Optional directory override for knowledge_base.

        Returns:
            List of DocumentChunk instances.
        """
        target_dir = Path(kb_dir) if kb_dir else self.kb_path
        if not target_dir.is_dir():
            raise FileNotFoundError(f"Knowledge base directory not found: {target_dir}")

        supported_exts = {".pdf", ".md", ".txt"}
        files = [
            f for f in target_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in supported_exts and not f.name.startswith(".")
        ]

        chunks: List[DocumentChunk] = []

        for f in sorted(files):
            filename = f.name
            raw_text = extract_text_from_file(f)
            cleaned = clean_text(raw_text)
            if not cleaned:
                continue

            rel_path = f.relative_to(target_dir)
            folder_category = rel_path.parts[0] if len(rel_path.parts) > 1 else "general"

            catalog_entry = DOCUMENT_CATALOG.get(filename, {})
            title = catalog_entry.get("title", f.stem.replace("_", " ").title())
            category = catalog_entry.get("category", folder_category)
            publisher = catalog_entry.get("publisher", "Technical Engineering Reference")
            doc_type = catalog_entry.get("doc_type", f.suffix.upper()[1:])
            source_url = catalog_entry.get("source_url", f"local://{filename}")

            text_chunks = chunk_text(cleaned, chunk_size=700, chunk_overlap=100)

            for idx, c_text in enumerate(text_chunks):
                chunk_id = f"{filename}_{idx}"
                metadata = {
                    "source": filename,
                    "title": title,
                    "category": category,
                    "publisher": publisher,
                    "doc_type": doc_type,
                    "source_url": source_url,
                    "chunk_index": idx,
                }
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=c_text,
                        source=filename,
                        title=title,
                        category=category,
                        publisher=publisher,
                        doc_type=doc_type,
                        source_url=source_url,
                        chunk_index=idx,
                        metadata=metadata,
                    )
                )

        return chunks

    def _generate_embeddings(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Generate L2-normalized embeddings for a list of text strings.
        Uses Gemini GenAI Embeddings if available, else local TF-IDF dense vectors.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        if self.genai_client is not None and not self.use_offline_fallback:
            try:
                embeddings_list: List[List[float]] = []
                batch_size = 20
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    res = self.genai_client.models.embed_content(
                        model=self.embedding_model,
                        contents=batch,
                    )
                    for emb in res.embeddings:
                        embeddings_list.append(emb.values)

                emb_matrix = np.array(embeddings_list, dtype=np.float32)
                faiss.normalize_L2(emb_matrix)
                self.active_embedding_type = f"gemini:{self.embedding_model}"
                return emb_matrix
            except Exception as e:
                logger.warning(f"Gemini embedding call failed: {e}. Using offline vectorizer.")

        # Offline Fallback using TF-IDF Dense Vectors
        self.active_embedding_type = "offline_tfidf"
        if self.offline_vectorizer is None or not is_query:
            if not is_query or self.offline_vectorizer is None:
                self.offline_vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=4096,
                    stop_words="english",
                )
                emb_matrix = self.offline_vectorizer.fit_transform(texts).toarray().astype(np.float32)
            else:
                emb_matrix = self.offline_vectorizer.transform(texts).toarray().astype(np.float32)
        else:
            emb_matrix = self.offline_vectorizer.transform(texts).toarray().astype(np.float32)

        faiss.normalize_L2(emb_matrix)
        return emb_matrix

    def ingest_knowledge_base(
        self,
        kb_dir: Optional[str] = None,
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingest all documents into the local FAISS vector database.

        Args:
            kb_dir: Optional path to knowledge_base.
            force_rebuild: If True, deletes existing collection and rebuilds.

        Returns:
            Dictionary summary with count of documents and chunks indexed.
        """
        if self.index is not None and len(self.chunks) > 0 and not force_rebuild:
            return {
                "status": "already_indexed",
                "indexed_documents": len(set(c.source for c in self.chunks)),
                "total_chunks": len(self.chunks),
                "embedding_type": self.active_embedding_type,
                "message": "Vector store is already populated. Use force_rebuild=True to overwrite.",
            }

        chunks = self.load_and_chunk_documents(kb_dir)
        if not chunks:
            return {
                "status": "no_documents",
                "total_chunks": 0,
                "message": "No valid documents found in knowledge base.",
            }

        texts = [c.text for c in chunks]
        emb_matrix = self._generate_embeddings(texts, is_query=False)

        dim = emb_matrix.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb_matrix)
        self.chunks = chunks

        # Persist index and metadata to disk
        self.save_index()

        unique_docs = len(set(c.source for c in chunks))
        return {
            "status": "success",
            "indexed_documents": unique_docs,
            "total_chunks": len(chunks),
            "vector_dimension": dim,
            "embedding_type": self.active_embedding_type,
            "vectorstore_dir": str(self.vectorstore_dir),
        }

    def save_index(self) -> None:
        """Save FAISS binary index, chunk metadata JSON, and config to disk."""
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_file))

        if self.chunks:
            meta_data = [c.to_dict() for c in self.chunks]
            self.metadata_file.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")

        config_data = {
            "embedding_type": self.active_embedding_type,
            "dimension": self.index.d if self.index is not None else 0,
            "total_chunks": len(self.chunks),
        }
        self.config_file.write_text(json.dumps(config_data, indent=2), encoding="utf-8")

    def load_index(self) -> bool:
        """Load FAISS binary index and chunk metadata from disk if present."""
        if self.index_file.is_file() and self.metadata_file.is_file():
            try:
                self.index = faiss.read_index(str(self.index_file))
                raw_meta = json.loads(self.metadata_file.read_text(encoding="utf-8"))
                self.chunks = [
                    DocumentChunk(
                        chunk_id=item["chunk_id"],
                        text=item["text"],
                        source=item["source"],
                        title=item.get("title", ""),
                        category=item.get("category", ""),
                        publisher=item.get("publisher", ""),
                        doc_type=item.get("doc_type", ""),
                        source_url=item.get("source_url", ""),
                        chunk_index=item.get("chunk_index", 0),
                        metadata=item.get("metadata", {}),
                    )
                    for item in raw_meta
                ]
                if self.config_file.is_file():
                    cfg = json.loads(self.config_file.read_text(encoding="utf-8"))
                    self.active_embedding_type = cfg.get("embedding_type", "unknown")

                # If offline vectorizer was used, fit it on existing chunks for fast queries
                if "offline" in self.active_embedding_type or self.genai_client is None:
                    self.offline_vectorizer = TfidfVectorizer(
                        ngram_range=(1, 2),
                        max_features=4096,
                        stop_words="english",
                    )
                    self.offline_vectorizer.fit([c.text for c in self.chunks])

                return True
            except Exception as e:
                logger.warning(f"Failed to load vectorstore: {e}")
                self.index = None
                self.chunks = []
                return False
        return False

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query vector database and retrieve the most relevant knowledge base chunks.

        Args:
            query_text: The technical inquiry string.
            top_k: Maximum number of chunks to return.
            category: Optional filter by document category.

        Returns:
            List of matching chunks with similarity score, text, and metadata.
        """
        if not query_text or not query_text.strip() or self.index is None or not self.chunks:
            return []

        # Generate query embedding
        query_emb = self._generate_embeddings([query_text.strip()], is_query=True)
        if query_emb.shape[1] != self.index.d:
            # Rebuild vectorizer on current chunks if dimension mismatch
            if self.offline_vectorizer is not None:
                self.offline_vectorizer.fit([c.text for c in self.chunks])
                query_emb = self._generate_embeddings([query_text.strip()], is_query=True)
            if query_emb.shape[1] != self.index.d:
                return []

        fetch_k = min(len(self.chunks), top_k * 4 if category else top_k)
        scores, indices = self.index.search(query_emb, fetch_k)

        formatted_results: List[Dict[str, Any]] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]
            if category and chunk.category.lower() != category.lower():
                continue

            sim_score = max(0.0, min(1.0, float(score)))

            formatted_results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "title": chunk.title,
                    "category": chunk.category,
                    "publisher": chunk.publisher,
                    "doc_type": chunk.doc_type,
                    "source_url": chunk.source_url,
                    "similarity_score": round(sim_score, 4),
                    "metadata": chunk.metadata,
                }
            )

            if len(formatted_results) >= top_k:
                break

        return formatted_results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return diagnostic statistics of the current vector database."""
        total = self.index.ntotal if self.index is not None else 0
        return {
            "vectorstore_dir": str(self.vectorstore_dir),
            "total_chunks": total,
            "unique_documents": len(set(c.source for c in self.chunks)),
            "dimension": self.index.d if self.index is not None else 0,
            "embedding_type": self.active_embedding_type,
        }
