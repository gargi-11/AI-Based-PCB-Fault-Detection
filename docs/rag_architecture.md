# RAG Architecture & Diagnostic Reasoning Subsystem

## 1. Overview & System Architecture

The **RAG (Retrieval-Augmented Generation) & Diagnostic Subsystem** serves as the intelligence layer of the AI-Based PCB Fault Detection system. It bridges low-level computer vision detections (e.g. missing components, broken/open copper traces, short circuits, mousebites) with authoritative electronics engineering literature, manufacturer datasheets, application notes, and IPC inspection/rework standards.

```
+-------------------------------------------------------------+
|                     KNOWLEDGE BASE                          |
|  - Texas Instruments Datasheets & Layout Guides             |
|  - Microchip Hardware Design Considerations                 |
|  - IPC-A-610 Inspection Criteria Summaries                  |
|  - IPC-7721 Jumper Wire Rework Standards                    |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                   DOCUMENT INGESTION                        |
|  - PDF text extraction (pypdf)                              |
|  - Markdown/Text loading (UTF-8)                            |
|  - Noise & whitespace normalization (clean_text)            |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                   SEMANTIC CHUNKING                         |
|  - Paragraph & sentence boundary awareness                  |
|  - Chunk size: 700 chars, Overlap: 100 chars                |
|  - Metadata enrichment (Source, Title, Category, DocType)   |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                  VECTOR EMBEDDINGS                          |
|  - Primary: Google GenAI (models/gemini-embedding-001)      |
|  - Offline Fallback: TF-IDF High-Dimensional Dense Vectors  |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|              LOCAL VECTOR STORE (FAISS)                     |
|  - IndexFlatIP with L2 cosine normalization                 |
|  - Local disk persistence (data/vectorstore/)               |
|  - Filterable by technical category                         |
+-------------------------------------------------------------+
                              |
                              |  (Query on detected fault)
                              v
+-------------------------------------------------------------+
|                   SEMANTIC RETRIEVER                        |
|  - Multi-query formulation from CV detector findings        |
|  - Top-k similarity ranking                                 |
|  - Deduplication & relevance thresholding                   |
+-------------------------------------------------------------+
                              |
                              |  (Grounded context + CV Findings)
                              v
+-------------------------------------------------------------+
|                GEMINI DIAGNOSTIC AGENT                      |
|  - Model: gemini-3.6-flash (google-genai SDK)               |
|  - Strict grounding: CV Detector is absolute source of truth|
|  - Schema-enforced structured JSON output                   |
|  - Offline heuristic fallback support                       |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|              STRUCTURED DIAGNOSTIC REPORT                   |
|  - detected_fault, confidence, evidence                     |
|  - probable_causes (physical, thermal, manufacturing)       |
|  - impact (functional, power, signal integrity)             |
|  - recommended_action (IPC-7721 / IPC-A-610 procedures)    |
|  - supporting_sources (provenance citations)                |
|  - limitations (optical inspection boundaries)              |
+-------------------------------------------------------------+
```

---

## 2. Ingested Technical Literature Catalog

All 11 technical documents currently indexed in the vector store:

| # | Filename | Category | Publisher | Size | Diagnostic Role |
|---|---|---|---|---|---|
| 1 | `ti_lm1117_datasheet.pdf` | `datasheets` | Texas Instruments | 2.79 MB | ESR and minimum output capacitance (10 µF) stability requirements to avoid LDO control loop oscillation. |
| 2 | `ti_ne555_datasheet.pdf` | `datasheets` | Texas Instruments | 2.25 MB | Timing networks, threshold/trigger biasing, and bypass decoupling capacitor specifications. |
| 3 | `ti_lm7805_datasheet.pdf` | `datasheets` | Texas Instruments | 2.58 MB | 3-terminal regulator bypass capacitance rules (0.33 µF in, 0.1 µF out) preventing parasitic oscillation. |
| 4 | `ti_snva558_thermal_pcb_design.pdf` | `pcb_faults` | Texas Instruments | 188 KB | Thermal overstress, trace current-carrying limits, and fusing/burnout mechanisms. |
| 5 | `ti_slva951_power_layout_faults.pdf` | `pcb_faults` | Texas Instruments | 164 KB | Ground loop impedance discontinuities, trace inductance, and signal integrity degradation. |
| 6 | `ti_snva021_pcb_layout_guidelines.pdf` | `pcb_faults` | Texas Instruments | 84 KB | High dl/dt loop routing faults and noise injection from disconnected return paths. |
| 7 | `ti_snoa405_smt_rework_guidelines.pdf` | `repair_guides` | Texas Instruments | 594 KB | SMT IC package desoldering, pad re-tinning, solder paste replenishment, and thermal profiles. |
| 8 | `ipc_7721_procedure_4_2_3_jumper_wire_reference.md` | `repair_guides` | IPC Reference | 1.8 KB | Official IPC Procedure 4.2.3 for jumper wire conductor repair (30 AWG Kynar, 1.5mm mask scraping, UV staking). |
| 9 | `microchip_an2519_hardware_design.pdf` | `component_reference` | Microchip Technology | 1.28 MB | Microcontroller decoupling, VCC/GND routing, RESET line pull-up requirements, oscillator shielding. |
| 10 | `ti_slva079_ldo_basics.pdf` | `component_reference` | Texas Instruments | 202 KB | LDO equivalent series resistance (ESR) boundaries and ripple rejection impacts. |
| 11 | `ipc_a_610_component_acceptance_reference.md` | `component_reference` | IPC Reference | 1.2 KB | IPC-A-610 Class 1/2/3 criteria for missing components, solder fillets, and conductor necking. |

---

## 3. How to Ingest & Rebuild the Vector Store

### Running the Ingestion CLI
To scan the `knowledge_base/` folder, chunk all documents, generate embeddings, and persist the index to `data/vectorstore/`:

```powershell
.\venv\Scripts\python.exe src/processing/ingest_knowledge_base.py
```

### Programmatic Ingestion
```python
from src.services.rag_service import RAGService

rag = RAGService()
result = rag.ingest_knowledge_base(force_rebuild=True)
print(f"Indexed {result['indexed_documents']} documents into {result['total_chunks']} chunks.")
```

---

## 4. How to Add New Documents

To add new datasheets, application notes, or IPC guides:
1. Place the `.pdf`, `.md`, or `.txt` file into the appropriate subdirectory under `knowledge_base/`:
   - `knowledge_base/datasheets/`
   - `knowledge_base/pcb_faults/`
   - `knowledge_base/repair_guides/`
   - `knowledge_base/component_reference/`
2. *(Optional)* Add a metadata entry to `DOCUMENT_CATALOG` in `src/services/rag_service.py` to supply formal titles, publishers, and source URLs. If omitted, metadata is inferred automatically from the file and directory names.
3. Run the ingestion command:
   ```powershell
   .\venv\Scripts\python.exe src/processing/ingest_knowledge_base.py
   ```

---

## 5. How Semantic Retrieval Works

1. **Query Formulation**: The `DiagnosticAgent` inspects the detected fault attributes (e.g. `fault_type="open_track"`, `location="x:120, y:340"`, `evidence="..."`) and generates multiple targeted domain queries.
2. **Embedding**: The query text is converted into an L2-normalized vector.
3. **Similarity Search**: FAISS calculates inner-product (cosine similarity) distances against all chunk vectors in the database.
4. **Ranking & Filtering**: Chunks are ranked by similarity score, deduplicated, and optionally filtered by category.
5. **Score Metric**: Returns similarity scores between `0.0` (unrelated) and `1.0` (exact semantic match).

---

## 6. How Gemini is Used & Grounding Principles

### Model Configuration
- **Model**: `gemini-3.6-flash` via official `google-genai` Python SDK.
- **Output Format**: Enforced JSON schema (`response_mime_type="application/json"`).
- **Temperature**: `0.1` (low temperature to eliminate creative drift and enforce technical precision).

### Grounding Rules
- **Detector Authority**: The Computer Vision detector remains the single source of truth. Gemini is strictly prohibited from overriding or altering `detected_fault`, `confidence`, or `evidence`.
- **Engineering Justification**: Gemini provides the *why* (physical and thermal root causes), the *impact* (circuit and signal integrity effects), and the *how* (step-by-step IPC-compliant repair instructions).
- **Offline Fallback**: If network issues, rate limits, or API outages occur, the agent automatically switches to an offline heuristic engine that outputs a fully compliant diagnostic report using local RAG context.

---

## 7. Diagnostic Report Schema

Every diagnostic analysis produces a standardized dictionary matching this format:

```json
{
  "detected_fault": "open_track",
  "confidence": 0.94,
  "evidence": "Broken copper trace detected between regulator output pin and filter capacitor C14 [Location: x=142, y=280, w=35, h=8]",
  "probable_causes": [
    "Local thermal overstress or transient overcurrent exceeding conductor fusing current (AN-2020).",
    "Mechanical stress, PCB flexure, or sharp handling resulting in trace fracture.",
    "Chemical over-etching or localized contamination during PCB fabrication."
  ],
  "impact": "Open-circuit discontinuity preventing power distribution or critical signal propagation to downstream components.",
  "recommended_action": "Execute IPC-7721 Procedure 4.2.3: Scrape solder mask 1.5mm on both sides of break, pre-tin exposed conductor, install 30 AWG insulated Kynar jumper wire with smooth concave fillets, and secure with UV-curable polymer staking.",
  "supporting_sources": [
    {
      "title": "IPC-7721 Procedure 4.2.3 Conductor Repair Reference",
      "source": "ipc_7721_procedure_4_2_3_jumper_wire_reference.md",
      "category": "repair_guides"
    }
  ],
  "limitations": "Optical inspection cannot detect micro-cracks inside inner layers of multilayer PCBs or determine subsurface copper adhesion."
}
```

---

## 8. Limitations & Boundary Conditions

1. **Optical Surface Boundary**: Computer vision and optical inspection operate on the top/bottom surface copper layers and components. Internal traces and power planes on multi-layer PCBs require X-ray inspection or time-domain reflectometry (TDR).
2. **Component Value Verification**: Optical inspection identifies missing or physically displaced components, but cannot measure internal electrical values (such as capacitor dielectric aging or resistor tolerance shift) without in-circuit testing (ICT).
3. **High-Density Ball Grid Arrays (BGA)**: Hidden solder joints under BGA packages cannot be inspected optically.
