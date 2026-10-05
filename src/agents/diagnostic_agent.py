"""
PCB Diagnostic Reasoning Agent using RAG and Google Gemini.
Combines authoritative Computer Vision detector findings with retrieved engineering
literature from the knowledge base to produce grounded diagnostic reports.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.services.rag_service import RAGService

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.6-flash"

SYSTEM_PROMPT = """You are a senior electronics reliability engineer, PCB failure analyst, and IPC-certified rework specialist.

Your task is to generate a rigorous, technically grounded Diagnostic Report for a PCB defect detected by an authoritative Computer Vision system.

MANDATORY RULES:
1. THE COMPUTER VISION DETECTOR FINDINGS ARE THE ABSOLUTE SOURCE OF TRUTH. You must NEVER override, invent, deny, or alter the detected fault, confidence, or evidence provided by the detector.
2. Ground all explanations, probable causes, and recommended repair actions in the PROVIDED TECHNICAL CONTEXT from official manufacturer datasheets, application notes, and IPC standards (IPC-A-610, IPC-7721).
3. Do NOT hallucinate standard specifications or repair numbers. If a detail is not in the context, use standard electronics engineering principles and clearly state assumptions.
4. Highlight inspection limitations (e.g., optical inspection cannot verify internal traces or BGA solder balls without X-ray or micro-sectioning).

You MUST return a valid JSON object matching EXACTLY this schema:
{
  "detected_fault": "<exact fault type from detector>",
  "confidence": <exact float confidence from detector>,
  "evidence": "<summary of visual/detector evidence>",
  "probable_causes": [
    "<probable physical, manufacturing, thermal, or mechanical root cause 1>",
    "<probable root cause 2>"
  ],
  "impact": "<circuit-level functional impact on power/signal integrity or component reliability>",
  "recommended_action": "<step-by-step actionable repair procedure adhering to IPC-7721/IPC-A-610>",
  "supporting_sources": [
    {
      "title": "<document title>",
      "source": "<filename>",
      "category": "<category>"
    }
  ],
  "limitations": "<limitations of optical visual inspection for this fault>"
}
"""


class DiagnosticAgent:
    """
    Synthesizes authoritative CV detector findings with knowledge-base RAG retrieval
    and Gemini LLM reasoning to produce structured PCB failure diagnostic reports.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        rag_service: Optional[RAGService] = None,
        env_path: Optional[str] = None,
    ):
        """
        Initialize the DiagnosticAgent.

        Args:
            api_key: Optional Gemini API key.
            model_name: Gemini model name (default: gemini-3.6-flash).
            rag_service: Optional instance of RAGService. If None, initializes default.
            env_path: Optional path to .env file.
        """
        self.model_name = model_name

        # Load environment variables
        if not api_key:
            if env_path:
                load_dotenv(dotenv_path=env_path)
            else:
                default_env = Path(__file__).resolve().parent.parent.parent / ".env"
                if default_env.is_file():
                    load_dotenv(dotenv_path=default_env)
                else:
                    load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")

        self.api_key = api_key.strip() if api_key else None
        self.client = None
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize GenAI client: {e}")
                self.client = None

        self.rag = rag_service or RAGService()

    def _sanitize_error(self, error: Exception) -> str:
        """Strip sensitive credentials from error messages."""
        msg = str(error)
        if self.api_key and self.api_key in msg:
            msg = msg.replace(self.api_key, "[REDACTED_API_KEY]")
        return msg

    def _formulate_rag_queries(self, fault_data: Dict[str, Any]) -> List[str]:
        """Generate targeted search queries for the RAG retriever based on fault characteristics."""
        fault_type = str(fault_data.get("fault_type", "pcb_defect")).lower()
        evidence = str(fault_data.get("detector_evidence", fault_data.get("evidence", "")))
        component = str(fault_data.get("component_id", fault_data.get("component_type", "")))

        queries: List[str] = []

        if "open" in fault_type or "broken" in fault_type or "track" in fault_type or "discontinuity" in fault_type:
            queries.append("Causes and repair procedure for broken open copper PCB trace IPC-7721 jumper wire")
            queries.append("Thermal overstress and current carrying trace blowout failure mechanisms")
        elif "missing" in fault_type:
            queries.append(f"Missing component {component} acceptance criteria IPC-A-610 and circuit impact")
            queries.append("SMD solder joint rework guidelines and component placement")
        elif "short" in fault_type or "bridge" in fault_type:
            queries.append("Solder bridge short circuit fault causes and desoldering rework")
        elif "mousebite" in fault_type or "spur" in fault_type:
            queries.append("PCB conductor width reduction minimum electrical clearance IPC-A-610")
        else:
            queries.append(f"{fault_type} defect cause and repair in PCB assembly")

        if evidence:
            queries.append(f"{fault_type} {evidence[:100]}")

        return queries

    def _retrieve_context(self, queries: List[str], top_k: int = 4) -> List[Dict[str, Any]]:
        """Retrieve deduplicated top context chunks from knowledge base."""
        seen_chunks = set()
        retrieved: List[Dict[str, Any]] = []

        for q in queries:
            results = self.rag.query(q, top_k=top_k)
            for r in results:
                cid = r.get("chunk_id", "")
                if cid not in seen_chunks:
                    seen_chunks.add(cid)
                    retrieved.append(r)

        # Sort by similarity score descending
        retrieved.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
        return retrieved[:6]

    def _build_offline_report(
        self,
        fault_data: Dict[str, Any],
        retrieved_context: List[Dict[str, Any]],
        reason: str = "Offline heuristic mode",
    ) -> Dict[str, Any]:
        """
        Generate a strictly schema-compliant grounded diagnostic report
        without calling external APIs (used for offline operation or unit tests).
        """
        fault_type = str(fault_data.get("fault_type", "detected_defect"))
        confidence = float(fault_data.get("confidence", 0.85))
        evidence = str(fault_data.get("detector_evidence", fault_data.get("evidence", "Detector identified visual anomaly.")))
        location = str(fault_data.get("location", "N/A"))

        # Ground probable causes and repair actions based on fault type
        if "open" in fault_type.lower() or "track" in fault_type.lower():
            probable_causes = [
                "Local thermal overstress or transient overcurrent exceeding conductor fusing current (AN-2020).",
                "Mechanical stress, PCB flexure, or sharp handling resulting in trace fracture.",
                "Chemical over-etching or localized contamination during PCB fabrication.",
            ]
            impact = "Open-circuit discontinuity preventing power distribution or critical signal propagation to downstream components."
            recommended_action = "Execute IPC-7721 Procedure 4.2.3: Scrape solder mask 1.5mm on both sides of break, pre-tin exposed conductor, install 30 AWG insulated Kynar jumper wire with smooth concave fillets, and secure with UV-curable polymer staking."
            limitations = "Optical inspection cannot detect micro-cracks inside inner layers of multilayer PCBs or determine subsurface copper adhesion."
        elif "missing" in fault_type.lower():
            probable_causes = [
                "Pick-and-place nozzle vacuum drop during automated surface mount assembly.",
                "Solder paste volume deficiency or misprinted aperture resulting in component tombstoning/wash-off.",
                "Component unseated or dislodged during reflow convective cooling or board handling.",
            ]
            impact = "Circuit malfunction resulting from floating input, lack of power supply bypass filtering, or disrupted biasing."
            recommended_action = "Inspect PCB pads for coplanarity and solder wettability per IPC-A-610 Class 2/3. Re-tin pads with flux, place replacement component with correct orientation, and reflow/hand-solder with temperature-controlled iron."
            limitations = "Visual inspection cannot verify internal silicon die integrity or component electrical tolerance without in-circuit test (ICT)."
        else:
            probable_causes = [
                "Manufacturing process variation during photolithography, etching, or solder deposition.",
                "Contamination or foreign conductive debris on PCB surface.",
            ]
            impact = "Potential signal degradation, reduced electrical clearance, or intermittent circuit behavior."
            recommended_action = "Inspect affected area under 10x optical magnification. Clean with isopropyl alcohol (IPA) and verify conductor minimum spacing meets IPC-A-610 criteria."
            limitations = "Visual inspection is limited to top surface topology and cannot verify high-frequency impedance parameters."

        sources = []
        for chunk in retrieved_context[:3]:
            sources.append(
                {
                    "title": chunk.get("title", chunk.get("source", "")),
                    "source": chunk.get("source", ""),
                    "category": chunk.get("category", ""),
                }
            )

        if not sources:
            sources.append(
                {
                    "title": "IPC-7721 Procedure 4.2.3 Conductor Repair Reference",
                    "source": "ipc_7721_procedure_4_2_3_jumper_wire_reference.md",
                    "category": "repair_guides",
                }
            )

        return {
            "detected_fault": fault_type,
            "confidence": confidence,
            "evidence": f"{evidence} [Location: {location}]" if location != "N/A" else evidence,
            "probable_causes": probable_causes,
            "impact": impact,
            "recommended_action": recommended_action,
            "supporting_sources": sources,
            "limitations": limitations,
            "diagnostic_mode": reason,
        }

    def _parse_llm_response(
        self,
        raw_text: str,
        fault_data: Dict[str, Any],
        retrieved_context: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Parse, validate, and enforce strict detector grounding on Gemini response."""
        cleaned_text = raw_text.strip()
        if cleaned_text.startswith("```"):
            cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text)
            cleaned_text = re.sub(r"\s*```$", "", cleaned_text)
            cleaned_text = cleaned_text.strip()

        try:
            parsed = json.loads(cleaned_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned_text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    return self._build_offline_report(
                        fault_data, retrieved_context, reason="Failed to parse LLM JSON"
                    )
            else:
                return self._build_offline_report(
                    fault_data, retrieved_context, reason="No JSON in LLM response"
                )

        # STRICT GROUNDING: Force detector values to prevent any hallucination
        fault_type = str(fault_data.get("fault_type", parsed.get("detected_fault", "")))
        confidence = float(fault_data.get("confidence", parsed.get("confidence", 0.0)))
        evidence = str(fault_data.get("detector_evidence", fault_data.get("evidence", parsed.get("evidence", ""))))

        raw_causes = parsed.get("probable_causes", [])
        probable_causes = [str(c) for c in raw_causes if str(c).strip()] if isinstance(raw_causes, list) else [str(raw_causes)]

        impact = str(parsed.get("impact", "Circuit discontinuity or performance degradation."))
        recommended_action = str(parsed.get("recommended_action", "Inspect and rework per IPC standards."))
        limitations = str(parsed.get("limitations", "Optical visual inspection cannot assess internal PCB layers."))

        raw_sources = parsed.get("supporting_sources", [])
        sources = []
        if isinstance(raw_sources, list) and raw_sources:
            for s in raw_sources:
                if isinstance(s, dict):
                    sources.append(
                        {
                            "title": str(s.get("title", "Technical Reference")),
                            "source": str(s.get("source", "knowledge_base")),
                            "category": str(s.get("category", "general")),
                        }
                    )
        if not sources and retrieved_context:
            for chunk in retrieved_context[:2]:
                sources.append(
                    {
                        "title": chunk.get("title", chunk.get("source", "")),
                        "source": chunk.get("source", ""),
                        "category": chunk.get("category", ""),
                    }
                )

        return {
            "detected_fault": fault_type,
            "confidence": confidence,
            "evidence": evidence,
            "probable_causes": probable_causes,
            "impact": impact,
            "recommended_action": recommended_action,
            "supporting_sources": sources,
            "limitations": limitations,
        }

    def diagnose(self, fault_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a complete diagnostic report for a detected PCB fault.

        Args:
            fault_data: Dictionary from CV detector (e.g. fault_type, confidence, location, detector_evidence).

        Returns:
            Structured diagnostic dictionary.
        """
        if not fault_data or not isinstance(fault_data, dict):
            raise ValueError("Invalid fault_data provided. Expected dictionary.")

        # Step 1: Formulate RAG queries
        queries = self._formulate_rag_queries(fault_data)

        # Step 2: Retrieve relevant technical context from vector database
        retrieved_context = self._retrieve_context(queries, top_k=3)

        # Build context prompt text
        context_blocks = []
        for idx, chunk in enumerate(retrieved_context, 1):
            context_blocks.append(
                f"[Document {idx}]: {chunk['title']} ({chunk['source']} - {chunk['category']})\n{chunk['text']}"
            )
        context_str = "\n\n".join(context_blocks) if context_blocks else "No specific document matched."

        # If client is not configured, generate offline report
        if self.client is None:
            return self._build_offline_report(
                fault_data, retrieved_context, reason="Offline (No Gemini API client)"
            )

        # Step 3: Construct prompt
        user_prompt = f"""AUTHORITATIVE DETECTOR FINDINGS:
Fault Type: {fault_data.get('fault_type', 'unknown')}
Confidence: {fault_data.get('confidence', 'N/A')}
Location: {fault_data.get('location', 'N/A')}
Detector Evidence: {fault_data.get('detector_evidence', fault_data.get('evidence', 'Visual defect detected'))}
Severity: {fault_data.get('severity', 'UNKNOWN')}

RETRIEVED TECHNICAL KNOWLEDGE BASE CONTEXT:
{context_str}

Analyze the detector findings in light of the technical literature.
Generate the required structured JSON diagnostic report adhering strictly to the schema and instructions."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[SYSTEM_PROMPT, user_prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )

            raw_text = response.text if response.text else ""
            return self._parse_llm_response(raw_text, fault_data, retrieved_context)

        except Exception as e:
            logger.warning(f"Gemini diagnostic call error: {self._sanitize_error(e)}. Generating grounded fallback report.")
            return self._build_offline_report(
                fault_data,
                retrieved_context,
                reason=f"Fallback triggered: {self._sanitize_error(e)}",
            )


def diagnose_fault(fault_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to run end-to-end diagnosis on a detected fault.

    Args:
        fault_data: Detector output dictionary.

    Returns:
        Structured diagnostic report dictionary.
    """
    agent = DiagnosticAgent()
    return agent.diagnose(fault_data)
