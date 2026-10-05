"""
Unit tests for DiagnosticAgent, schema enforcement, RAG formulation, and mocked LLM reasoning.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from src.agents.diagnostic_agent import DiagnosticAgent, diagnose_fault


class TestDiagnosticAgent(unittest.TestCase):
    """Test suite for DiagnosticAgent with mocked Gemini responses and offline heuristics."""

    def setUp(self):
        # Create agent with mock api key
        self.agent = DiagnosticAgent(api_key="mock_test_key_12345")

    def test_query_formulation(self):
        """Test RAG query formulation for various fault types."""
        # Case 1: Open track
        fault_open = {
            "fault_type": "open_track",
            "confidence": 0.92,
            "detector_evidence": "Discontinuity in copper trace",
        }
        queries_open = self.agent._formulate_rag_queries(fault_open)
        self.assertTrue(any("broken" in q.lower() or "jumper" in q.lower() for q in queries_open))

        # Case 2: Missing component
        fault_missing = {
            "fault_type": "missing_component",
            "component_id": "C14",
            "confidence": 0.88,
        }
        queries_missing = self.agent._formulate_rag_queries(fault_missing)
        self.assertTrue(any("missing" in q.lower() or "rework" in q.lower() for q in queries_missing))

    def test_offline_report_generation(self):
        """Test offline grounded report output schema and contents."""
        fault_data = {
            "fault_type": "open_track",
            "confidence": 0.95,
            "location": "x=120, y=340, w=40, h=10",
            "detector_evidence": "Broken trace between C14 and U1 pin 3",
            "severity": "CRITICAL",
        }
        report = self.agent._build_offline_report(fault_data, [], reason="Unit test offline mode")

        # Verify exact detector values
        self.assertEqual(report["detected_fault"], "open_track")
        self.assertEqual(report["confidence"], 0.95)
        self.assertIn("Broken trace", report["evidence"])
        self.assertIn("Location: x=120, y=340", report["evidence"])

        # Verify schema elements
        self.assertIsInstance(report["probable_causes"], list)
        self.assertGreater(len(report["probable_causes"]), 0)
        self.assertIn("IPC-7721", report["recommended_action"])
        self.assertIn("limitations", report)
        self.assertIsInstance(report["supporting_sources"], list)

    def test_parse_llm_response_strict_grounding(self):
        """Test parsing valid LLM JSON and ensuring detector values are preserved."""
        fault_data = {
            "fault_type": "missing_component",
            "confidence": 0.89,
            "detector_evidence": "Expected C14 capacitor not found on PCB pads",
        }
        mock_llm_json = json.dumps(
            {
                "detected_fault": "hallucinated_fault",  # Attempt to change fault type
                "confidence": 0.50,  # Attempt to change confidence
                "evidence": "hallucinated evidence",
                "probable_causes": ["Vacuum pick failure", "Reflow wash-off"],
                "impact": "Loss of voltage regulation stability and ripple filtering.",
                "recommended_action": "Clean pads with solder wick, apply tacky flux, place 10uF tantalum capacitor, and reflow per IPC-A-610.",
                "supporting_sources": [
                    {
                        "title": "Understanding LDO Voltage Regulators",
                        "source": "ti_slva079_ldo_basics.pdf",
                        "category": "component_reference",
                    }
                ],
                "limitations": "Visual inspection cannot test component capacitance or ESR.",
            }
        )

        report = self.agent._parse_llm_response(mock_llm_json, fault_data, [])

        # STRICT GROUNDING: Model MUST NOT override detector values
        self.assertEqual(report["detected_fault"], "missing_component")
        self.assertEqual(report["confidence"], 0.89)
        self.assertEqual(report["evidence"], "Expected C14 capacitor not found on PCB pads")

        # LLM reasoning fields
        self.assertEqual(len(report["probable_causes"]), 2)
        self.assertIn("Loss of voltage regulation", report["impact"])
        self.assertIn("IPC-A-610", report["recommended_action"])
        self.assertEqual(len(report["supporting_sources"]), 1)
        self.assertEqual(report["supporting_sources"][0]["source"], "ti_slva079_ldo_basics.pdf")

    @patch("google.genai.Client")
    def test_diagnose_with_mocked_gemini_client(self, mock_genai_client_class):
        """Test diagnose method with mocked Gemini API response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "detected_fault": "open_track",
                "confidence": 0.93,
                "evidence": "Copper track fracture near test point TP3",
                "probable_causes": ["Mechanical stress from board de-paneling", "Thermal overstress"],
                "impact": "Complete signal loss to sensor interrupt line.",
                "recommended_action": "Repair track with IPC-7721 jumper wire procedure 4.2.3.",
                "supporting_sources": [
                    {
                        "title": "IPC-7721 Conductor Repair",
                        "source": "ipc_7721_procedure_4_2_3_jumper_wire_reference.md",
                        "category": "repair_guides",
                    }
                ],
                "limitations": "X-ray required to verify buried trace continuity.",
            }
        )
        mock_client.models.generate_content.return_value = mock_response

        agent = DiagnosticAgent(api_key="test_key")
        agent.client = mock_client

        fault_input = {
            "fault_type": "open_track",
            "confidence": 0.93,
            "location": "x=50, y=100",
            "detector_evidence": "Copper track fracture near test point TP3",
            "severity": "HIGH",
        }

        report = agent.diagnose(fault_input)

        self.assertEqual(report["detected_fault"], "open_track")
        self.assertEqual(report["confidence"], 0.93)
        self.assertIn("IPC-7721", report["recommended_action"])
        self.assertEqual(len(report["supporting_sources"]), 1)
        self.assertTrue(mock_client.models.generate_content.called)


if __name__ == "__main__":
    unittest.main()
