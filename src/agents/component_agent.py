"""
PCB Component Identification Agent using Google Gemini Vision.
Identifies electronic components on PCB photographs and returns structured data.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Supported component categories
VALID_COMPONENT_TYPES = [
    "resistor",
    "capacitor",
    "diode",
    "LED",
    "transistor",
    "IC",
    "inductor",
    "connector",
    "crystal/oscillator",
    "fuse",
    "potentiometer",
    "switch",
    "other",
]

DEFAULT_MODEL = "gemini-3.6-flash"

SYSTEM_PROMPT = """You are an expert electronics and PCB visual inspection assistant.
Analyze the provided photograph of a Printed Circuit Board (PCB) and identify all clearly visible electronic components.

Allowed component categories:
- resistor
- capacitor
- diode
- LED
- transistor
- IC
- inductor
- connector
- crystal/oscillator
- fuse
- potentiometer
- switch
- other

Guidelines:
1. Only identify components that are clearly recognizable in the image. Do NOT hallucinate components.
2. If a component type is ambiguous or unfamiliar, classify it as "other" or omit it.
3. For "reference", only provide the silkscreen reference designator (e.g., R1, C4, U2, D1, Q3, J5) if it is clearly visible and legible next to the component. Otherwise, set "reference" to null.
4. Assign a realistic "confidence" score between 0.0 and 1.0 reflecting visual clarity.
5. Provide a brief visual description (package style such as SMD/Through-hole, color, visible markings).
6. Provide a concise summary of the PCB image in "image_summary".

You MUST return a valid JSON object with exactly this schema:
{
  "image_summary": "short description of the PCB",
  "components": [
    {
      "type": "component type from allowed categories",
      "reference": "reference designator if visible, otherwise null",
      "confidence": 0.95,
      "description": "short visual description"
    }
  ]
}
"""


class ComponentIdentificationAgent:
    """Agent for identifying electronic components on PCB images using Gemini Vision."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        env_path: Optional[str] = None,
    ):
        """
        Initialize the ComponentIdentificationAgent.

        Args:
            api_key: Optional Gemini API key. If not provided, loaded from .env.
            model_name: Gemini model name to use (default: gemini-3.6-flash).
            env_path: Optional path to .env file.
        """
        self.model_name = model_name

        # Load environment variables if api_key is not explicitly passed
        if not api_key:
            if env_path:
                load_dotenv(dotenv_path=env_path)
            else:
                # Default search from project root
                default_env = Path(__file__).resolve().parent.parent.parent / ".env"
                if default_env.is_file():
                    load_dotenv(dotenv_path=default_env)
                else:
                    load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key or not api_key.strip():
            raise ValueError(
                "GEMINI_API_KEY not found. Please ensure it is configured in the .env file."
            )

        self._api_key = api_key.strip()
        self.client = genai.Client(api_key=self._api_key)

    def _sanitize_error(self, error: Exception) -> str:
        """Remove sensitive API keys from error messages."""
        msg = str(error)
        if self._api_key and self._api_key in msg:
            msg = msg.replace(self._api_key, "[REDACTED_API_KEY]")
        return msg

    def _parse_response(self, raw_text: str) -> Dict[str, Any]:
        """
        Parse and sanitize the JSON response from Gemini.
        Ensures the returned structure strictly matches the expected schema.
        """
        if not raw_text or not raw_text.strip():
            return {
                "image_summary": "Empty response received from model",
                "components": [],
            }

        cleaned_text = raw_text.strip()

        # Strip markdown code fences if present
        if cleaned_text.startswith("```"):
            cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text)
            cleaned_text = re.sub(r"\s*```$", "", cleaned_text)
            cleaned_text = cleaned_text.strip()

        try:
            parsed = json.loads(cleaned_text)
        except json.JSONDecodeError:
            # Attempt to extract JSON substring
            match = re.search(r"\{.*\}", cleaned_text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    return {
                        "image_summary": "Failed to parse model response as JSON",
                        "components": [],
                        "raw_response": cleaned_text,
                    }
            else:
                return {
                    "image_summary": "No valid JSON found in model response",
                    "components": [],
                    "raw_response": cleaned_text,
                }

        # Normalize schema structure
        summary = str(parsed.get("image_summary", "PCB image analyzed"))
        raw_components = parsed.get("components", [])

        normalized_components: List[Dict[str, Any]] = []
        if isinstance(raw_components, list):
            for item in raw_components:
                if not isinstance(item, dict):
                    continue

                comp_type = str(item.get("type", "other")).strip()
                reference = item.get("reference")
                if reference is not None:
                    reference = str(reference).strip()
                    if not reference or reference.lower() in ("null", "none", "n/a", "unknown"):
                        reference = None

                try:
                    confidence = float(item.get("confidence", 0.0))
                    confidence = max(0.0, min(1.0, confidence))
                except (ValueError, TypeError):
                    confidence = 0.0

                description = str(item.get("description", "")).strip()

                normalized_components.append(
                    {
                        "type": comp_type,
                        "reference": reference,
                        "confidence": confidence,
                        "description": description,
                    }
                )

        return {
            "image_summary": summary,
            "components": normalized_components,
        }

    def identify(self, image_path: str) -> Dict[str, Any]:
        """
        Identify components from a PCB image file path.

        Args:
            image_path: Path to the PCB image file.

        Returns:
            Dict matching {"image_summary": str, "components": List[Dict]}
        """
        path = Path(image_path)
        if not path.is_file():
            return {
                "image_summary": f"Image file not found: {image_path}",
                "components": [],
                "error": f"Image file not found: {image_path}",
            }

        # Load and validate image
        try:
            image = Image.open(path)
            image.load()
        except Exception as e:
            return {
                "image_summary": f"Failed to load image file: {path.name}",
                "components": [],
                "error": f"Invalid or unreadable image file: {str(e)}",
            }

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    SYSTEM_PROMPT,
                    "Analyze this PCB photograph and extract all visible electronic components in the required JSON format.",
                    image,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )

            raw_text = response.text if response.text else ""
            return self._parse_response(raw_text)

        except Exception as e:
            sanitized_err = self._sanitize_error(e)
            return {
                "image_summary": "Error occurred during Gemini Vision analysis",
                "components": [],
                "error": sanitized_err,
            }


def identify_components(image_path: str) -> Dict[str, Any]:
    """
    Convenience function to identify electronic components in a PCB photograph.

    Args:
        image_path: Path to the PCB image.

    Returns:
        Structured dictionary with image summary and detected components list.
    """
    try:
        agent = ComponentIdentificationAgent()
        return agent.identify(image_path=image_path)
    except Exception as e:
        # Avoid leaking key if initialization fails
        err_msg = str(e)
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key and api_key in err_msg:
            err_msg = err_msg.replace(api_key, "[REDACTED_API_KEY]")
        return {
            "image_summary": "Agent initialization failed",
            "components": [],
            "error": err_msg,
        }
