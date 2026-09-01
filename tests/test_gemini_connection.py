import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai


def test_connection():
    # 1. Load .env file from project root
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    if not env_path.is_file():
        print(f"Error: .env file not found at {env_path}")
        return False, None, None, "File .env not found"

    load_dotenv(dotenv_path=env_path)

    # 2. Verify GEMINI_API_KEY exists without displaying its value
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        print("Error: GEMINI_API_KEY is missing or empty in .env")
        return False, None, None, "GEMINI_API_KEY is missing or empty"

    # 3. Initialize Google GenAI client
    model_name = "gemini-3.6-flash"
    prompt = "Reply with exactly: Gemini API connection successful."

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )

        response_text = response.text.strip() if response.text else ""
        print(response_text)
        return True, model_name, response_text, None

    except Exception as e:
        err_msg = str(e)
        # Redact the actual API key from any error message
        if api_key in err_msg:
            err_msg = err_msg.replace(api_key, "[REDACTED_KEY]")
        print(f"Error during API call: {err_msg}")
        return False, model_name, None, err_msg


if __name__ == "__main__":
    test_connection()
