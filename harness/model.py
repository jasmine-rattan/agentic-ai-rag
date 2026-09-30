"""Ollama API wrapper for gemma3:4b."""

import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:4b"


def call_gemma(prompt: str, system: str = None, stream: bool = False) -> str:
    """Send a prompt to gemma3:4b via Ollama and return the response text."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": stream,
    }
    if system:
        payload["system"] = system

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure Ollama is running:\n"
            "  brew install ollama\n"
            "  ollama serve\n"
            "  ollama pull gemma3:4b"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama request timed out after 120 seconds.")

    if stream:
        # streaming: concatenate response tokens
        text = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                text += chunk.get("response", "")
                if chunk.get("done"):
                    break
        return text.strip()
    else:
        data = response.json()
        return data.get("response", "").strip()
