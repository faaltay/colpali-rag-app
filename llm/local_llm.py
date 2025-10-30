# Local LLM adapter. If LOCAL_LLM_URL is set, POST {"prompt": "..."} and expect {"text": "..."}.
import os
import requests
from typing import Dict

LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:8080/generate")

class LocalLLM:
    def __init__(self, url: str = None):
        self.url = url or LOCAL_LLM_URL

    def generate(self, prompt: str, **kwargs) -> str:
        # If the user configured a real local model server, call it
        try:
            resp = requests.post(self.url, json={"prompt": prompt, **kwargs}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            # accept different shapes
            if isinstance(data, dict) and "text" in data:
                return data["text"]
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"]
            if isinstance(data, str):
                return data
            return str(data)
        except Exception:
            # fallback: simple deterministic mock reply for dev/testing
            return f"[mock reply] Received prompt of length {len(prompt)}"