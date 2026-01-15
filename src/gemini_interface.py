# src/gemini_interface.py
import os
import random

class GeminiClient:
    _warned = False  # class-level flag

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")

    def propose_belief(self, prompt: str) -> float:
        if not self.api_key:
            return self._fallback("API key not set")

        try:
            # Real API call would go here
            raise RuntimeError("Quota exceeded")  # simulate failure
        except Exception as e:
            return self._fallback(str(e))

    def _fallback(self, reason: str) -> float:
        if not GeminiClient._warned:
            print("[Gemini] API unavailable — using fallback belief.")
            GeminiClient._warned = True
        return 0.5
